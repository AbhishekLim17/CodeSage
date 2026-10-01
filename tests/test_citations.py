from __future__ import annotations

import pytest

from codebase_ai.rag.citations import CitationReport, validate_citations
from codebase_ai.retrieval.retriever import Source


def span(path: str, start: int, end: int) -> Source:
    return Source(
        path=path,
        language="python",
        start_line=start,
        end_line=end,
        text="x = 1",
        symbols=(),
        kinds=("function",),
        score=1.0,
        rank=1,
        chunk_ids=("c",),
    )


SOURCES = [span("app/users.py", 10, 40), span("app/auth.py", 1, 30), span("web/api.js", 5, 25)]


def check(text: str, sources=SOURCES) -> CitationReport:
    return validate_citations(text, sources)


def test_single_marker():
    report = check("Users are normalized on save [1].")
    assert report.cited == (1,)
    assert report.invalid == ()
    assert report.clean_text == "Users are normalized on save [1]."
    assert report.marker_count == 1


@pytest.mark.parametrize(
    ("text", "cited"),
    [
        ("See [1, 2].", (1, 2)),
        ("See [1,3].", (1, 3)),
        ("See [1 and 2].", (1, 2)),
        ("See [1-3].", (1, 2, 3)),
        ("See [2–3].", (2, 3)),
        ("See [Source 2].", (2,)),
        ("See [sources 1, 3].", (1, 3)),
        ("See [1][2].", (1, 2)),
        ("See [3] then [1] then [3] again.", (3, 1)),
    ],
)
def test_marker_forms(text, cited):
    report = check(text)
    assert report.cited == cited
    assert report.invalid == ()
    assert report.clean_text == text


def test_uncited_sources_are_listed():
    report = check("Only the first matters [1].")
    assert report.uncited == (2, 3)
    assert report.grounded


def test_no_citations_means_not_grounded():
    report = check("It probably works like most frameworks do.")
    assert not report.grounded
    assert report.cited == ()
    assert report.validity is None
    assert report.uncited == (1, 2, 3)


def test_marker_to_a_missing_source_is_removed_and_reported():
    report = check("It saves the user [1] and emails them [7].")
    assert report.cited == (1,)
    assert report.invalid == (7,)
    assert report.clean_text == "It saves the user [1] and emails them."
    assert report.validity == 0.5


def test_partly_valid_marker_keeps_the_valid_numbers():
    report = check("Both do it [2, 9].")
    assert report.cited == (2,)
    assert report.invalid == (9,)
    assert report.clean_text == "Both do it [2]."


def test_zero_is_not_a_source():
    report = check("Nothing here [0].")
    assert report.invalid == (0,)
    assert report.clean_text == "Nothing here."


def test_only_invalid_markers_leave_it_ungrounded_with_zero_validity():
    report = check("Claim [8].")
    assert not report.grounded
    assert report.validity == 0.0


def test_invalid_then_valid_chain_keeps_the_space():
    assert check("Claim [9][1].").clean_text == "Claim [1]."


def test_an_absurd_range_is_not_expanded():
    report = check("See [1-999999].")
    assert report.cited == (1,)
    assert report.invalid == (999999,)


def test_a_reversed_range_is_read_as_its_endpoints():
    report = check("See [3-1].")
    assert set(report.cited) == {1, 3}


class TestBracketsThatAreNotCitations:
    def test_array_indexing(self):
        report = check("It reads items[1] and rows[2][3] from the list.")
        assert report.cited == ()
        assert report.invalid == ()
        assert report.marker_count == 0
        assert report.clean_text == "It reads items[1] and rows[2][3] from the list."

    def test_call_result_indexing(self):
        assert check("It uses parse(x)[1] here.").marker_count == 0

    def test_inline_code(self):
        report = check("The value `args[9]` is a `[7]` literal.")
        assert report.invalid == ()
        assert report.clean_text == "The value `args[9]` is a `[7]` literal."

    def test_fenced_code(self):
        text = "Example:\n```python\nx = data [9]\ny = z [2]\n```\nDone [1]."
        report = check(text)
        assert report.cited == (1,)
        assert report.invalid == ()
        assert "data [9]" in report.clean_text

    def test_markers_after_code_are_still_read(self):
        report = check("Use `items[1]` first [2].")
        assert report.cited == (2,)

    def test_a_chain_after_indexing_is_not_extended_from_the_index(self):
        # ``matrix[1][2]`` is indexing; the citation that follows the sentence is still found.
        report = check("Read matrix[1][2] here [3].")
        assert report.cited == (3,)
        assert report.marker_count == 1
        assert report.clean_text == "Read matrix[1][2] here [3]."

    def test_markdown_link_text_is_not_a_marker(self):
        assert check("See [the docs](http://x.y) for more.").marker_count == 0


class TestLocations:
    def test_location_inside_a_source_is_verified(self):
        report = check("It lives in app/users.py:12-20 [1].")
        assert report.unverified_locations == ()

    def test_single_line_location(self):
        assert check("See app/auth.py:5 [2].").unverified_locations == ()

    def test_a_file_that_was_never_shown_is_flagged(self):
        report = check("It lives in app/billing.py:12-20.")
        assert report.unverified_locations == ("app/billing.py:12-20",)

    def test_lines_outside_the_shown_span_are_flagged(self):
        report = check("It lives in app/users.py:100-120 [1].")
        assert report.unverified_locations == ("app/users.py:100-120",)

    def test_a_span_that_starts_inside_but_runs_past_the_end_is_flagged(self):
        assert check("See app/users.py:30-50.").unverified_locations == ("app/users.py:30-50",)

    def test_a_reversed_range_is_flagged(self):
        assert check("See app/users.py:30-20.").unverified_locations == ("app/users.py:30-20",)

    def test_a_shorter_path_matches_by_suffix(self):
        assert check("See users.py:12-20 [1].").unverified_locations == ()

    def test_a_partial_directory_name_does_not_match(self):
        sources = [span("app/users.py", 1, 50)]
        assert validate_citations("See pp/users.py:12-20.", sources).unverified_locations == ("pp/users.py:12-20",)

    def test_repeated_locations_are_reported_once(self):
        text = "See app/billing.py:1-2 and again app/billing.py:1-2."
        assert check(text).unverified_locations == ("app/billing.py:1-2",)

    def test_locations_inside_fenced_code_are_ignored(self):
        text = "```\nTraceback: app/billing.py:12\n```"
        assert check(text).unverified_locations == ()

    def test_plain_numbers_and_urls_are_not_locations(self):
        text = "Listens on localhost:8080 and example.com:443, version 1.2:3."
        assert check(text).unverified_locations == ()

    def test_a_location_with_no_sources_at_all_is_unverified(self):
        assert validate_citations("See app/users.py:1-2.", []).unverified_locations == ("app/users.py:1-2",)


def test_no_sources_at_all():
    report = validate_citations("Claim [1].", [])
    assert report.invalid == (1,)
    assert report.cited == ()
    assert report.uncited == ()
    assert report.clean_text == "Claim."


def test_empty_text():
    report = check("")
    assert report.clean_text == ""
    assert report.marker_count == 0
    assert report.uncited == (1, 2, 3)


def test_marker_at_the_start_of_a_line():
    report = check("[1] does it.\n[2] too.")
    assert report.cited == (1, 2)
    assert report.clean_text == "[1] does it.\n[2] too."


def test_marker_counts_each_number():
    assert check("A [1][2] and [1, 3].").marker_count == 4
