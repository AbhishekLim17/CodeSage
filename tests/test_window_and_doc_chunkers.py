from __future__ import annotations

import pytest

from codebase_ai.chunking import chunk_file
from codebase_ai.chunking.window_chunker import chunk_windows, window_ranges
from helpers import chunk_fixture, line_count, line_of, make_source


def test_window_ranges_overlap_and_end_exactly_at_the_last_line():
    assert window_ranges(0, 24, 10, 3) == [(0, 9), (7, 16), (14, 23), (21, 24)]


def test_window_ranges_short_range_is_one_window():
    assert window_ranges(5, 8, 60, 10) == [(5, 8)]


@pytest.mark.parametrize(("window", "overlap"), [(0, 0), (10, 10), (10, -1), (5, 9)])
def test_window_ranges_rejects_bad_parameters(window, overlap):
    with pytest.raises(ValueError):
        window_ranges(0, 20, window, overlap)


def test_chunk_windows_cover_every_line_and_report_1_based_lines():
    text = "\n".join(f"line {i}" for i in range(1, 26))
    chunks = chunk_windows(make_source(text, "data.txt"), "repo", window_lines=10, window_overlap=3)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (8, 17), (15, 24), (22, 25)]
    assert chunks[1].text.split("\n")[0] == "line 8"
    assert all(c.kind == "window" and c.symbol is None for c in chunks)


def test_chunk_windows_skip_blank_windows_and_empty_files():
    text = "first\n" + "\n" * 30 + "last"
    chunks = chunk_windows(make_source(text, "x.txt"), "repo", window_lines=5, window_overlap=1)
    assert chunks[0].start_line == 1
    assert all(c.text.strip() for c in chunks)
    assert chunk_windows(make_source("\n\n", "x.txt"), "repo") == []


def test_yaml_config_becomes_a_single_window():
    (chunk,) = chunk_fixture("config/settings.yaml")
    assert (chunk.start_line, chunk.end_line) == (1, line_count("config/settings.yaml"))
    assert chunk.kind == "window" and chunk.language == "yaml"


def test_unsupported_code_language_falls_back_to_windows():
    source = make_source("fn main() {\n    println!(\"hi\");\n}\n", "main.rs")
    chunks = chunk_file(source, "repo")
    assert [c.kind for c in chunks] == ["window"]
    assert chunks[0].language == "rust"


def test_markdown_sections_use_heading_paths():
    chunks = chunk_fixture("README.md")
    assert [c.symbol for c in chunks] == [
        "Sample Repo",
        "Sample Repo > Setup",
        "Sample Repo > Usage > CLI",
        "Sample Repo > Usage > API",
    ]
    assert all(c.kind == "doc_section" for c in chunks)


def test_markdown_ignores_hash_lines_inside_code_fences():
    setup = next(c for c in chunk_fixture("README.md") if c.symbol == "Sample Repo > Setup")
    assert "# this hash is a shell comment" in setup.text
    assert setup.start_line == line_of("README.md", "## Setup")
    assert setup.end_line == line_of("README.md", "make install") + 1  # the closing fence


def test_markdown_heading_without_body_merges_into_its_first_child():
    usage = next(c for c in chunk_fixture("README.md") if c.symbol == "Sample Repo > Usage > CLI")
    assert usage.start_line == line_of("README.md", "## Usage")
    assert usage.text.startswith("## Usage")
    assert "### CLI" in usage.text and "Use the command line interface." in usage.text


def test_markdown_chunks_have_no_leading_or_trailing_blank_lines():
    for chunk in chunk_fixture("README.md"):
        lines = chunk.text.split("\n")
        assert lines[0].strip() and lines[-1].strip()


def test_markdown_oversized_section_is_split_into_windows():
    body = "\n".join(f"paragraph {i}" for i in range(300))
    chunks = chunk_file(make_source(f"# Big\n\n{body}\n", "big.md"), "repo", max_lines=100)
    assert len(chunks) > 2
    assert all(c.symbol == "Big" for c in chunks)
    assert all(c.end_line - c.start_line + 1 <= 100 for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 302


def test_markdown_without_headings_is_one_chunk():
    (chunk,) = chunk_file(make_source("just text\nmore text\n", "notes.md"), "repo")
    assert chunk.symbol is None and (chunk.start_line, chunk.end_line) == (1, 2)


def test_markdown_trailing_heading_only_section_is_kept():
    chunks = chunk_file(make_source("# A\n\ntext\n\n## Empty\n", "a.md"), "repo")
    assert [c.symbol for c in chunks] == ["A", "A > Empty"]
