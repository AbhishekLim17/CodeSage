from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.retrieval.overview import OverviewRetriever
from codebase_ai.retrieval.repo_map import (
    MAP_PATH,
    build_repo_map,
    describe_file,
    map_source,
    readme_source,
)
from codebase_ai.retrieval.retriever import Retriever
from helpers import FakeEmbedder


def make_repo(root: Path, files: dict[str, str]) -> Path:
    repo = root / "proj"
    for relative, text in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return repo


@pytest.fixture
def build(tmp_path):
    opened: list[RepoIndex] = []

    def _build(files: dict[str, str]) -> RepoIndex:
        repo = make_repo(tmp_path, files)
        index = RepoIndex(repo, tmp_path / f"idx{len(opened)}")
        Indexer(index, FakeEmbedder()).run()
        opened.append(index)
        return index

    yield _build
    for index in opened:
        index.close()


class TestDescribeFile:
    @pytest.mark.parametrize(
        ("head", "language", "expected"),
        [
            ('"""Turn files into chunks.\n\nMore detail here."""\nimport os\n', "python", "Turn files into chunks."),
            ('"""\nSecond-line summary.\n"""\n', "python", "Second-line summary."),
            ('# a comment\n"""Docstring after a comment."""\n', "python", "Docstring after a comment."),
            ("import os\n\nprint(1)\n", "python", None),
            ("# just a comment, no docstring\nimport os\n", "python", None),
            ("/**\n * Gateway for the cloud functions.\n * @module gateway\n */\nexport const x = 1;\n", "javascript", "Gateway for the cloud functions."),
            ("/* Seeds the emulators. */\nconst a = 1;\n", "javascript", "Seeds the emulators."),
            ("// Handles the request queue.\n// Second line.\nfunc main() {}\n", "go", "Handles the request queue."),
            ("'use strict';\n\n// Shared mailer.\nmodule.exports = {};\n", "javascript", "Shared mailer."),
            ("/**\n * Copyright 2024 Someone. All rights reserved.\n */\nconst a = 1;\n", "javascript", None),
            ("// @ts-check\nconst a = 1;\n", "javascript", None),
            ("#!/usr/bin/env node\n// Runs the job.\n", "javascript", "Runs the job."),
            ("# Sample Repo\n\nIntro.\n", "markdown", "Sample Repo"),
            ("import { a } from './a';\nexport const b = a;\n", "javascript", None),
            ("", "python", None),
        ],
    )
    def test_first_meaningful_line(self, head, language, expected):
        assert describe_file(head, language) == expected

    def test_a_long_sentence_is_shortened(self):
        described = describe_file('"""' + "word " * 100 + '"""', "python")
        assert described is not None and len(described) <= 110 and described.endswith("…")


class TestBuildRepoMap:
    FILES: ClassVar[dict[str, str]] = {
        "README.md": "# Proj\n\nA demo project that does things.\n",
        "pyproject.toml": "[project]\nname='proj'\n",
        "src/proj/__init__.py": '"""The proj package: does the things."""\n',
        "src/proj/core/__init__.py": '"""Core logic."""\n',
        "src/proj/core/engine.py": '"""The engine that runs everything."""\n\nclass Engine:\n    pass\n\ndef _private():\n    pass\n\ndef start():\n    pass\n',
        "src/proj/api/__init__.py": '"""HTTP API."""\n',
        "src/proj/api/routes.py": 'def route_users():\n    return 1\n',
        "tests/test_engine.py": "def test_engine():\n    assert True\n",
        "docs/guide.md": "# The Guide\n\nHow to use it.\n",
    }

    def test_lists_directories_with_counts_languages_purpose_and_definitions(self, build):
        repo_map = build_repo_map(build(self.FILES))
        text = repo_map.text
        assert text.startswith("Repository map (generated from the index; it is not a source file).")
        assert "src/proj/core/  (2 files, python) - Core logic.  defines Engine, start" in text
        assert "src/proj/api/  (2 files, python) - HTTP API.  defines route_users" in text

    def test_a_chain_of_single_child_directories_is_one_entry(self, build):
        text = build_repo_map(build(self.FILES)).text
        assert "src/proj/  (" in text
        assert "\nsrc/  (" not in text  # src/ only contains proj/, so the entry is src/proj/

    def test_private_names_are_not_listed(self, build):
        assert "_private" not in build_repo_map(build(self.FILES)).text

    def test_tests_are_named_but_not_described(self, build):
        text = build_repo_map(build(self.FILES)).text
        assert "Tests: tests/ (1 file)." in text
        assert "test_engine" not in text

    def test_documentation_titles_are_listed_and_the_readme_is_left_to_its_own_source(self, build):
        text = build_repo_map(build(self.FILES)).text
        assert "docs/guide.md - The Guide" in text
        assert "Documentation files" in text
        assert "  README.md -" not in text

    def test_code_files_that_describe_themselves_are_listed_but_not_package_inits(self, build):
        text = build_repo_map(build(self.FILES)).text
        assert "src/proj/core/engine.py - The engine that runs everything." in text
        assert "__init__.py -" not in text

    def test_top_level_files_and_language_counts(self, build):
        text = build_repo_map(build(self.FILES)).text
        assert "Files at the top level: README.md, pyproject.toml." in text
        assert "python 6" in text

    def test_reports_which_directories_it_names(self, build):
        repo_map = build_repo_map(build(self.FILES))
        assert repo_map.lists("src/proj/core") and repo_map.lists("src/proj") and repo_map.lists("docs/")
        assert repo_map.lists("src")  # part of a collapsed entry
        assert not repo_map.lists("nowhere")
        assert repo_map.files == 9 and repo_map.readme_path == "README.md"

    def test_stays_within_its_budget(self, build):
        files = {f"pkg{n}/mod{m}.py": f'"""Module {n}-{m} does thing number {n}{m}."""\n\ndef f{n}_{m}():\n    pass\n' for n in range(30) for m in range(6)}
        index = build(files)
        small = build_repo_map(index, budget_tokens=400)
        large = build_repo_map(index, budget_tokens=3000)
        assert small.tokens <= 400 * 1.15
        assert large.tokens > small.tokens
        assert len(small.directories) < len(large.directories)

    def test_an_empty_index_is_reported_plainly(self, build):
        repo_map = build_repo_map(build({"blob.bin": "\x00\x01"}))
        assert repo_map.files == 0 and "empty" in repo_map.text

    def test_the_text_contains_no_square_bracket_numbers_that_look_like_citations(self, build):
        import re

        assert not re.search(r"\[\d+\]", build_repo_map(build(self.FILES)).text)


class TestMapSources:
    def test_the_map_is_a_numbered_source_that_is_not_a_file(self, build):
        repo_map = build_repo_map(build(TestBuildRepoMap.FILES))
        source = map_source(repo_map)
        assert source.path == MAP_PATH and source.role == "map" and source.kinds == ("map",)
        assert source.text == repo_map.text
        assert source.location == f"{MAP_PATH}:1-{repo_map.text.count(chr(10)) + 1}"

    def test_the_readme_source_has_exact_lines_and_is_capped(self, build):
        body = "\n".join(f"line {n}" for n in range(1, 200))
        index = build({"README.md": f"# Title\n\n{body}\n", "a.py": "x = 1\n"})
        source = readme_source(index, build_repo_map(index))
        assert source is not None and source.path == "README.md" and source.start_line == 1
        assert source.end_line <= 60
        assert source.text.split("\n")[0] == "# Title"
        assert len(source.text.split("\n")) == source.end_line - source.start_line + 1

    def test_no_readme_means_no_readme_source(self, build):
        index = build({"a.py": "x = 1\n"})
        assert readme_source(index, build_repo_map(index)) is None


class TestOverviewRetriever:
    FILES: ClassVar[dict[str, str]] = {**TestBuildRepoMap.FILES}

    def retriever(self, index, **kwargs) -> OverviewRetriever:
        return OverviewRetriever(Retriever(index, FakeEmbedder()), **kwargs)

    def test_a_question_about_one_thing_is_untouched(self, build):
        index = build(self.FILES)
        plain = Retriever(index, FakeEmbedder()).retrieve("engine start")
        wrapped = self.retriever(index).retrieve("engine start")
        assert [s.location for s in wrapped.sources] == [s.location for s in plain.sources]
        assert not wrapped.overview

    def test_an_overview_question_gets_the_map_and_the_readme_first(self, build):
        result = self.retriever(build(self.FILES)).retrieve("What does this project do?")
        assert result.overview
        assert [s.role for s in result.sources[:2]] == ["map", "match"]
        assert result.sources[0].path == MAP_PATH and result.sources[1].path == "README.md"
        assert len(result.sources) > 2  # ordinary retrieval results follow

    def test_the_readme_is_not_added_twice(self, build):
        index = build(self.FILES)
        retriever = OverviewRetriever(Retriever(index, FakeEmbedder()), is_overview=lambda q: True)
        result = retriever.retrieve("README project demo things")
        assert [s.path for s in result.sources].count("README.md") == 1

    def test_the_total_stays_within_the_budget_but_keeps_one_retrieved_source(self, build):
        index = build(self.FILES)
        base = Retriever(index, FakeEmbedder(), budget_tokens=300)
        result = OverviewRetriever(base, map_tokens=200).retrieve("Give me an overview of the codebase")
        assert result.sources[0].role == "map"
        retrieved = [s for s in result.sources if s.role != "map" and s.path != "README.md"]
        assert retrieved  # at least one ordinary source survives
        assert sum(s.tokens for s in result.sources if s.role != "map" and s.path != "README.md") <= 300

    def test_the_wrapper_exposes_what_callers_need(self, build):
        index = build(self.FILES)
        wrapper = self.retriever(index)
        assert wrapper.mode == "vector" and wrapper.index is index

    def test_a_repository_without_a_readme_still_gets_the_map(self, build):
        result = self.retriever(build({"a.py": '"""Alpha."""\n\ndef a():\n    pass\n'})).retrieve("Describe this repo")
        assert result.overview and result.sources[0].role == "map"
        assert all(s.path != "README.md" for s in result.sources)


class TestFoundOnAnUnfamiliarRepository:
    """Regressions from running the map on psf/requests, which it had not been tuned on."""

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ("Developer Interface\n===================\n\nThis part covers...\n", "Developer Interface"),
            ("=====\nTitle\n=====\n\nBody text here.\n", "Title"),
            ("Requests: HTTP for Humans\n-------------------------\n", "Requests: HTTP for Humans"),
            ("Just a paragraph with no heading at all.\n", None),
            ("", None),
        ],
    )
    def test_restructuredtext_titles_are_found(self, head, expected):
        assert describe_file(head, "rst") == expected

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ('"""\nrequests.api\n~~~~~~~~~~~~\n\nThis module implements the Requests API.\n"""\n', "This module implements the Requests API."),
            ('"""requests.adapters\n\nTransport adapters for Requests.\n"""\n', "Transport adapters for Requests."),
            ('"""\nUtilities\n"""\n', "Utilities"),  # a one-word title with no dot is a real description
            ('"""\nrequests.certs\n"""\n', None),  # a module that only restates its own name says nothing
        ],
    )
    def test_a_docstring_that_starts_with_the_modules_own_name_is_read_past_it(self, head, expected):
        assert describe_file(head, "python") == expected

    def test_text_files_are_not_documentation_and_rst_files_carry_their_titles(self, build):
        files = {
            "README.md": "# Proj\n\nA demo.\n",
            "requirements.txt": "pytest\nruff\n",
            "docs/requirements.txt": "sphinx\n",
            "docs/api.rst": "Developer Interface\n===================\n\nThe API.\n",
            "src/app.py": '"""\napp.core\n~~~~~~~~\n\nThe application core.\n"""\n',
        }
        text = build_repo_map(build(files)).text
        assert "requirements.txt" not in text.split("Documentation files")[-1]
        assert "docs/api.rst - Developer Interface" in text
        assert "src/app.py - The application core." in text
