from __future__ import annotations

import shutil

import pytest
from typer.testing import CliRunner

from codebase_ai.cli import app
from codebase_ai.config import get_settings
from codebase_ai.ingest.git_source import GitSourceError, RemoteRepo, clone_dir_for, parse_git_url
from helpers import FIXTURE_REPO, FakeEmbedder, ScriptedProvider

runner = CliRunner()
URL = "https://github.com/acme/shop"


@pytest.fixture
def git(monkeypatch):
    """Replace the network step with a copy of the sample repo, and record what was asked of it."""

    class Fake:
        calls: list[dict]
        error: Exception | None = None

    fake = Fake()
    fake.calls = []

    def sync(remote: RemoteRepo, root, *, branch=None, **kwargs):
        fake.calls.append({"remote": remote, "root": root, "branch": branch})
        if fake.error is not None:
            raise fake.error
        target = clone_dir_for(remote, root)
        if not target.exists():
            shutil.copytree(FIXTURE_REPO, target)
        return target

    monkeypatch.setattr("codebase_ai.ingest.git_source.sync_repo", sync)
    monkeypatch.setattr("codebase_ai.index.embedder.create_embedder", lambda settings: FakeEmbedder())
    return fake


def test_index_clones_then_indexes_and_says_so(git):
    result = runner.invoke(app, ["index", URL])
    assert result.exit_code == 0, result.output
    assert "Fetching https://github.com/acme/shop" in result.output and "Files indexed" in result.output
    (call,) = git.calls
    assert call["remote"].display == URL and call["branch"] is None


def test_a_branch_is_passed_on(git):
    assert runner.invoke(app, ["index", URL, "--branch", "release/2"]).exit_code == 0
    assert git.calls[0]["branch"] == "release/2"


def test_a_branch_makes_no_sense_for_a_local_folder(git, sample_repo):
    result = runner.invoke(app, ["index", str(sample_repo), "--branch", "dev"])
    assert result.exit_code == 2 and "--branch only applies to a git URL" in result.output
    assert git.calls == []


def test_other_commands_use_the_clone_and_never_touch_the_network(git):
    assert runner.invoke(app, ["index", URL]).exit_code == 0
    git.calls.clear()
    stats = runner.invoke(app, ["stats", URL])
    assert stats.exit_code == 0 and "Chunks" in stats.output
    search = runner.invoke(app, ["search", URL, "shopping cart total price", "-k", "2"])
    assert search.exit_code == 0 and "web/cart.js" in search.output
    assert git.calls == []


def test_ask_works_on_a_cloned_url(git, monkeypatch):
    assert runner.invoke(app, ["index", URL]).exit_code == 0
    llm = ScriptedProvider("cartTotal adds up the prices [1].")
    monkeypatch.setattr("codebase_ai.llm.base.create_provider", lambda settings, *, provider=None, model=None: llm)
    result = runner.invoke(app, ["ask", URL, "shopping cart total price"])
    assert result.exit_code == 0, result.output
    assert "cartTotal adds up" in result.output and "[1] web/cart.js:" in result.output


def test_a_url_that_was_never_cloned_says_how_to_clone_it(git):
    for command in (["stats", URL], ["search", URL, "x"], ["ask", URL, "x"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 1, command
        text = " ".join(result.output.split())  # the terminal wraps long lines
        assert "has not been cloned yet" in text and f"codebase-ai index {URL}" in text
    assert git.calls == []


def test_a_cloned_but_unindexed_url_says_how_to_index_it(git):
    clone_dir_for(parse_git_url(URL), get_settings().index_dir).mkdir(parents=True)
    result = runner.invoke(app, ["stats", URL])
    assert result.exit_code == 1 and f"No index for {URL}. Run: codebase-ai index {URL}" in " ".join(result.output.split())


@pytest.mark.parametrize(
    "bad",
    ["http://github.com/acme/shop", "git://github.com/acme/shop", "file:///etc", "ext::sh -c id", "-oProxyCommand=x"],
)
def test_unsupported_urls_are_usage_errors(git, bad):
    result = runner.invoke(app, ["index", bad])
    assert result.exit_code == 2 and git.calls == []


def test_a_url_with_a_token_is_refused_and_the_token_is_never_printed(git):
    result = runner.invoke(app, ["index", "https://ghp_supersecrettoken@github.com/acme/shop"])
    assert result.exit_code == 2 and "credentials" in result.output
    assert "ghp_supersecrettoken" not in result.output and git.calls == []


def test_a_failed_clone_is_a_plain_message_and_exit_code_1(git):
    git.error = GitSourceError("The repository was not found, or you do not have access to it.")
    result = runner.invoke(app, ["index", URL])
    assert result.exit_code == 1 and "not found, or you do not have access" in result.output
    assert "Traceback" not in result.output


def test_serve_validates_a_url_but_does_not_clone_it(git, monkeypatch):
    launched = []
    monkeypatch.setattr("subprocess.call", lambda command, env=None: launched.append(env) or 0)
    assert runner.invoke(app, ["serve", URL]).exit_code == 0
    assert launched[0]["CODEBASE_AI_REPO"] == URL and git.calls == []
    assert runner.invoke(app, ["serve", "http://github.com/acme/shop"]).exit_code == 2
    assert len(launched) == 1
