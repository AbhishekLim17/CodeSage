from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from codebase_ai.ingest.git_source import (
    ALLOWED_PROTOCOLS,
    GitSourceError,
    RemoteRepo,
    clone_dir_for,
    looks_like_url,
    parse_git_url,
    sync_repo,
)


class TestParseGitUrl:
    @pytest.mark.parametrize(
        ("text", "display", "name"),
        [
            ("https://github.com/owner/repo", "https://github.com/owner/repo", "repo"),
            ("https://github.com/owner/repo.git", "https://github.com/owner/repo", "repo"),
            ("  https://github.com/owner/repo/  ", "https://github.com/owner/repo", "repo"),
            ("https://git.example.com:8443/team/sub/project.git", "https://git.example.com:8443/team/sub/project", "project"),
            ("git@github.com:owner/repo.git", "git@github.com:owner/repo", "repo"),
            ("ssh://git@github.com/owner/repo.git", "ssh://git@github.com/owner/repo", "repo"),
        ],
    )
    def test_accepted_forms(self, text, display, name):
        remote = parse_git_url(text)
        assert (remote.display, remote.name) == (display, name)

    def test_the_url_given_to_git_keeps_its_own_spelling(self):
        assert parse_git_url("https://github.com/owner/repo.git").clone_url == "https://github.com/owner/repo.git"
        assert parse_git_url("git@github.com:owner/repo.git").clone_url == "git@github.com:owner/repo.git"

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("http://github.com/owner/repo", "not supported"),
            ("git://github.com/owner/repo", "not supported"),
            ("file:///etc/passwd", "not supported"),
            ("ftp://host/repo", "not supported"),
            ("ext::sh -c 'touch /tmp/pwned'", "not a repository URL"),
            ("ext::sh -c touch% /tmp/pwned", "not a repository URL"),
            ("/home/me/repo", "https:// and ssh"),
            ("C:\\code\\repo", "https:// and ssh"),
            ("owner/repo", "https:// and ssh"),
            ("-oProxyCommand=touch /tmp/x", "not a repository URL"),
            ("--upload-pack=touch /tmp/x", "not a repository URL"),
            ("git@-oProxyCommand:owner/repo", "not a repository URL"),
            ("https://github.com/owner/repo with space", "not a repository URL"),
            ("", "not a repository URL"),
            ("https://github.com", "no repository path"),
            ("https:///owner/repo", "not a repository URL"),
            ("https://-evil.example/owner/repo", "not a repository URL"),
        ],
    )
    def test_refused_forms(self, text, reason):
        with pytest.raises(GitSourceError, match=reason):
            parse_git_url(text)

    @pytest.mark.parametrize(
        "text",
        [
            "https://user:hunter2@github.com/owner/repo",
            "https://ghp_abcdef1234567890@github.com/owner/repo",
            "ssh://user:secret@host/owner/repo",
        ],
    )
    def test_urls_with_credentials_are_refused_without_echoing_them(self, text):
        with pytest.raises(GitSourceError) as caught:
            parse_git_url(text)
        assert "credentials" in str(caught.value)
        assert "hunter2" not in str(caught.value) and "secret" not in str(caught.value) and "ghp_" not in str(caught.value)

    def test_the_key_ignores_case_and_a_dot_git_suffix_but_tells_repositories_apart(self):
        a = parse_git_url("https://GitHub.com/Owner/Repo.git")
        b = parse_git_url("https://github.com/owner/repo")
        c = parse_git_url("https://github.com/owner/other")
        assert a.key == b.key and a.key != c.key
        assert a.key.startswith("repo-")

    def test_hostile_names_become_safe_folder_names(self):
        remote = parse_git_url("https://github.com/owner/we ird" .replace(" ", "%20"))
        assert "/" not in remote.key and "\\" not in remote.key and ".." not in remote.key
        assert parse_git_url("https://github.com/owner/..").name == "repo"


class TestLooksLikeUrl:
    @pytest.mark.parametrize(
        "text",
        [
            "https://github.com/o/r",
            "http://github.com/o/r",
            "file:///tmp/x",
            "git@github.com:o/r.git",
            "ext::sh -c id",
            "ssh://git@host/o/r",
        ],
    )
    def test_remotes(self, text):
        assert looks_like_url(text)

    @pytest.mark.parametrize("text", ["/home/me/repo", "C:\\code\\repo", "C:/code/repo", "./repo", "..", "repo", "~/code/repo"])
    def test_local_paths(self, text):
        assert not looks_like_url(text)


class FakeGit:
    """Records every git call and answers from a script; stands in for ``subprocess.run``."""

    def __init__(self, *, returncode=0, stdout="", stderr="", raises=None, origin=None):
        self.calls: list[dict] = []
        self.returncode, self.stdout, self.stderr, self.raises, self.origin = returncode, stdout, stderr, raises, origin

    def __call__(self, command, **kwargs):
        self.calls.append({"command": command, **kwargs})
        if self.raises:
            raise self.raises
        if "get-url" in command and self.origin is not None:
            return subprocess.CompletedProcess(command, 0, stdout=self.origin + "\n", stderr="")
        return subprocess.CompletedProcess(command, self.returncode, stdout=self.stdout, stderr=self.stderr)


REMOTE = RemoteRepo("https://github.com/owner/repo", "https://github.com/owner/repo.git", "repo")


class TestSyncWithAFakeGit:
    def test_a_first_run_makes_a_shallow_locked_down_clone(self, tmp_path):
        git = FakeGit()
        target = sync_repo(REMOTE, tmp_path, runner=git)
        assert target == clone_dir_for(REMOTE, tmp_path) == tmp_path / "clones" / REMOTE.key
        (call,) = git.calls
        command = call["command"]
        assert command[0] == "git" and "clone" in command
        for flag in ("--depth", "--no-tags", "--single-branch", "--no-recurse-submodules", "--template="):
            assert flag in command
        assert command[command.index("--depth") + 1] == "1"
        assert "protocol.ext.allow=never" in command
        assert any(part.startswith("core.hooksPath=") for part in command)
        assert command[-3:] == ["--", REMOTE.clone_url, str(target)]  # the URL can only ever be read as a URL

    def test_git_is_told_never_to_prompt_and_only_to_speak_https_and_ssh(self, tmp_path):
        git = FakeGit()
        sync_repo(REMOTE, tmp_path, runner=git)
        env = git.calls[0]["env"]
        assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_ALLOW_PROTOCOL"] == ALLOWED_PROTOCOLS == "https:ssh"
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert git.calls[0]["timeout"] == 600 and git.calls[0]["check"] is False

    def test_a_branch_is_passed_as_a_branch_and_must_look_like_one(self, tmp_path):
        git = FakeGit()
        sync_repo(REMOTE, tmp_path, branch="release/1.2", runner=git)
        command = git.calls[0]["command"]
        assert command[command.index("--branch") + 1] == "release/1.2"
        for bad in ("--upload-pack=x", "-x", "a b", "a;b", "", "$(id)"):
            with pytest.raises(GitSourceError, match="branch"):
                sync_repo(REMOTE, tmp_path, branch=bad, runner=git)

    def test_an_existing_clone_is_updated_in_place_and_never_deleted(self, tmp_path):
        target = clone_dir_for(REMOTE, tmp_path)
        (target / ".git").mkdir(parents=True)
        marker = target / "keep.txt"
        marker.write_text("mine", encoding="utf-8")
        git = FakeGit(origin=REMOTE.clone_url)
        assert sync_repo(REMOTE, tmp_path, runner=git) == target
        verbs = [next(a for a in c["command"] if a in ("remote", "fetch", "reset", "clone")) for c in git.calls]
        assert verbs == ["remote", "fetch", "reset"]
        assert marker.read_text(encoding="utf-8") == "mine"
        fetch = git.calls[1]["command"]
        assert "--depth" in fetch and fetch[-2:] == ["origin", "HEAD"] and git.calls[1]["cwd"] == target
        assert git.calls[2]["command"][-3:] == ["reset", "--hard", "FETCH_HEAD"]

    def test_updating_a_branch_fetches_that_branch(self, tmp_path):
        target = clone_dir_for(REMOTE, tmp_path)
        (target / ".git").mkdir(parents=True)
        git = FakeGit(origin=REMOTE.clone_url)
        sync_repo(REMOTE, tmp_path, branch="dev", runner=git)
        assert git.calls[1]["command"][-2:] == ["origin", "dev"]

    def test_a_folder_in_the_way_is_reported_not_removed(self, tmp_path):
        target = clone_dir_for(REMOTE, tmp_path)
        target.mkdir(parents=True)
        (target / "notes.txt").write_text("not a clone", encoding="utf-8")
        git = FakeGit()
        with pytest.raises(GitSourceError, match="not a git clone made by this tool"):
            sync_repo(REMOTE, tmp_path, runner=git)
        assert (target / "notes.txt").exists() and git.calls == []

    def test_a_clone_of_a_different_repository_is_refused(self, tmp_path):
        target = clone_dir_for(REMOTE, tmp_path)
        (target / ".git").mkdir(parents=True)
        with pytest.raises(GitSourceError, match="different repository"):
            sync_repo(REMOTE, tmp_path, runner=FakeGit(origin="https://github.com/someone/else.git"))

    @pytest.mark.parametrize(
        ("stderr", "expected"),
        [
            ("fatal: could not read Username for 'https://github.com': terminal prompts disabled", "could not authenticate"),
            ("fatal: unable to access 'https://x/': The requested URL returned error: 403", "could not authenticate"),
            ("remote: Repository not found.\nfatal: repository 'https://x/' not found", "not found"),
            ("fatal: unable to access 'https://x/': Could not resolve host: x", "Could not reach the server"),
            ("fatal: Remote branch nope not found in upstream origin", "branch does not exist"),
            ("fatal: something unexpected happened", "Git failed: fatal: something unexpected happened"),
        ],
    )
    def test_git_failures_become_plain_messages(self, tmp_path, stderr, expected):
        with pytest.raises(GitSourceError, match=expected):
            sync_repo(REMOTE, tmp_path, runner=FakeGit(returncode=128, stderr=stderr))

    def test_git_missing_and_git_hanging_are_reported(self, tmp_path):
        with pytest.raises(GitSourceError, match="git is not installed"):
            sync_repo(REMOTE, tmp_path, runner=FakeGit(raises=FileNotFoundError()))
        with pytest.raises(GitSourceError, match="longer than 5 seconds"):
            sync_repo(REMOTE, tmp_path / "b", timeout=5, runner=FakeGit(raises=subprocess.TimeoutExpired("git", 5)))


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestWithRealGit:
    """A real clone of a local repository: the command line is right and the result is usable."""

    @staticmethod
    def git(*args, cwd):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=cwd, check=True, capture_output=True, text=True,
        )

    @pytest.fixture
    def origin(self, tmp_path) -> Path:
        origin = tmp_path / "origin"
        origin.mkdir()
        self.git("init", "-q", "-b", "main", cwd=origin)
        (origin / "app.py").write_text('"""Demo app."""\n\ndef hello():\n    return 1\n', encoding="utf-8")
        self.git("add", ".", cwd=origin)
        self.git("commit", "-q", "-m", "first", cwd=origin)
        self.git("commit", "-q", "--allow-empty", "-m", "second", cwd=origin)
        return origin

    def remote_for(self, origin: Path) -> RemoteRepo:
        url = origin.as_uri()
        return RemoteRepo(url, url, "origin")

    def test_clone_is_shallow_has_the_files_and_no_hooks(self, origin, tmp_path):
        root = tmp_path / "index-root"
        target = sync_repo(self.remote_for(origin), root, protocols="file")
        assert (target / "app.py").read_text(encoding="utf-8").startswith('"""Demo app."""')
        assert (target / ".git" / "shallow").exists()  # only the latest commit
        hooks = target / ".git" / "hooks"
        assert not hooks.exists() or not any(hooks.iterdir())  # no sample hooks: --template= gave it none

    def test_a_second_run_picks_up_new_commits_and_leaves_the_folder_alone(self, origin, tmp_path):
        root = tmp_path / "index-root"
        remote = self.remote_for(origin)
        target = sync_repo(remote, root, protocols="file")
        (origin / "second.py").write_text("x = 2\n", encoding="utf-8")
        self.git("add", ".", cwd=origin)
        self.git("commit", "-q", "-m", "third", cwd=origin)
        assert sync_repo(remote, root, protocols="file") == target
        assert (target / "second.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_the_default_protocols_refuse_a_local_repository(self, origin, tmp_path):
        with pytest.raises(GitSourceError):
            sync_repo(self.remote_for(origin), tmp_path / "index-root")  # https:ssh only: file:// is not allowed

    def test_an_unknown_branch_is_a_plain_error(self, origin, tmp_path):
        with pytest.raises(GitSourceError, match="branch does not exist"):
            sync_repo(self.remote_for(origin), tmp_path / "index-root", branch="nope", protocols="file")

    def test_the_clone_can_be_indexed_like_any_local_repository(self, origin, tmp_path, fake_embedder):
        from codebase_ai.index.indexer import Indexer, RepoIndex

        target = sync_repo(self.remote_for(origin), tmp_path / "index-root", protocols="file")
        index = RepoIndex(target, tmp_path / "idx")
        try:
            report = Indexer(index, fake_embedder).run()
            assert report.files_indexed == 1 and index.keyword.count() >= 1  # .git is never walked
        finally:
            index.close()
