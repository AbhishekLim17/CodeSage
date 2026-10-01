"""Index a repository from a git URL: a shallow clone into a folder this tool manages.

A URL typed by a user is untrusted input handed to ``git``, which can run programs (``ext::`` transports, hooks, filters)
and read local files (``file://``). So this module is deliberately narrow:

* only ``https://host/path`` and ``ssh`` URLs are accepted (``user@host:path``, ``ssh://...``); plain ``http://``,
  ``git://``, ``file://``, ``ext::`` and local paths are refused, and ``GIT_ALLOW_PROTOCOL`` says the same to git itself;
* a URL that carries a password or token is refused (it would be written to ``.git/config`` and to logs); private
  repositories work through git's own credential helper or an ssh key, never through this tool;
* nothing user-supplied can be read as an option: the URL follows ``--``, and a branch name must look like a branch name;
* the clone has no templates or hooks (``--template=`` plus an empty hooks path), no submodules and no LFS downloads,
  and git never prompts (``GIT_TERMINAL_PROMPT=0``), so it cannot hang waiting for a password;
* every git call has a timeout, and this module never deletes anything: a folder in the way is reported, not removed.

The clone is shallow (``--depth 1``: the latest commit only), lives under ``<INDEX_DIR>/clones/<name>-<hash>/`` and is
updated in place on the next run. The indexer treats it like any local repository, so secrets are filtered the same way.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 600
ALLOWED_PROTOCOLS = "https:ssh"
_SCP_LIKE = re.compile(r"^(?P<user>[\w.-]+)@(?P<host>[\w.-]+):(?P<path>[\w./~+-]+)$")
_HOST = re.compile(r"^[A-Za-z0-9.-]+(?::\d{1,5})?$")
_BRANCH = re.compile(r"^[A-Za-z0-9][\w./-]*$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class GitSourceError(RuntimeError):
    """A repository could not be fetched; the message is safe to show to the user."""


@dataclass(frozen=True)
class RemoteRepo:
    """A repository on another machine, in the form it is shown to the user and the form handed to git."""

    display: str  # e.g. ``https://github.com/owner/repo``
    clone_url: str  # what ``git clone`` receives (the same as ``display`` unless a ``.git`` suffix or scp form is involved)
    name: str  # ``repo``, for the folder name

    @property
    def key(self) -> str:
        digest = hashlib.sha1(self.display.lower().encode("utf-8")).hexdigest()[:8]
        return f"{self.name.lower()}-{digest}"


def looks_like_url(text: str) -> bool:
    """True for anything that is not a local path and should be read as a git remote (accepted or not)."""
    stripped = text.strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", stripped) or _SCP_LIKE.match(stripped) or "::" in stripped)


def _name_from(path: str) -> str:
    last = path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    return _SAFE_NAME.sub("-", last).strip("-.") or "repo"


def parse_git_url(text: str) -> RemoteRepo:
    """Validate ``text`` as a git remote and return it. Raises ``GitSourceError`` saying what is wrong."""
    url = text.strip()
    if not url or any(ch.isspace() for ch in url) or url.startswith("-"):
        raise GitSourceError("That is not a repository URL.")
    scp = _SCP_LIKE.match(url)
    if scp:
        if scp.group("host").startswith(("-", ".")):
            raise GitSourceError("That is not a repository URL.")
        display = f"{scp.group('user')}@{scp.group('host')}:{scp.group('path')}"
        return RemoteRepo(display.removesuffix(".git"), display, _name_from(scp.group("path")))
    match = re.match(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://(?P<rest>.*)$", url)
    if not match:
        raise GitSourceError("Only https:// and ssh repository URLs are supported.")
    scheme = match.group("scheme").lower()
    if scheme not in ("https", "ssh"):
        hint = " Use the https:// URL." if scheme in ("http", "git") else ""
        raise GitSourceError(f"'{scheme}://' URLs are not supported: only https:// and ssh.{hint}")
    authority, _, path = match.group("rest").partition("/")
    userinfo, _, host = authority.rpartition("@")
    if ":" in userinfo or (scheme == "https" and userinfo):
        raise GitSourceError(
            "The URL contains credentials. Remove them and use git's credential helper (https) or an ssh key, so no "
            "password or token is written into the clone or a log."
        )
    if not _HOST.match(host) or host.startswith(("-", ".")):
        raise GitSourceError("That is not a repository URL.")
    if not path.strip("/"):
        raise GitSourceError("The URL has no repository path.")
    display = f"{scheme}://{userinfo + '@' if userinfo else ''}{host}/{path.strip('/')}"
    return RemoteRepo(display.removesuffix(".git"), url, _name_from(path))


def clone_dir_for(remote: RemoteRepo, root: Path) -> Path:
    """Where the clone of ``remote`` lives under ``root`` (an index directory). Does not create it."""
    return Path(root) / "clones" / remote.key


Runner = Callable[..., subprocess.CompletedProcess]


def _friendly(stderr: str) -> str:
    text = stderr.lower()
    if "remote branch" in text and "not found" in text:
        return "That branch does not exist in the repository."
    if any(w in text for w in ("could not read username", "authentication failed", "permission denied", "error: 401", "error: 403")):
        return (
            "Git could not authenticate. Private repositories need a credential helper (https) or an ssh key that "
            "works with plain `git clone`; this tool never asks for or stores a password."
        )
    if any(w in text for w in ("not found", "does not exist", "could not read from remote", "error: 404")):
        return "The repository was not found, or you do not have access to it."
    if any(w in text for w in ("could not resolve host", "failed to connect", "timed out", "connection", "unable to access")):
        return "Could not reach the server. Check the URL and the network connection."
    return "Git failed: " + (stderr.strip().splitlines()[-1] if stderr.strip() else "no details")


def _git(
    args: list[str],
    *,
    cwd: Path | None,
    scratch: Path,
    timeout: float,
    runner: Runner,
    protocols: str,
) -> subprocess.CompletedProcess:
    no_hooks = scratch / "no-hooks"  # a folder that does not exist: git finds no hooks in it
    command = ["git", "-c", f"core.hooksPath={no_hooks}", "-c", "protocol.ext.allow=never", *args]
    env = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ALLOW_PROTOCOL": protocols,
        "GIT_LFS_SKIP_SMUDGE": "1",
    }  # the user's own git config stays in force, so their credential helper still works for private repositories
    try:
        return runner(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitSourceError("git is not installed, or is not on the PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitSourceError(f"git took longer than {int(timeout)} seconds and was stopped.") from exc


def sync_repo(
    remote: RemoteRepo,
    root: Path,
    *,
    branch: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner = subprocess.run,
    protocols: str = ALLOWED_PROTOCOLS,
) -> Path:
    """Clone ``remote`` (shallow) under ``root``, or bring an earlier clone up to date. Returns the folder.

    ``root`` is the index directory, so clones sit beside the indexes. ``protocols`` and ``runner`` exist so the
    tests can use a local repository and a fake git; the CLI and UI never change them.
    """
    if branch is not None and not _BRANCH.match(branch):
        raise GitSourceError("That is not a valid branch name.")
    target = clone_dir_for(remote, root)
    scratch = Path(root) / "clones"
    scratch.mkdir(parents=True, exist_ok=True)

    def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        return _git(args, cwd=cwd, scratch=scratch, timeout=timeout, runner=runner, protocols=protocols)

    if target.exists():
        if not (target / ".git").is_dir():
            raise GitSourceError(f"{target} exists but is not a git clone made by this tool. Move it away and try again.")
        origin = run(["remote", "get-url", "origin"], cwd=target)
        if origin.returncode != 0 or origin.stdout.strip() != remote.clone_url:
            raise GitSourceError(f"{target} is a clone of a different repository. Move it away and try again.")
        ref = branch or "HEAD"
        fetched = run(["fetch", "--depth", "1", "--no-tags", "--no-recurse-submodules", "origin", ref], cwd=target)
        if fetched.returncode != 0:
            raise GitSourceError(_friendly(fetched.stderr))
        reset = run(["reset", "--hard", "FETCH_HEAD"], cwd=target)
        if reset.returncode != 0:
            raise GitSourceError(_friendly(reset.stderr))
        return target

    command = ["clone", "--depth", "1", "--no-tags", "--single-branch", "--no-recurse-submodules", "--template="]
    if branch:
        command += ["--branch", branch]
    result = run([*command, "--", remote.clone_url, str(target)])
    if result.returncode != 0:
        raise GitSourceError(_friendly(result.stderr))
    return target
