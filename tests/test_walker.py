from __future__ import annotations

import hashlib
import os

import pytest

from codebase_ai.ingest.walker import WalkLimits, WalkReport, load_source_file, walk_repo


def write(root, relpath, content="x = 1\n", binary=False):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")
    return path


def walk(root, **kwargs):
    report = WalkReport()
    files = list(walk_repo(root, report=report, **kwargs))
    return files, report


def test_yields_indexable_files_sorted_with_kind_and_language(tmp_path):
    write(tmp_path, "b.py")
    write(tmp_path, "a/z.js", "let a = 1;\n")
    write(tmp_path, "a/README.md", "# hi\n")
    write(tmp_path, "cfg.yaml", "a: 1\n")
    files, report = walk(tmp_path)
    # deterministic: a directory's own files (sorted) first, then its sub-directories
    assert [f.path for f in files] == ["b.py", "cfg.yaml", "a/README.md", "a/z.js"]
    assert {f.path: (f.kind, f.language) for f in files}["a/z.js"] == ("code", "javascript")
    assert report.files_yielded == 4


def test_file_hash_is_sha256_of_raw_bytes(tmp_path):
    write(tmp_path, "a.py", "print('hi')\n")
    (file,), _ = walk(tmp_path)
    assert file.file_hash == hashlib.sha256(b"print('hi')\n").hexdigest()


def test_crlf_is_normalised_and_bom_stripped(tmp_path):
    write(tmp_path, "a.py", b"\xef\xbb\xbfx = 1\r\ny = 2\r\n", binary=True)
    (file,), _ = walk(tmp_path)
    assert file.text == "x = 1\ny = 2\n"


def test_root_and_nested_gitignore_are_honoured(tmp_path):
    write(tmp_path, ".gitignore", "ignored_dir/\n*.log.txt\n")
    write(tmp_path, "ignored_dir/a.py")
    write(tmp_path, "debug.log.txt", "noise")
    write(tmp_path, "keep.py")
    write(tmp_path, "sub/.gitignore", "local.txt\n")
    write(tmp_path, "sub/local.txt", "mine")
    write(tmp_path, "sub/keep.txt", "shared")
    files, report = walk(tmp_path)
    paths = [f.path for f in files]
    assert "keep.py" in paths and "sub/keep.txt" in paths
    assert "ignored_dir/a.py" not in paths
    assert "debug.log.txt" not in paths
    assert "sub/local.txt" not in paths
    assert report.skipped["gitignored"] == 3  # the ignored dir (once), the .log.txt file, sub/local.txt


def test_nested_gitignore_does_not_leak_to_siblings(tmp_path):
    write(tmp_path, "sub/.gitignore", "local.txt\n")
    write(tmp_path, "sub/local.txt", "mine")
    write(tmp_path, "other/local.txt", "not ignored here")
    files, _ = walk(tmp_path)
    paths = [f.path for f in files]
    assert "other/local.txt" in paths
    assert "sub/local.txt" not in paths


def test_denied_directories_are_pruned(tmp_path):
    write(tmp_path, "node_modules/pkg/index.js")
    write(tmp_path, ".git/hooks/pre-commit.py")
    write(tmp_path, "dist/bundle.js")
    write(tmp_path, "src/app.py")
    files, report = walk(tmp_path)
    assert [f.path for f in files] == ["src/app.py"]
    assert report.skipped["denied_dir"] == 3


def test_secret_files_are_never_yielded(tmp_path):
    write(tmp_path, ".env.local", "API_KEY=abc\n")
    write(tmp_path, "certs/server.pem", "x")
    write(tmp_path, "svc-firebase-adminsdk-1.json", "{}")
    write(tmp_path, ".env.example", "API_KEY=\n")
    files, report = walk(tmp_path)
    assert [f.path for f in files] == [".env.example"]
    assert report.skipped["secret_file"] == 3


def test_file_containing_a_secret_pattern_is_skipped(tmp_path):
    fake_key = "AKIA" + "IOSFODNN7EXAMPLE"
    write(tmp_path, "leaky.py", f'KEY = "{fake_key}"\n')
    write(tmp_path, "clean.py")
    files, report = walk(tmp_path)
    assert [f.path for f in files] == ["clean.py"]
    assert report.skipped["secret_pattern"] == 1
    assert report.examples["secret_pattern"] == ["leaky.py"]


def test_binary_files_are_skipped(tmp_path):
    write(tmp_path, "fake.py", b"abc\x00def", binary=True)
    files, report = walk(tmp_path)
    assert files == [] and report.skipped["binary"] == 1


def test_size_limits_differ_for_code_and_config(tmp_path):
    write(tmp_path, "big.py", "x = 1\n" * 100)
    write(tmp_path, "big.json", '{"a": 1}\n' * 100)
    files, report = walk(tmp_path, limits=WalkLimits(max_file_bytes=1000, max_config_bytes=100))
    assert [f.path for f in files] == ["big.py"]
    assert report.skipped["too_large"] == 1


def test_minified_files_are_skipped(tmp_path):
    write(tmp_path, "min.js", "var a=" + "1+" * 2000 + "1;\n")
    files, report = walk(tmp_path)
    assert files == [] and report.skipped["minified_or_generated"] == 1


@pytest.mark.parametrize("name", ["package-lock.json", "yarn.lock", "poetry.lock", "app.min.js", "go.sum"])
def test_lockfiles_and_generated_files_are_skipped(tmp_path, name):
    write(tmp_path, name, "{}\n")
    files, report = walk(tmp_path)
    assert files == [] and report.skipped["lockfile_or_generated"] == 1


def test_empty_and_unsupported_files_are_skipped(tmp_path):
    write(tmp_path, "empty.py", "  \n\n")
    write(tmp_path, "image.png", b"\x89PNG", binary=True)
    files, report = walk(tmp_path)
    assert files == []
    assert report.skipped["empty"] == 1 and report.skipped["unsupported_type"] == 1


def test_symlinks_are_not_followed(tmp_path):
    outside = tmp_path / "outside"
    write(outside, "secret.py", "x = 1\n")
    repo = tmp_path / "repo"
    write(repo, "real.py")
    try:
        os.symlink(outside, repo / "linked_dir", target_is_directory=True)
        os.symlink(outside / "secret.py", repo / "linked.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    files, report = walk(repo)
    assert [f.path for f in files] == ["real.py"]
    assert report.skipped["symlink"] == 2


def test_load_source_file_reports_reason(tmp_path):
    path = write(tmp_path, ".env", "A=1\n")
    assert load_source_file(path, ".env") == (None, "secret_file")
    ok = write(tmp_path, "a.py")
    source, reason = load_source_file(ok, "a.py")
    assert reason is None and source is not None and source.path == "a.py"
