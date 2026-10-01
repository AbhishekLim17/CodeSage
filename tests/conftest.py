from __future__ import annotations

import os
import shutil
from pathlib import Path

# Colour codes in the middle of a sentence break tests that read terminal output, and some shells and CI systems force
# them on. The CLI creates its rich consoles when it is imported, which happens after this file loads but before any
# fixture runs, so the variables have to be cleared here.
for _name in ("FORCE_COLOR", "FORCE_TERMINAL", "TTY_COMPATIBLE", "CLICOLOR_FORCE"):
    os.environ.pop(_name, None)

import pytest

from codebase_ai.config import Settings, get_settings
from helpers import FIXTURE_REPO, FakeEmbedder


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """No developer env vars or .env file leak into tests, and nothing is written to ~/.codebase_ai."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index-root"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_repo(tmp_path) -> Path:
    """A writable copy of tests/fixtures/sample_repo."""
    target = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, target)
    return target
