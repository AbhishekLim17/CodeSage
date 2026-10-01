"""Detect secret files and high-entropy tokens so they never reach an index, prompt or log.

This is a best-effort safety net, not a guarantee: it catches well-known credential file names
and token formats. It deliberately errs on the side of skipping a file.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

# `.env` variants that are meant to be committed as templates.
_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}

_SECRET_FILE_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*firebase-adminsdk*.json",
    "*service-account*.json",
    "*serviceaccount*.json",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "*.tfstate",
    "*.tfvars",
)

# name -> compiled regex. Only formats with a distinctive prefix/shape, to keep false positives low.
_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY(?: BLOCK)?-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}\b"),
    "stripe_live_key": re.compile(r"\b[sr]k_live_[0-9A-Za-z]{20,}\b"),
}


def is_secret_filename(path: str) -> bool:
    """True if the file name matches a known credential/secret file pattern."""
    name = PurePosixPath(path).name
    if name in _ENV_TEMPLATES:
        return False
    return any(fnmatchcase(name, glob) for glob in _SECRET_FILE_GLOBS)


def find_secret(text: str) -> str | None:
    """Return the name of the first secret pattern found in ``text``, else ``None``."""
    for name, pattern in _SECRET_PATTERNS.items():
        if pattern.search(text):
            return name
    return None
