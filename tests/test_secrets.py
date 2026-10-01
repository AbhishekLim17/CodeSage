from __future__ import annotations

import pytest

from codebase_ai.ingest.secrets import find_secret, is_secret_filename


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "config/.env.staging",
        "certs/server.pem",
        "deploy.key",
        "home/.ssh/id_rsa",
        "id_ed25519.pub",
        "my-app-firebase-adminsdk-abc12-0123456789.json",
        "prod-service-account.json",
        "credentials.json",
        "infra/terraform.tfstate",
        "prod.tfvars",
        ".npmrc",
    ],
)
def test_secret_filenames_are_flagged(path):
    assert is_secret_filename(path)


@pytest.mark.parametrize(
    "path",
    [".env.example", ".env.sample", ".env.template", "src/app.py", "package.json", "README.md", "keyboard.js"],
)
def test_normal_filenames_are_not_flagged(path):
    assert not is_secret_filename(path)


# Fake credentials are assembled from parts so this file never contains a complete token-shaped string.
@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("private_key", "-----BEGIN " + "RSA PRIVATE KEY-----\nabc"),
        ("private_key", "-----BEGIN " + "PRIVATE KEY-----\nabc"),
        ("aws_access_key", 'aws_key = "' + "AKIA" + "IOSFODNN7EXAMPLE" + '"'),
        ("github_token", "token: " + "ghp_" + "a" * 36),
        ("slack_token", "SLACK=" + "xoxb-" + "1234567890-abcdefghij"),
        ("google_api_key", "key=" + "AIza" + "x" * 35),
        ("anthropic_key", "k=" + "sk-ant-" + "a" * 30),
        ("stripe_live_key", "k=" + "sk_live_" + "a" * 24),
    ],
)
def test_secret_patterns_are_detected(name, text):
    assert find_secret(text) == name


@pytest.mark.parametrize(
    "text",
    [
        "password = get_password()",
        "const key = process.env.API_KEY;",
        "sk = 'short'",
        "AKIA is a prefix",
        "# see https://example.com/tokens",
    ],
)
def test_ordinary_code_is_not_flagged(text):
    assert find_secret(text) is None
