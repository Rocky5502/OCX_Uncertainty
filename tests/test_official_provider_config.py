from __future__ import annotations

from scripts.check_official_provider_config import host


def test_host_returns_hostname_without_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert host("OPENAI_BASE_URL") == "api.openai.com"


def test_host_returns_none_for_missing_value(monkeypatch):
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    assert host("GEMINI_BASE_URL") is None
