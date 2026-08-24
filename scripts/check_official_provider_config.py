"""Validate direct OpenAI/Gemini configuration without printing secrets."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        raise SystemExit(".env is missing")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def host(name: str) -> str | None:
    value = os.environ.get(name)
    return urlparse(value).hostname if value else None


def main() -> int:
    load_env()
    errors: list[str] = []
    global_host = host("OPENCODER_LLM_BASE_URL")
    openai_host = host("OPENAI_BASE_URL")
    gemini_host = host("GEMINI_BASE_URL")

    if global_host:
        errors.append(
            "OPENCODER_LLM_BASE_URL is set and overrides both official endpoints"
        )
    if openai_host != "api.openai.com":
        errors.append("OPENAI_BASE_URL must resolve to api.openai.com")
    if gemini_host != "generativelanguage.googleapis.com":
        errors.append(
            "GEMINI_BASE_URL must resolve to generativelanguage.googleapis.com"
        )

    shared = os.environ.get("ZHIZENGZENG_API_KEY")
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name)
        if not value or value.startswith("replace_with_"):
            errors.append(f"{name} is missing or still a placeholder")
        elif shared and value == shared:
            errors.append(f"{name} still equals the shared gateway credential")

    if errors:
        print("Official-provider configuration is not ready:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Official-provider configuration is ready.")
    print("- OpenAI endpoint: api.openai.com")
    print("- Gemini endpoint: generativelanguage.googleapis.com")
    print("- Credentials: present and distinct from the shared gateway key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
