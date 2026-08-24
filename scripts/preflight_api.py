"""Preflight an OpenCoder LLM backend before launching experiments."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from urllib.parse import urlparse
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencoder.llm import LLMClient  # noqa: E402


def _redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-...REDACTED", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer ...REDACTED", text)
    return text


def _default_model(backend: str) -> str:
    if backend == "gemini":
        return "gemini-2.5-flash"
    if backend == "offline":
        return "offline-heuristic"
    return "gpt-4o-mini"


def _diagnose_endpoint(endpoint: str, timeout: int, resolve_ip: str | None = None) -> None:
    if endpoint.startswith("offline://"):
        print(f"endpoint={endpoint}", flush=True)
        return

    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        print(f"endpoint={endpoint} (unable to parse host)", flush=True)
        return

    if resolve_ip:
        print(f"dns_override={host}->{resolve_ip}", flush=True)
    else:
        try:
            resolved = socket.gethostbyname_ex(host)
            print(f"dns={resolved[2]}", flush=True)
        except Exception as exc:
            print(f"dns failed: {_redact(str(exc))}", flush=True)
            return

    try:
        connect_host = resolve_ip or host
        with socket.create_connection((connect_host, port), timeout=timeout):
            print(f"tcp={connect_host}:{port} reachable", flush=True)
    except Exception as exc:
        print(f"tcp={host}:{port} failed: {_redact(repr(exc))}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("OPENCODER_LLM_BACKEND", "openai"))
    ap.add_argument("--model", default=os.environ.get("OPENCODER_LLM_MODEL"))
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--sleep", type=int, default=60)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    backend = args.backend.lower()
    model = args.model or _default_model(backend)
    last_error = ""
    for attempt in range(1, args.retries + 1):
        print(
            f"API preflight attempt {attempt}/{args.retries}: backend={backend} model={model}",
            flush=True,
        )
        try:
            temperature = None if model == "claude-sonnet-5" else 0.0
            client = LLMClient(
                backend=backend,
                model=model,
                temperature=temperature,
                max_tokens=32,
                timeout=args.timeout,
            )
            print(f"endpoint={client.endpoint}", flush=True)
            if getattr(client, "logical_endpoint", client.endpoint) != client.endpoint:
                print(f"logical_endpoint={client.logical_endpoint}", flush=True)
            if getattr(client, "host_header", None):
                print(f"host_header={client.host_header}", flush=True)
            print(f"verify_ssl={client.verify_ssl}", flush=True)
            _diagnose_endpoint(
                client.endpoint,
                min(args.timeout, 10),
                getattr(client, "resolve_ip", None),
            )
            response = client.complete_one(
                "Reply with exactly: OpenCoder API ready.",
                system="You are a concise connectivity checker.",
                return_logprobs=False,
            )
            print(f"API preflight OK: {response.text.strip()}", flush=True)
            metadata = (response.raw or {}).get("_response_metadata") or {}
            payload = {
                "status": "COMPLETED",
                "backend": backend,
                "requested_model": model,
                "served_model": metadata.get("served_model"),
                "response_id": metadata.get("response_id"),
                "temperature": temperature,
                "usage": client.usage_snapshot(),
                "response_matches_instruction": response.text.strip() == "OpenCoder API ready.",
            }
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return
        except Exception as exc:
            last_error = _redact(str(exc))
            print(f"API preflight failed: {last_error}", flush=True)
            if attempt < args.retries:
                print(f"Sleeping {args.sleep}s before retry...", flush=True)
                time.sleep(args.sleep)

    raise SystemExit(f"API preflight failed after {args.retries} attempt(s): {last_error}")


if __name__ == "__main__":
    main()
