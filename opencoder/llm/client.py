"""Pluggable LLM client for OpenCoder.

The research runs use chat-completion style APIs. OpenAI is the default
"ChatGPT" backend; Gemini is routed through an OpenAI-compatible endpoint
so the rest of the pipeline can stay provider-neutral. Set
OPENCODER_LLM_BASE_URL for gateways such as ZhiZengZeng.

Token logprobs are requested only on backends known to support them. When
logprobs are unavailable, downstream uncertainty still uses
self-consistency and semantic variance.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import urllib3


_BACKEND_ALIASES = {
    "chatgpt": "openai",
    "gpt": "openai",
    "google": "gemini",
    "mock": "offline",
    "local": "offline",
    "zz": "zhizengzeng",
    "zhipu": "zhizengzeng",
    "zhi": "zhizengzeng",
}

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "zhizengzeng": "gpt-4o-mini",
    "offline": "offline-heuristic",
}

_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "zhizengzeng": "https://api.zhizengzeng.com/v1",
}


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so experiments do not require python-dotenv."""
    if os.environ.get("OPENCODER_DISABLE_DOTENV") == "1":
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@contextmanager
def _temporary_dns_override(host: Optional[str], ip: Optional[str]):
    if not host or not ip:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def patched_getaddrinfo(request_host, port, *args, **kwargs):
        if request_host == host:
            return original_getaddrinfo(ip, port, *args, **kwargs)
        return original_getaddrinfo(request_host, port, *args, **kwargs)

    socket.getaddrinfo = patched_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


@dataclass
class LLMResponse:
    text: str
    logprobs: Optional[List[float]] = None          # per-token logprob of the chosen token
    tokens: Optional[List[str]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_logprob(self) -> Optional[float]:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)

    @property
    def token_entropy_proxy(self) -> Optional[float]:
        """Entropy proxy from per-token logprobs of the chosen token.
        Higher = less confident. Bounded approximation: -mean(logprob)."""
        mlp = self.mean_logprob
        return None if mlp is None else -mlp


class LLMClient:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = 0.2,
        max_tokens: int = 1024,
        timeout: int = 120,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ):
        _load_dotenv()
        self.backend = _BACKEND_ALIASES.get(backend.lower(), backend.lower())
        env_model = (
            os.environ.get("OPENCODER_LLM_MODEL")
            or os.environ.get(f"{self.backend.upper()}_MODEL")
        )
        if env_model:
            self.model = env_model
        elif model == _DEFAULT_MODELS["openai"] and self.backend != "openai":
            self.model = _DEFAULT_MODELS.get(self.backend, model)
        else:
            self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.reasoning_effort = (
            os.environ.get("OPENCODER_LLM_REASONING_EFFORT")
            or reasoning_effort
        )
        env_thinking_budget = os.environ.get("OPENCODER_LLM_THINKING_BUDGET")
        self.thinking_budget = (
            int(env_thinking_budget)
            if env_thinking_budget is not None
            else thinking_budget
        )
        self.timeout = int(os.environ.get("OPENCODER_LLM_TIMEOUT", timeout))
        self.request_count = 0
        self.retry_count = 0
        self.failed_attempt_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.response_audit: List[Dict[str, Any]] = []
        self.supports_logprobs = False
        self.host_header = os.environ.get("OPENCODER_LLM_HOST_HEADER")
        self.resolve_ip = os.environ.get("OPENCODER_LLM_RESOLVE_IP")
        self.resolve_host: Optional[str] = None
        self.verify_ssl = os.environ.get("OPENCODER_LLM_VERIFY_SSL", "1") not in {
            "0",
            "false",
            "False",
            "no",
        }

        if self.backend == "offline":
            self.endpoint = "offline://heuristic"
            self.api_key = None
            self.supports_logprobs = True
            return

        base_url = (
            os.environ.get("OPENCODER_LLM_BASE_URL")
            or os.environ.get(f"{self.backend.upper()}_BASE_URL")
            or _DEFAULT_BASE_URLS.get(self.backend)
        )
        connect_base_url = os.environ.get("OPENCODER_LLM_CONNECT_BASE_URL")
        if self.resolve_ip and base_url:
            self.endpoint = base_url.rstrip("/") + "/chat/completions"
            self.logical_endpoint = self.endpoint
            self.resolve_host = urlparse(self.endpoint).hostname
        elif connect_base_url:
            self.endpoint = connect_base_url.rstrip("/") + "/chat/completions"
            self.logical_endpoint = (base_url or connect_base_url).rstrip("/") + "/chat/completions"
        elif base_url:
            self.endpoint = base_url.rstrip("/") + "/chat/completions"
            self.logical_endpoint = self.endpoint
        else:
            raise ValueError(f"Unknown backend: {backend}. Expected offline, openai/chatgpt, gemini, or zhizengzeng.")

        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            warnings.warn(
                "OPENCODER_LLM_VERIFY_SSL=0 disables TLS certificate verification. "
                "Use only for explicitly approved diagnostics/experiments.",
                RuntimeWarning,
                stacklevel=2,
            )

        shared_key = os.environ.get("OPENCODER_LLM_API_KEY")
        if self.backend == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY") or shared_key
            logprob_override = os.environ.get("OPENCODER_LLM_LOGPROBS")
            direct_openai = "api.openai.com" in self.endpoint
            self.supports_logprobs = logprob_override == "1" or (
                direct_openai and logprob_override != "0"
            )
        elif self.backend == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY") or shared_key
        elif self.backend == "zhizengzeng":
            self.api_key = shared_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self.supports_logprobs = os.environ.get("OPENCODER_LLM_LOGPROBS", "0") == "1"

        if not self.api_key:
            key_name = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "zhizengzeng": "OPENCODER_LLM_API_KEY",
            }.get(self.backend, "API_KEY")
            raise RuntimeError(
                f"Missing API key for backend={self.backend}. Set {key_name}."
            )

    def _post_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.host_header:
            headers["Host"] = self.host_header
        retries = max(1, int(os.environ.get("OPENCODER_LLM_RETRIES", "3")))
        retry_sleep = float(os.environ.get("OPENCODER_LLM_RETRY_SLEEP", "2"))
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with _temporary_dns_override(self.resolve_host, self.resolve_ip):
                    response = requests.post(
                        self.endpoint,
                        json=body,
                        headers=headers,
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                    )
                if response.status_code not in {429} and response.status_code < 500:
                    break
                if attempt == retries:
                    break
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.SSLError,
            ) as exc:
                last_exc = exc
                if attempt == retries:
                    raise
            time.sleep(retry_sleep * attempt)
        if response is None:
            if last_exc:
                raise last_exc
            raise RuntimeError(f"{self.backend} request failed before receiving a response.")
        if response.status_code == 429:
            raise RuntimeError(f"{self.backend} rate limited (429). Back off and retry.")
        if response.status_code == 402:
            raise RuntimeError(f"{self.backend} payment required (402). Check billing/credits.")
        if response.status_code in {401, 403}:
            raise RuntimeError(f"{self.backend} authentication failed ({response.status_code}).")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:800]
            raise RuntimeError(f"{self.backend} request failed ({response.status_code}): {detail}") from exc
        data = response.json()
        self.retry_count += max(0, attempt - 1)
        self.failed_attempt_count += max(0, attempt - 1)
        self.request_count += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(
            usage.get("total_tokens")
            or (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
        )
        self.response_audit.append({
            "request_index": self.request_count,
            "requested_model": self.model,
            "response_id": data.get("id"),
            "served_model": data.get("model"),
            "created": data.get("created"),
            "usage": usage,
            "finish_reasons": [
                choice.get("finish_reason")
                for choice in data.get("choices", [])
                if isinstance(choice, dict)
            ],
        })
        return data

    def usage_snapshot(self) -> Dict[str, int]:
        return {
            "requests": self.request_count,
            "retries": self.retry_count,
            "failed_attempts": self.failed_attempt_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def response_audit_snapshot(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.response_audit]

    def _post_chat_with_fallbacks(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._post_chat(body)
        except RuntimeError as first_error:
            lowered = str(first_error).lower()
            body_without_logprobs = dict(body)
            had_logprobs = any(k in body_without_logprobs for k in ("logprobs", "top_logprobs"))
            body_without_logprobs.pop("logprobs", None)
            body_without_logprobs.pop("top_logprobs", None)
            if had_logprobs and any(term in lowered for term in ("logprob", "top_logprobs", "unsupported")):
                self.supports_logprobs = False
                try:
                    return self._post_chat(body_without_logprobs)
                except RuntimeError as second_error:
                    if body_without_logprobs.get("n", 1) > 1:
                        return self._post_chat_many(body_without_logprobs)
                    raise second_error
            if body.get("n", 1) > 1 and any(term in lowered for term in (" n ", "'n'", '"n"', "unsupported")):
                return self._post_chat_many(body_without_logprobs if had_logprobs else body)
            raise

    def _post_chat_many(self, body: Dict[str, Any]) -> Dict[str, Any]:
        n = int(body.get("n", 1))
        choices: List[Dict[str, Any]] = []
        single = dict(body)
        single["n"] = 1
        single.pop("logprobs", None)
        single.pop("top_logprobs", None)
        self.supports_logprobs = False
        base_seed = single.get("seed")
        for sample_index in range(n):
            if base_seed is not None:
                single["seed"] = int(base_seed) + sample_index
            data = self._post_chat(single)
            response_metadata = {
                "response_id": data.get("id"),
                "served_model": data.get("model"),
                "created": data.get("created"),
                "usage": data.get("usage") or {},
            }
            for choice in data.get("choices", []):
                annotated = dict(choice)
                annotated["_response_metadata"] = response_metadata
                choices.append(annotated)
        return {"choices": choices}

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        return_logprobs: bool = True,
        seed: Optional[int] = None,
    ) -> List[LLMResponse]:
        if self.backend == "offline":
            return [
                self._complete_offline(messages, sample_index=i, return_logprobs=return_logprobs)
                for i in range(n)
            ]

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "n": n,
        }
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            body["temperature"] = effective_temperature
        request_seed = self.seed if seed is None else seed
        if request_seed is not None:
            body["seed"] = int(request_seed)
        thinking_budget = getattr(self, "thinking_budget", None)
        reasoning_effort = getattr(self, "reasoning_effort", None)
        if thinking_budget is not None:
            body["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "thinking_budget": int(thinking_budget),
                        "include_thoughts": False,
                    }
                }
            }
        elif reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if return_logprobs and self.supports_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 1

        data = self._post_chat_with_fallbacks(body)
        choices = list(data.get("choices", []))
        if n > 1 and len(choices) < n:
            missing_body = dict(body)
            missing_body["n"] = n - len(choices)
            missing_data = self._post_chat_many(missing_body)
            choices.extend(missing_data.get("choices", []))

        out: List[LLMResponse] = []
        response_metadata = {
            "response_id": data.get("id"),
            "served_model": data.get("model"),
            "created": data.get("created"),
            "usage": data.get("usage") or {},
        }
        for choice in choices[:n]:
            text = choice["message"]["content"] or ""
            lp = choice.get("logprobs") or {}
            content_lp = lp.get("content") if isinstance(lp, dict) else None
            logprobs, tokens = None, None
            if content_lp:
                logprobs = [t.get("logprob") for t in content_lp if t.get("logprob") is not None]
                tokens = [t.get("token") for t in content_lp]
            raw = dict(choice)
            raw.setdefault("_response_metadata", response_metadata)
            out.append(LLMResponse(text=text, logprobs=logprobs, tokens=tokens, raw=raw))
        if not out:
            raise RuntimeError(f"{self.backend} returned no choices: {data}")
        return out

    def complete_one(self, prompt: str, system: Optional[str] = None, **kw) -> LLMResponse:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, **kw)[0]

    def _complete_offline(
        self,
        messages: List[Dict[str, str]],
        *,
        sample_index: int = 0,
        return_logprobs: bool = True,
    ) -> LLMResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        prompt = messages[-1]["content"] if messages else ""
        sys_l = system.lower()

        if "strict json" in sys_l and '"steps"' in system:
            text = json.dumps({
                "steps": [
                    {"index": 1, "description": "Understand the target function signature and expected behavior."},
                    {"index": 2, "description": "Inspect retrieved repository evidence for helper APIs and conventions."},
                    {"index": 3, "description": "Implement the missing Python code and keep it consistent with the repository."},
                ]
            })
        elif "evidence types" in sys_l and '"weights"' in system:
            step = prompt.splitlines()[0].replace("Step:", "").strip() or prompt.strip()
            text = json.dumps({
                "weights": {"api": 0.34, "context": 0.33, "similar_code": 0.33},
                "queries": {"api": step, "context": step, "similar_code": step},
            })
        elif "single concise paragraph" in sys_l:
            sig = self._first_match(r"Signature:\s*(.+)", prompt) or "repository item"
            text = (
                f"{sig} is a repository symbol extracted for retrieval. "
                "Use it as API evidence when the target task appears to call or mirror this behavior."
            )
        elif "single implementation step is specified" in sys_l:
            text = "The step is partly specified by the signature and retrieved context, but implementation details may remain open."
        elif "repairing python code" in sys_l:
            text = "```python\n" + self._extract_failed_code(prompt) + "\n```"
        elif "senior python engineer" in sys_l:
            text = "```python\n" + self._offline_code(prompt, sample_index=sample_index) + "\n```"
        else:
            text = "Offline OpenCoder heuristic response."

        logprobs = [-0.15 - (0.05 * (sample_index % 3))] * max(1, len(text.split())) if return_logprobs else None
        return LLMResponse(
            text=text,
            logprobs=logprobs,
            tokens=text.split() if return_logprobs else None,
            raw={"backend": "offline", "sample_index": sample_index},
        )

    @staticmethod
    def _first_match(pattern: str, text: str) -> Optional[str]:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_failed_code(prompt: str) -> str:
        m = re.search(r"# Failed Code\s*```(?:python)?\s*(.*?)```", prompt, re.DOTALL)
        return (m.group(1) if m else "pass").strip()

    @staticmethod
    def _extract_function_stub(text: str) -> tuple[Optional[str], Optional[str]]:
        m = re.search(
            r"(def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:\n"
            r"(?:[ \t]+\"\"\".*?\"\"\"\n?)?)",
            text,
            re.DOTALL,
        )
        if not m:
            return None, None
        return m.group(1).rstrip(), m.group(2)

    def _offline_code(self, prompt: str, *, sample_index: int = 0) -> str:
        if "Complete the missing Python code" in prompt:
            return "pass"

        stub, name = self._extract_function_stub(prompt)
        if not stub or not name:
            return "# Offline heuristic could not infer the target.\npass"

        indent = "    "
        if name == "reverse":
            return (
                f"{stub}\n"
                f"{indent}if not is_string(input_string):\n"
                f"{indent*2}raise InvalidInputError(input_string)\n\n"
                f"{indent}return input_string[::-1]"
            )
        if name == "camel_case_to_snake":
            return (
                f"{stub}\n"
                f"{indent}if not is_string(input_string):\n"
                f"{indent*2}raise InvalidInputError(input_string)\n\n"
                f"{indent}if not is_camel_case(input_string):\n"
                f"{indent*2}return input_string\n\n"
                f"{indent}return CAMEL_CASE_REPLACE_RE.sub(lambda m: m.group(1) + separator, input_string).lower()"
            )
        if name == "snake_case_to_camel":
            return (
                f"{stub}\n"
                f"{indent}if not is_string(input_string):\n"
                f"{indent*2}raise InvalidInputError(input_string)\n\n"
                f"{indent}if not is_snake_case(input_string):\n"
                f"{indent*2}return input_string\n\n"
                f"{indent}return ''.join(word.capitalize() for word in input_string.split(separator))"
            )

        body = "return None" if sample_index % 2 else "pass"
        return f"{stub}\n{indent}{body}"
