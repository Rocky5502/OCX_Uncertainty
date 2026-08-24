"""Text/code encoder.

Default: UniXcoder via HuggingFace transformers. If torch/transformers
are not installed, falls back to a deterministic hashing-based encoder
so the pipeline still runs end-to-end for smoke tests.
"""
from __future__ import annotations

import hashlib
import importlib.util
from typing import List, Sequence

import numpy as np


class Encoder:
    def __init__(self, model_name: str = "microsoft/unixcoder-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._backend = None
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("transformers"):
            self._backend = "hash"
            self.dim = 256
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(device).eval()
            self._backend = "hf"
            self.dim = self._model.config.hidden_size
        except Exception:
            self._backend = "hash"
            self.dim = 256

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        if self._backend == "hf":
            return self._encode_hf(list(texts), batch_size)
        return self._encode_hash(list(texts))

    def _encode_hf(self, texts: List[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._model(**enc)
            # Mean-pool over tokens, mask padding.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy()
            vecs.append(pooled)
        v = np.vstack(vecs)
        # L2 normalize for cosine similarity.
        n = np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)
        return v / n

    def _encode_hash(self, texts: List[str]) -> np.ndarray:
        # Deterministic fallback so pipeline runs without torch.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)
        return out / n
