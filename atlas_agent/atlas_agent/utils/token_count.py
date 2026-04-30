from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _whitespace_count(text: str) -> int:
    return len(text.split()) if text else 0


def _tiktoken_count(text: str, model: Optional[str] = None) -> Optional[int]:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return None


def _hf_count(text: str, model: Optional[str] = None) -> Optional[int]:
    try:
        from transformers import AutoTokenizer

        # fall back to a widely-available tokenizer if model not provided
        tok_name = model if model is not None else "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        return len(tokenizer.encode(text or ""))
    except Exception:
        return None


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Return tokenizer-exact token count when possible, else fall back to whitespace heuristic.

    Tries tiktoken first, then HuggingFace tokenizers, then whitespace split.
    """
    if not text:
        return 0

    try:
        val = _tiktoken_count(text, model)
        if val is not None:
            return val
    except Exception as exc:
        logger.debug("tiktoken failed: %s", exc)

    try:
        val = _hf_count(text, model)
        if val is not None:
            return val
    except Exception as exc:
        logger.debug("HF tokenizer failed: %s", exc)

    return _whitespace_count(text)
