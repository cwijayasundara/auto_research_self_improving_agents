"""Shared LLM JSON response parsing utilities.

Handles markdown code blocks, trailing text, and graceful fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown code blocks.

    Tries in order:
    1. Strip markdown ```json ... ``` wrapper
    2. Parse as raw JSON
    3. Return empty dict on failure
    """
    cleaned = text.strip()
    if not cleaned:
        return {}

    # Strip markdown code block wrapper
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop the opening fence line (e.g. ```json or ```)
        lines = lines[1:]
        # Keep only lines up to (but not including) the closing fence
        inner: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        cleaned = "\n".join(inner).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse LLM JSON: %s", cleaned[:200])
        return {}
