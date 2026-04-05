"""Token-budgeted memory context assembly and deduplication."""

from __future__ import annotations

import logging

from evoagent.core.protocols import MemoryBackend

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def compress_context(
    memory: MemoryBackend,
    task: str,
    token_budget: int = 4000,
) -> str:
    """Assemble relevant memories into a token-budgeted context string."""
    char_budget = token_budget * CHARS_PER_TOKEN
    semantic_items = list(memory.list_all("semantic").values())
    episodic_items = list(memory.list_all("episodic").values())

    parts: list[str] = ["\n\n## Relevant Past Experience"]
    current_len = len(parts[0])

    meta = [m for m in semantic_items if m.get("type") == "meta_instruction"]
    regular = [m for m in semantic_items if m.get("type") != "meta_instruction"]

    if meta:
        header = "### Strategy Guidelines (from cross-run analysis)"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in meta:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if regular:
        header = "### Learned Facts & Patterns"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in regular:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if episodic_items:
        header = "### Recent Run Summaries"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in episodic_items[-10:]:
                task_str = mem.get("task", "?")[:80]
                score = mem.get("score", "?")
                summary = mem.get("summary", "")[:100]
                line = f"- [{score}] {task_str}: {summary}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    return "\n".join(parts) if len(parts) > 1 else ""


def _jaccard_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def deduplicate_semantic(memory: MemoryBackend, threshold: float = 0.7) -> int:
    """Remove near-duplicate semantic memories. Returns count removed."""
    items = memory.list_all("semantic")
    keys = list(items.keys())
    to_remove: set[str] = set()

    for i in range(len(keys)):
        if keys[i] in to_remove:
            continue
        content_i = items[keys[i]].get("content", str(items[keys[i]]))
        for j in range(i + 1, len(keys)):
            if keys[j] in to_remove:
                continue
            content_j = items[keys[j]].get("content", str(items[keys[j]]))
            if _jaccard_similarity(content_i, content_j) >= threshold:
                to_remove.add(keys[j])

    for key in to_remove:
        memory.delete("semantic", key)
    return len(to_remove)
