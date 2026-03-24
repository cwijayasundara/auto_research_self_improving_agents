"""Memory compression utilities.

Provides deduplication, consolidation, and token-budgeted context building
to prevent unbounded memory growth across evolution cycles.
"""

import logging
import uuid

from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two strings based on word tokens."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def deduplicate_semantic(memory_store: MemoryStore, threshold: float = 0.7) -> int:
    """Remove near-duplicate semantic memories using Jaccard similarity."""
    memories = memory_store.list_all("semantic")
    if len(memories) < 2:
        return 0

    keys_to_remove: set[str] = set()
    for i in range(len(memories)):
        if memories[i]["_key"] in keys_to_remove:
            continue
        for j in range(i + 1, len(memories)):
            if memories[j]["_key"] in keys_to_remove:
                continue
            content_a = memories[i].get("content", str(memories[i]))
            content_b = memories[j].get("content", str(memories[j]))
            if _jaccard_similarity(content_a, content_b) > threshold:
                keys_to_remove.add(memories[j]["_key"])

    for key in keys_to_remove:
        memory_store.delete("semantic", key)

    if keys_to_remove:
        logger.info("Deduplicated %d semantic memories", len(keys_to_remove))
    return len(keys_to_remove)


def consolidate_episodic(
    memory_store: MemoryStore,
    current_cycle: int,
    max_recent: int = 10,
) -> int:
    """Consolidate older episodic memories, keeping the most recent intact."""
    memories = memory_store.list_all("episodic")
    if len(memories) <= max_recent:
        return 0

    memories.sort(key=lambda m: m.get("_stored_at", ""), reverse=True)
    old = memories[max_recent:]

    if not old:
        return 0

    groups: dict[str, list[dict]] = {}
    for mem in old:
        stored_at = mem.get("_stored_at", "")
        date_key = stored_at[:10] if len(stored_at) >= 10 else "unknown"
        groups.setdefault(date_key, []).append(mem)

    consolidated_count = 0
    for date_key, group in groups.items():
        summaries = []
        for mem in group:
            summary = mem.get("summary", mem.get("task", str(mem.get("_key", ""))))
            summaries.append(summary)

        consolidated_summary = f"Consolidated {len(group)} episodes from {date_key}: " + "; ".join(
            summaries
        )

        for mem in group:
            memory_store.delete("episodic", mem["_key"])
            consolidated_count += 1

        memory_store.store(
            "episodic",
            f"consolidated-{date_key}-{uuid.uuid4().hex[:8]}",
            {
                "summary": consolidated_summary,
                "type": "consolidated",
                "source_count": len(group),
                "date_bucket": date_key,
                "consolidated_at_cycle": current_cycle,
            },
        )

    if consolidated_count:
        logger.info(
            "Consolidated %d old episodic memories into %d summaries",
            consolidated_count,
            len(groups),
        )
    return consolidated_count


def compress_memory_context(
    memory_store: MemoryStore,
    task: str,
    token_budget: int = 2000,
) -> str:
    """Build memory context within a token budget."""
    char_budget = token_budget * 4

    semantic = memory_store.search("semantic", task)
    episodic = memory_store.search("episodic", task)

    if not semantic and not episodic:
        return ""

    parts: list[str] = ["\n\n## Relevant Past Experience"]
    current_len = len(parts[0])

    if semantic:
        header = "### Learned Facts & Patterns"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in semantic:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if episodic and current_len < char_budget:
        header = "### Previous Runs"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in episodic:
                line = f"- {mem.get('summary', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if len(parts) <= 1:
        return ""

    return "\n".join(parts)
