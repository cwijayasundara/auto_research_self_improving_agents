"""Sleep-time compute: offline cross-run trace analysis."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.protocols import MemoryBackend

logger = logging.getLogger(__name__)

SLEEP_REVIEW_PROMPT = (
    "You are analyzing execution traces from multiple agent runs.\n\n"
    "## Episodic Memories\n{episodic_summaries}\n\n"
    "## Execution Traces\n{trace_summaries}\n\n"
    "Identify:\n"
    "1. consistent_successes\n2. recurring_failures\n"
    "3. task_type_insights\n4. meta_instructions (3-5 rules)\n\n"
    "Respond as JSON with those 4 keys (each a list of strings)."
)

CONTRADICTION_PROMPT = (
    "You are reviewing a cluster of related facts from an agent's memory.\n\n"
    "## Facts\n{facts}\n\n"
    "Identify any contradictions between these facts.\n"
    "For each contradiction found, specify:\n"
    "- fact_a: the text of the first contradicting fact\n"
    "- fact_b: the text of the second contradicting fact\n"
    "- resolution: a brief explanation of which is more accurate\n"
    "- keep: the key of the fact to keep (use the key= value shown in the facts list)\n\n"
    'Respond as JSON: {{"contradictions": [{{"fact_a": ..., "fact_b": ..., '
    '"resolution": ..., "keep": "<key>"}}]}}\n'
    'If no contradictions exist, respond with {{"contradictions": []}}.'
)


def run_sleep_review(
    llm: BaseChatModel,
    memory: MemoryBackend,
    traces_dir: Path,
    prompt_template: str | None = None,
    max_traces: int = 20,
    detect_contradictions_flag: bool = True,
) -> dict[str, Any]:
    """Run sleep-time review and store meta-instructions."""
    contradictions_removed = 0
    if detect_contradictions_flag:
        try:
            contradictions_removed = detect_contradictions(llm, memory)
        except Exception as exc:
            logger.warning("Contradiction detection failed: %s", exc)

    traces = _load_traces(traces_dir, limit=max_traces)
    if not traces:
        return {
            "meta_instructions": [],
            "status": "no_traces",
            "contradictions_removed": contradictions_removed,
        }

    trace_summaries = "\n".join(f"- {_summarize_trace(t)}" for t in traces)

    episodic_items = memory.list_all("episodic")
    ep_lines = []
    for _key, data in list(episodic_items.items())[:15]:
        ep_lines.append(
            f"- Task: {data.get('task', '?')[:80]} | Score: {data.get('score', '?')}"
        )
    episodic_summaries = "\n".join(ep_lines) or "No episodic memories yet."

    template = prompt_template or SLEEP_REVIEW_PROMPT
    prompt = template.format(
        episodic_summaries=episodic_summaries,
        trace_summaries=trace_summaries,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    result = parse_llm_json(response.content)

    stored = 0
    for instruction in result.get("meta_instructions", []):
        if isinstance(instruction, str) and instruction.strip():
            memory.store("semantic", f"meta-{uuid.uuid4().hex[:8]}", {
                "type": "meta_instruction",
                "content": instruction.strip(),
                "source": "sleep_review",
            })
            stored += 1

    logger.info("Sleep review: %d traces, %d meta-instructions stored", len(traces), stored)

    return {
        "meta_instructions": result.get("meta_instructions", []),
        "consistent_successes": result.get("consistent_successes", []),
        "recurring_failures": result.get("recurring_failures", []),
        "status": "complete",
        "contradictions_removed": contradictions_removed,
    }


def detect_contradictions(
    llm: BaseChatModel,
    memory: MemoryBackend,
    max_scan: int = 50,
) -> int:
    """Detect and remove contradictory facts from semantic memory.

    Loads semantic memories (skipping meta_instructions), clusters them by
    Jaccard similarity, and for each cluster with 2+ items asks the LLM to
    find contradictions. Removes the losing fact and returns count removed.
    """
    all_items = memory.list_all("semantic")
    # Filter out meta_instructions
    items = {
        key: data
        for key, data in all_items.items()
        if data.get("type") != "meta_instruction"
    }

    if not items:
        return 0

    # Limit scan size
    item_list = list(items.items())[:max_scan]

    clusters = _cluster_memories(item_list, threshold=0.25)

    removed = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        facts_lines = []
        for key, data in cluster:
            content = data.get("content", "")
            facts_lines.append(f'- key="{key}": {content}')
        facts_text = "\n".join(facts_lines)

        prompt = CONTRADICTION_PROMPT.format(facts=facts_text)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = parse_llm_json(response.content)
        except Exception as exc:
            logger.warning("Contradiction LLM call failed: %s", exc)
            return removed

        for contradiction in result.get("contradictions", []):
            keep_key = contradiction.get("keep", "")
            fact_a_text = contradiction.get("fact_a", "")
            fact_b_text = contradiction.get("fact_b", "")

            # Find which key corresponds to which fact text
            key_to_remove: str | None = None
            for key, data in cluster:
                content = data.get("content", "")
                if key == keep_key:
                    continue
                # If this key's content matches the non-kept fact, remove it
                if content == fact_a_text and keep_key != key:
                    key_to_remove = key
                    break
                if content == fact_b_text and keep_key != key:
                    key_to_remove = key
                    break

            # Fallback: if keep_key is explicitly one of the cluster keys,
            # remove the other fact that matches
            if key_to_remove is None:
                cluster_keys = {k for k, _ in cluster}
                if keep_key in cluster_keys:
                    for key, data in cluster:
                        if key != keep_key:
                            content = data.get("content", "")
                            if content in (fact_a_text, fact_b_text):
                                key_to_remove = key
                                break

            if key_to_remove:
                deleted = memory.delete("semantic", key_to_remove)
                if deleted:
                    removed += 1
                    logger.info(
                        "Contradiction resolved: removed key=%s, kept key=%s",
                        key_to_remove,
                        keep_key,
                    )

    return removed


def _cluster_memories(
    items: list[tuple[str, dict[str, Any]]],
    threshold: float = 0.3,
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Cluster memory items by Jaccard similarity of their content words.

    Returns a list of clusters (each cluster is a list of (key, data) pairs).
    Items with no overlap above the threshold form singleton clusters.
    """

    def _tokenize(text: str) -> set[str]:
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    def _jaccard(s1: set[str], s2: set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    token_sets = [
        (key, data, _tokenize(data.get("content", "")))
        for key, data in items
    ]

    clusters: list[list[tuple[str, dict[str, Any]]]] = []
    assigned = [False] * len(token_sets)

    for i, (key_i, data_i, tokens_i) in enumerate(token_sets):
        if assigned[i]:
            continue
        cluster: list[tuple[str, dict[str, Any]]] = [(key_i, data_i)]
        assigned[i] = True
        for j, (key_j, data_j, tokens_j) in enumerate(token_sets):
            if assigned[j] or j == i:
                continue
            if _jaccard(tokens_i, tokens_j) >= threshold:
                cluster.append((key_j, data_j))
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def _load_traces(traces_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not traces_dir.exists():
        return []
    files = sorted(traces_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    traces = []
    for f in files[:limit]:
        try:
            traces.append(json.loads(f.read_text()))
        except Exception:
            continue
    return traces


def _summarize_trace(trace: dict[str, Any]) -> str:
    task = trace.get("task", "unknown")[:100]
    duration = trace.get("duration_seconds", "?")
    steps = trace.get("step_count", 0)
    return f"Task: {task} | Duration: {duration}s, Steps: {steps}"
