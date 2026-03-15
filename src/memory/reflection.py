"""Post-run reflection engine.

Analyzes completed agent trajectories and generates reflections
that are stored as episodic and semantic memories.
"""

import json
import logging
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import REFLECTION_PROMPT
from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _safe_json_parse(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse reflection JSON, using raw text")
        return {
            "strategy": cleaned,
            "worked_well": "",
            "improvements": "",
            "facts": [],
            "patterns": [],
        }


def generate_reflection(
    llm: BaseChatModel,
    task: str,
    output: str,
    tool_calls: list[dict[str, Any]],
    grader_results: dict[str, Any],
) -> dict[str, Any]:
    """Generate a reflection on an agent run using the LLM."""
    prompt = REFLECTION_PROMPT.format(
        task=task,
        output=output[:3000],
        tool_calls=json.dumps(tool_calls[:20], indent=2),
        grader_results=json.dumps(grader_results, indent=2),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return _safe_json_parse(response.content)


def reflect_and_store(
    llm: BaseChatModel,
    memory_store: MemoryStore,
    run_id: str,
    task: str,
    output: str,
    tool_calls: list[dict[str, Any]],
    grader_results: dict[str, Any],
) -> dict[str, Any]:
    """Run reflection and store results as episodic + semantic memories."""
    reflection = generate_reflection(llm, task, output, tool_calls, grader_results)

    # Store episodic memory (full run experience)
    episodic_data = {
        "run_id": run_id,
        "task": task,
        "summary": reflection.get("strategy", ""),
        "worked_well": reflection.get("worked_well", ""),
        "improvements": reflection.get("improvements", ""),
        "score": grader_results.get("average_score", 0),
        "output_preview": output[:500],
    }
    memory_store.store("episodic", run_id, episodic_data)

    # Store semantic memories (extracted facts and patterns)
    facts = reflection.get("facts", [])
    patterns = reflection.get("patterns", [])

    for fact in facts:
        if isinstance(fact, str) and fact.strip():
            fact_id = f"fact-{uuid.uuid4().hex[:8]}"
            memory_store.store(
                "semantic",
                fact_id,
                {"type": "fact", "content": fact, "source_run": run_id},
            )

    for pattern in patterns:
        if isinstance(pattern, str) and pattern.strip():
            pattern_id = f"pattern-{uuid.uuid4().hex[:8]}"
            memory_store.store(
                "semantic",
                pattern_id,
                {"type": "pattern", "content": pattern, "source_run": run_id},
            )

    logger.info(
        "Reflection stored: %d facts, %d patterns for run %s",
        len(facts), len(patterns), run_id,
    )
    return reflection
