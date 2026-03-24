# Reliable Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the evolution engine's grading, prompt optimization, and skill learning more reliable through rubric anchoring, multi-judge scoring, claim verification, factual spot-checking, and pairwise prompt comparison.

**Architecture:** In-place enhancement of existing graders and optimizer. Three new files (multi_judge.py, claim_verification.py, fact_checker.py) plus modifications to prompts, analyzer, state, and optimizer. LangGraph analyzer restructured from sequential to parallel grader execution.

**Tech Stack:** Python 3.12, LangGraph, LangChain, ThreadPoolExecutor, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-reliable-grading-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agent/prompts.py` | Modify | All new prompt templates (anchored rubrics, 6 perspective prompts, claim/pairwise/fact-check prompts) |
| `src/evolution/graders/multi_judge.py` | Create | Multi-judge wrapper: runs 3 perspective judges in parallel, returns median score |
| `src/evolution/graders/claim_verification.py` | Create | Claim extraction + verification grader, blends with spot-check |
| `src/evolution/graders/fact_checker.py` | Create | Factual spot-check: selects verifiable claims, searches, compares |
| `src/evolution/analyzer.py` | Modify | Parallel LangGraph layout, swap to multi-judge, add 4th grader |
| `src/evolution/state.py` | Modify | Add `claim_verification` field to `AnalyzerState` |
| `src/evolution/prompt_optimizer.py` | Modify | Pairwise validation before saving new prompts |
| `tests/unit/test_multi_judge.py` | Create | Tests for multi-judge aggregation, error handling, parallelism |
| `tests/unit/test_claim_verification.py` | Create | Tests for claim extraction, verification, score calculation |
| `tests/unit/test_fact_checker.py` | Create | Tests for claim selection, spot-check scoring, failure handling |
| `tests/unit/test_analyzer_parallel.py` | Create | Tests for parallel analyzer graph, 4-grader classification |
| `tests/unit/test_pairwise.py` | Create | Tests for pairwise scoring, task selection, decision logic |

---

### Task 1: Rubric Anchoring — Update Grading Prompts

**Files:**
- Modify: `src/agent/prompts.py:114-142`

- [ ] **Step 1: Write test for anchored prompts**

Create `tests/unit/test_prompts.py`:
```python
"""Tests that grading prompts contain rubric anchors."""

from src.agent.prompts import TASK_COMPLETION_PROMPT, QUALITY_PROMPT


def test_task_completion_prompt_has_anchors():
    assert "0.9" in TASK_COMPLETION_PROMPT
    assert "0.5" in TASK_COMPLETION_PROMPT
    assert "0.2" in TASK_COMPLETION_PROMPT
    assert "cited sources" in TASK_COMPLETION_PROMPT.lower()


def test_quality_prompt_has_anchors():
    assert "0.9" in QUALITY_PROMPT
    assert "0.5" in QUALITY_PROMPT
    assert "0.2" in QUALITY_PROMPT
    assert "structure" in QUALITY_PROMPT.lower() or "organized" in QUALITY_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prompts.py -v`
Expected: FAIL — current prompts don't contain anchor scores

- [ ] **Step 3: Update TASK_COMPLETION_PROMPT with anchored rubric**

In `src/agent/prompts.py`, replace the existing `TASK_COMPLETION_PROMPT`:

```python
TASK_COMPLETION_PROMPT = (
    "You are judging whether a research agent successfully completed its task.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "Score the output on these criteria:\n"
    "1. Did the agent address the core question?\n"
    "2. Is the output well-structured and readable?\n"
    "3. Are claims supported by cited sources?\n"
    "4. Is the analysis thorough and accurate?\n\n"
    "## Scoring Guide\n"
    "- 0.9: Directly answers the question with 3+ cited sources, structured sections, "
    "no factual errors, thorough coverage of all subtopics asked about.\n"
    "- 0.5: Partially addresses the question but misses key aspects, 1-2 sources, "
    "some structure but gaps in analysis, may have minor inaccuracies.\n"
    "- 0.2: Off-topic or superficial, no sources, unstructured, factual errors "
    "or hallucinated claims, fails to answer the core question.\n\n"
    "Respond as JSON with keys:\n"
    "- score: float between 0.0 and 1.0\n"
    "- passed: boolean (true if score >= 0.75)\n"
    "- reasoning: brief explanation of the score"
)
```

- [ ] **Step 4: Update QUALITY_PROMPT with anchored rubric**

In `src/agent/prompts.py`, replace the existing `QUALITY_PROMPT`:

```python
QUALITY_PROMPT = (
    "You are judging the quality of a research agent's output.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "Rate the output quality on these dimensions:\n"
    "1. **Accuracy** (0-1): Are the facts correct and well-sourced?\n"
    "2. **Depth** (0-1): Is the analysis thorough?\n"
    "3. **Clarity** (0-1): Is the writing clear and well-organized?\n"
    "4. **Relevance** (0-1): Does the output stay focused on the task?\n\n"
    "## Scoring Guide\n"
    "- 0.9: Clear logical structure with sections, accurate facts with citations, "
    "deep analysis that covers nuances and trade-offs, stays tightly focused on the task.\n"
    "- 0.5: Readable but loosely organized, mostly accurate but some unsourced claims, "
    "surface-level analysis, some tangential content.\n"
    "- 0.2: Disorganized or incoherent, factual errors, shallow or repetitive, "
    "significant off-topic content or filler.\n\n"
    "Respond as JSON with keys:\n"
    "- accuracy: float\n- depth: float\n- clarity: float\n- relevance: float\n"
    "- overall_score: float (weighted average)\n"
    "- reasoning: brief explanation"
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_prompts.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_prompts.py src/agent/prompts.py
git commit -m "feat: add rubric anchoring to grading prompts"
```

---

### Task 2: Add Perspective Judge Prompts

**Files:**
- Modify: `src/agent/prompts.py`
- Modify: `tests/unit/test_prompts.py`

- [ ] **Step 1: Write test for perspective prompts**

Append to `tests/unit/test_prompts.py`:
```python
from src.agent.prompts import (
    TC_COMPLETENESS_PROMPT,
    TC_EVIDENCE_PROMPT,
    TC_ACCURACY_PROMPT,
    Q_STRUCTURE_PROMPT,
    Q_DEPTH_PROMPT,
    Q_RELEVANCE_PROMPT,
)


def test_perspective_prompts_have_required_placeholders():
    for prompt in [
        TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT,
        Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT,
    ]:
        assert "{task}" in prompt, f"Missing {{task}} placeholder in {prompt[:50]}"
        assert "{output}" in prompt, f"Missing {{output}} placeholder in {prompt[:50]}"
        assert "0.9" in prompt, f"Missing high anchor in {prompt[:50]}"
        assert "0.5" in prompt, f"Missing mid anchor in {prompt[:50]}"
        assert "0.2" in prompt, f"Missing low anchor in {prompt[:50]}"


def test_perspective_prompts_request_json():
    for prompt in [
        TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT,
        Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT,
    ]:
        assert "json" in prompt.lower(), f"Missing JSON instruction in {prompt[:50]}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prompts.py::test_perspective_prompts_have_required_placeholders -v`
Expected: FAIL — imports don't exist yet

- [ ] **Step 3: Add 6 perspective prompts to prompts.py**

Add after the existing `QUALITY_PROMPT` in `src/agent/prompts.py`:

```python
# --- Multi-Judge Perspective Prompts (Task Completion) ---

TC_COMPLETENESS_PROMPT = (
    "You are judging a research agent's output from the perspective of COMPLETENESS.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: Covers all aspects and subtopics of the question, no significant gaps.\n"
    "- 0.5: Addresses the main question but misses 1-2 important subtopics.\n"
    "- 0.2: Only touches on the topic superficially, major aspects missing.\n\n"
    "Focus primarily on whether the output covers all aspects of the question. "
    "Other quality dimensions (sources, accuracy, structure) are handled by other judges "
    "-- your job is only completeness of coverage.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)

TC_EVIDENCE_PROMPT = (
    "You are judging a research agent's output from the perspective of EVIDENCE.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: 3+ claims backed by cited sources with URLs or named references.\n"
    "- 0.5: Some claims sourced but others stated without evidence.\n"
    "- 0.2: No sources cited, or sources are fabricated/irrelevant.\n\n"
    "Focus primarily on whether claims are backed by cited sources. "
    "A well-sourced output with minor structural issues should still score high. "
    "Other quality dimensions are handled by other judges.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)

TC_ACCURACY_PROMPT = (
    "You are judging a research agent's output from the perspective of ACCURACY.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: All facts appear correct, reasoning is sound, no contradictions.\n"
    "- 0.5: Mostly accurate but contains 1-2 questionable claims or minor errors.\n"
    "- 0.2: Contains clear factual errors, contradictions, or fabricated information.\n\n"
    "Focus primarily on factual correctness and reasoning quality. "
    "Other quality dimensions are handled by other judges.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)

# --- Multi-Judge Perspective Prompts (Quality) ---

Q_STRUCTURE_PROMPT = (
    "You are judging a research agent's output from the perspective of STRUCTURE.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: Clear sections with headings, logical flow, easy to scan and read.\n"
    "- 0.5: Some structure but inconsistent formatting or unclear organization.\n"
    "- 0.2: No clear structure, wall of text, hard to follow.\n\n"
    "Focus primarily on organization, readability, and formatting. "
    "Other quality dimensions are handled by other judges.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)

Q_DEPTH_PROMPT = (
    "You are judging a research agent's output from the perspective of DEPTH.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: Thorough analysis covering nuances, trade-offs, and multiple perspectives.\n"
    "- 0.5: Addresses the topic but stays surface-level, lacks nuance.\n"
    "- 0.2: Shallow or repetitive, no real analysis beyond restating the obvious.\n\n"
    "Focus primarily on thoroughness of analysis and depth of insight. "
    "Other quality dimensions are handled by other judges.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)

Q_RELEVANCE_PROMPT = (
    "You are judging a research agent's output from the perspective of RELEVANCE.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output\n{output}\n\n"
    "## Scoring Guide\n"
    "- 0.9: Every paragraph directly serves the task, no tangents or filler.\n"
    "- 0.5: Mostly on-topic but includes some tangential content or padding.\n"
    "- 0.2: Significant off-topic content, filler, or answers a different question.\n\n"
    "Focus primarily on whether the output stays focused on the task. "
    "Other quality dimensions are handled by other judges.\n\n"
    'Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}'
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/prompts.py tests/unit/test_prompts.py
git commit -m "feat: add 6 perspective judge prompts for multi-judge grading"
```

---

### Task 3: Multi-Judge Grader

**Files:**
- Create: `src/evolution/graders/multi_judge.py`
- Create: `tests/unit/test_multi_judge.py`

- [ ] **Step 1: Write tests for multi-judge aggregation**

Create `tests/unit/test_multi_judge.py`:
```python
"""Tests for multi-judge grading with median aggregation."""

import json
from unittest.mock import MagicMock

from src.evolution.graders.multi_judge import (
    _aggregate_judge_scores,
    _run_single_judge,
    multi_judge_task_completion,
    multi_judge_quality,
)


def test_aggregate_median_of_three():
    scores = [0.8, 0.6, 0.9]
    score, reasoning = _aggregate_judge_scores(scores, ["r1", "r2", "r3"])
    assert score == 0.8  # median


def test_aggregate_low_agreement_flag():
    scores = [0.3, 0.9, 0.7]  # max - min = 0.6 > 0.3
    score, reasoning = _aggregate_judge_scores(scores, ["r1", "r2", "r3"])
    assert "low_agreement" in reasoning


def test_aggregate_high_agreement_no_flag():
    scores = [0.7, 0.8, 0.75]  # max - min = 0.1 < 0.3
    score, reasoning = _aggregate_judge_scores(scores, ["r1", "r2", "r3"])
    assert "low_agreement" not in reasoning


def test_aggregate_single_score_fallback():
    scores = [0.6]
    score, reasoning = _aggregate_judge_scores(scores, ["r1"])
    assert score == 0.6


def test_aggregate_empty_returns_default():
    score, reasoning = _aggregate_judge_scores([], [])
    assert score == 0.5
    assert "grader_error" in reasoning


def _make_mock_llm(score: float) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=json.dumps({"score": score, "reasoning": "test"})
    )
    return llm


def test_run_single_judge_returns_score():
    llm = _make_mock_llm(0.85)
    score, reasoning = _run_single_judge(llm, "prompt {task} {output}", "task1", "output1")
    assert score == 0.85
    assert reasoning == "test"


def test_run_single_judge_handles_parse_error():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="not json at all")
    score, reasoning = _run_single_judge(llm, "prompt {task} {output}", "task1", "output1")
    assert score == 0.5  # default on parse failure


def test_run_single_judge_handles_exception():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM timeout")
    score, reasoning = _run_single_judge(llm, "prompt {task} {output}", "task1", "output1")
    assert score is None  # signal to exclude from median


def test_multi_judge_task_completion_returns_grader_result():
    llm = _make_mock_llm(0.8)
    result = multi_judge_task_completion(llm, "research task", "some output")
    assert isinstance(result, dict)
    assert result["name"] == "task_completion"
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["passed"], bool)


def test_multi_judge_quality_returns_grader_result():
    llm = _make_mock_llm(0.7)
    result = multi_judge_quality(llm, "research task", "some output")
    assert isinstance(result, dict)
    assert result["name"] == "quality"
    assert 0.0 <= result["score"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_multi_judge.py -v`
Expected: FAIL -- module doesn't exist

- [ ] **Step 3: Implement multi_judge.py**

Create `src/evolution/graders/multi_judge.py`:
```python
"""Multi-judge grading with perspective-specific judges and median aggregation.

Each LLM grader (task_completion, quality) is replaced by 3 perspective judges
that run in parallel. Scores are aggregated via median for outlier resistance.
"""

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import (
    TC_ACCURACY_PROMPT,
    TC_COMPLETENESS_PROMPT,
    TC_EVIDENCE_PROMPT,
    Q_DEPTH_PROMPT,
    Q_RELEVANCE_PROMPT,
    Q_STRUCTURE_PROMPT,
    TASK_COMPLETION_PROMPT,
    QUALITY_PROMPT,
)
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)

LOW_AGREEMENT_THRESHOLD = 0.3


def _parse_judge_response(text: str) -> dict:
    """Parse JSON from a judge LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _run_single_judge(
    llm: BaseChatModel,
    prompt_template: str,
    task: str,
    output: str,
) -> tuple[float | None, str]:
    """Run a single judge. Returns (score, reasoning) or (None, error) on failure."""
    try:
        prompt = prompt_template.format(task=task, output=output[:4000])
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_judge_response(response.content)
        score = parsed.get("score")
        if score is None:
            return 0.5, "parse_error: no score in response"
        return float(score), parsed.get("reasoning", "")
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return None, str(exc)


def _aggregate_judge_scores(
    scores: list[float],
    reasonings: list[str],
) -> tuple[float, str]:
    """Aggregate judge scores via median. Flag low agreement."""
    if not scores:
        return 0.5, "grader_error: no judge scores available"

    if len(scores) == 1:
        return scores[0], reasonings[0] if reasonings else ""

    median_score = statistics.median(scores)
    spread = max(scores) - min(scores)

    parts = [f"judges={[round(s, 2) for s in scores]}", f"median={median_score:.2f}"]
    if spread > LOW_AGREEMENT_THRESHOLD:
        parts.append("low_agreement")
    parts.extend(r for r in reasonings if r)

    return round(median_score, 3), "; ".join(parts)


def _run_multi_judge(
    llm: BaseChatModel,
    judge_prompts: list[str],
    fallback_prompt: str,
    task: str,
    output: str,
) -> tuple[float, str]:
    """Run multiple judges in parallel, aggregate via median.

    If 2+ judges fail, retries with the fallback (original single) prompt.
    If that also fails, returns (0.5, "grader_error").
    """
    scores: list[float] = []
    reasonings: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_single_judge, llm, p, task, output): i
            for i, p in enumerate(judge_prompts)
        }
        for future in as_completed(futures):
            score, reasoning = future.result()
            if score is not None:
                scores.append(score)
                reasonings.append(reasoning)

    if len(scores) >= 2:
        return _aggregate_judge_scores(scores, reasonings)

    # Fallback: too many judges failed, try single prompt
    logger.warning(
        "Only %d/%d judges succeeded, retrying with fallback prompt",
        len(scores), len(judge_prompts),
    )
    score, reasoning = _run_single_judge(llm, fallback_prompt, task, output)
    if score is not None:
        return score, f"fallback: {reasoning}"
    return 0.5, "grader_error: all judges and fallback failed"


def multi_judge_task_completion(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade task completion using 3 perspective judges."""
    score, reasoning = _run_multi_judge(
        llm,
        judge_prompts=[TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT],
        fallback_prompt=TASK_COMPLETION_PROMPT,
        task=task,
        output=output,
    )
    return GraderResult(
        name="task_completion",
        score=score,
        passed=score >= 0.75,
        reasoning=reasoning,
    )


def multi_judge_quality(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade output quality using 3 perspective judges."""
    score, reasoning = _run_multi_judge(
        llm,
        judge_prompts=[Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT],
        fallback_prompt=QUALITY_PROMPT,
        task=task,
        output=output,
    )
    return GraderResult(
        name="quality",
        score=score,
        passed=score >= 0.75,
        reasoning=reasoning,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_multi_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evolution/graders/multi_judge.py tests/unit/test_multi_judge.py
git commit -m "feat: add multi-judge grading with parallel perspective judges"
```

---

### Task 4: Claim Verification Grader + Prompts

**Files:**
- Create: `src/evolution/graders/claim_verification.py`
- Create: `tests/unit/test_claim_verification.py`
- Modify: `src/agent/prompts.py`

- [ ] **Step 1: Add claim prompts to prompts.py**

Add to `src/agent/prompts.py`:
```python
# --- Claim Verification Prompts ---

CLAIM_EXTRACTION_PROMPT = (
    "Extract the key factual claims from this research output. Only extract claims "
    "that appear verbatim or are clearly stated in the text. Do not infer or "
    "fabricate claims. Extract up to 10 claims.\n\n"
    "## Agent Output\n{output}\n\n"
    'Respond as JSON: {"claims": ["claim 1", "claim 2", ...]}'
)

CLAIM_VERIFICATION_PROMPT = (
    "Verify each claim against the source output. For each claim, determine:\n"
    "- Is it supported by a cited source in the output?\n"
    "- Is it contradicted by other claims in the output?\n"
    "- Is it suspiciously specific without any source?\n\n"
    "## Claims\n{claims}\n\n"
    "## Full Output\n{output}\n\n"
    'Respond as JSON: {"verdicts": [{"claim": "...", '
    '"verdict": "supported|unsupported|contradicted", "reasoning": "..."}]}'
)
```

- [ ] **Step 2: Write tests for claim verification**

Create `tests/unit/test_claim_verification.py`:
```python
"""Tests for claim-level verification grader."""

import json
from unittest.mock import MagicMock

from src.evolution.graders.claim_verification import (
    _compute_consistency_score,
    _extract_claims,
    _verify_claims,
    grade_claims,
)


def _make_llm_returning(data: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(data))
    return llm


def test_compute_consistency_all_supported():
    verdicts = [
        {"claim": "a", "verdict": "supported"},
        {"claim": "b", "verdict": "supported"},
    ]
    assert _compute_consistency_score(verdicts) == 1.0


def test_compute_consistency_mixed():
    verdicts = [
        {"claim": "a", "verdict": "supported"},
        {"claim": "b", "verdict": "unsupported"},
        {"claim": "c", "verdict": "contradicted"},
    ]
    assert abs(_compute_consistency_score(verdicts) - 1 / 3) < 0.01


def test_compute_consistency_empty():
    assert _compute_consistency_score([]) == 0.5


def test_extract_claims_happy_path():
    llm = _make_llm_returning({"claims": ["claim1", "claim2"]})
    claims = _extract_claims(llm, "some output text")
    assert claims == ["claim1", "claim2"]


def test_extract_claims_parse_failure():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="not json")
    claims = _extract_claims(llm, "output")
    assert claims == []


def test_verify_claims_happy_path():
    llm = _make_llm_returning({
        "verdicts": [
            {"claim": "c1", "verdict": "supported", "reasoning": "ok"},
        ]
    })
    verdicts = _verify_claims(llm, ["c1"], "output")
    assert len(verdicts) == 1
    assert verdicts[0]["verdict"] == "supported"


def test_grade_claims_returns_grader_result():
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"claims": ["EV sales grew 35%"]})),
        MagicMock(content=json.dumps({
            "verdicts": [{"claim": "EV sales grew 35%", "verdict": "supported", "reasoning": "cited"}]
        })),
    ]
    result = grade_claims(llm, "task", "output with EV sales grew 35%")
    assert result["name"] == "claim_verification"
    assert result["score"] == 1.0
    assert result["passed"] is True


def test_grade_claims_no_claims_extracted():
    llm = _make_llm_returning({"claims": []})
    result = grade_claims(llm, "task", "very short output")
    assert result["score"] == 0.5
    assert result["passed"] is False


def test_grade_claims_with_spot_check_blending():
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"claims": ["claim1"]})),
        MagicMock(content=json.dumps({
            "verdicts": [{"claim": "claim1", "verdict": "supported", "reasoning": "ok"}]
        })),
    ]
    # consistency=1.0, spot_check=0.5 -> 0.6*1.0 + 0.4*0.5 = 0.8
    result = grade_claims(llm, "task", "output", spot_check_score=0.5)
    assert abs(result["score"] - 0.8) < 0.01
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_claim_verification.py -v`
Expected: FAIL -- module doesn't exist

- [ ] **Step 4: Implement claim_verification.py**

Create `src/evolution/graders/claim_verification.py`:
```python
"""Claim-level verification grader.

Extracts factual claims from agent output, then verifies each claim for
internal consistency and source alignment. Optionally blends with
factual spot-check score from fact_checker module.
"""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)

CLAIM_PASS_THRESHOLD = 0.6


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse claim grader JSON response")
        return {}


def _extract_claims(llm: BaseChatModel, output: str) -> list[str]:
    """Extract factual claims from agent output."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(output=output[:4000])
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        claims = parsed.get("claims", [])
        return [c for c in claims if isinstance(c, str) and c.strip()]
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return []


def _verify_claims(
    llm: BaseChatModel,
    claims: list[str],
    output: str,
) -> list[dict]:
    """Verify extracted claims against the output."""
    claims_str = "\n".join(f"- {c}" for c in claims)
    prompt = CLAIM_VERIFICATION_PROMPT.format(claims=claims_str, output=output[:4000])
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        return parsed.get("verdicts", [])
    except Exception as exc:
        logger.warning("Claim verification failed: %s", exc)
        return []


def _compute_consistency_score(verdicts: list[dict]) -> float:
    """Compute internal consistency score from claim verdicts."""
    if not verdicts:
        return 0.5  # fallback for no claims

    supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return round(supported / len(verdicts), 3)


def grade_claims(
    llm: BaseChatModel,
    task: str,
    output: str,
    spot_check_score: float | None = None,
) -> GraderResult:
    """Grade agent output via claim extraction + verification.

    Args:
        llm: Language model for extraction and verification.
        task: The original research task.
        output: The agent's output text.
        spot_check_score: Optional score from fact_checker (0-1).
            If provided, blends: 0.6 * consistency + 0.4 * spot_check.
            If None, uses consistency score only.
    """
    claims = _extract_claims(llm, output)
    if not claims:
        return GraderResult(
            name="claim_verification",
            score=0.5,
            passed=False,
            reasoning="No factual claims extracted from output",
        )

    verdicts = _verify_claims(llm, claims, output)
    if not verdicts:
        return GraderResult(
            name="claim_verification",
            score=0.5,
            passed=False,
            reasoning="Claim verification returned no verdicts",
        )

    consistency_score = _compute_consistency_score(verdicts)

    if spot_check_score is not None:
        final_score = round(0.6 * consistency_score + 0.4 * spot_check_score, 3)
        reasoning_prefix = (
            f"consistency={consistency_score:.2f}, spot_check={spot_check_score:.2f}, "
            f"blended={final_score:.2f}"
        )
    else:
        final_score = consistency_score
        reasoning_prefix = f"consistency={consistency_score:.2f} (no spot-check)"

    verdict_summary = ", ".join(
        f"{v.get('claim', '?')[:40]}={v.get('verdict', '?')}" for v in verdicts[:5]
    )

    return GraderResult(
        name="claim_verification",
        score=final_score,
        passed=final_score >= CLAIM_PASS_THRESHOLD,
        reasoning=f"{reasoning_prefix}; {verdict_summary}",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_claim_verification.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent/prompts.py src/evolution/graders/claim_verification.py tests/unit/test_claim_verification.py
git commit -m "feat: add claim-level verification grader"
```

---

### Task 5: Factual Spot-Check via Search

**Files:**
- Create: `src/evolution/graders/fact_checker.py`
- Create: `tests/unit/test_fact_checker.py`
- Modify: `src/agent/prompts.py`

- [ ] **Step 1: Add fact-check prompts to prompts.py**

Add to `src/agent/prompts.py`:
```python
# --- Factual Spot-Check Prompts ---

CLAIM_SELECTION_PROMPT = (
    "From these claims, select the 2-3 most objectively verifiable ones. "
    "Prefer claims with specific numbers, dates, percentages, or named entities. "
    "Avoid subjective or opinion-based claims.\n\n"
    "## Claims\n{claims}\n\n"
    'Respond as JSON: {"selected": ["claim 1", "claim 2"]}'
)

FACT_CHECK_PROMPT = (
    "Does the search evidence support, contradict, or not address this claim?\n\n"
    "## Claim\n{claim}\n\n"
    "## Search Results\n{search_results}\n\n"
    'Respond as JSON: {"verdict": "corroborated|contradicted|inconclusive", '
    '"reasoning": "brief explanation"}'
)
```

- [ ] **Step 2: Write tests for fact checker**

Create `tests/unit/test_fact_checker.py`:
```python
"""Tests for factual spot-check via search."""

import json
from unittest.mock import MagicMock

from src.evolution.graders.fact_checker import (
    _compute_spot_check_score,
    _select_verifiable_claims,
    _check_single_claim,
    spot_check_claims,
)


def _make_llm_returning(data: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(data))
    return llm


def test_compute_score_all_corroborated():
    verdicts = ["corroborated", "corroborated"]
    assert _compute_spot_check_score(verdicts) == 1.0


def test_compute_score_all_contradicted():
    verdicts = ["contradicted", "contradicted"]
    assert _compute_spot_check_score(verdicts) == 0.0


def test_compute_score_mixed():
    verdicts = ["corroborated", "inconclusive", "contradicted"]
    assert abs(_compute_spot_check_score(verdicts) - 0.5) < 0.01


def test_compute_score_empty():
    assert _compute_spot_check_score([]) is None


def test_select_verifiable_claims():
    llm = _make_llm_returning({"selected": ["claim A", "claim B"]})
    result = _select_verifiable_claims(llm, ["claim A", "claim B", "claim C"])
    assert result == ["claim A", "claim B"]


def test_check_single_claim_corroborated():
    llm = _make_llm_returning({"verdict": "corroborated", "reasoning": "matches"})
    search = MagicMock()
    search._run.return_value = "Search result text confirming claim"
    verdict = _check_single_claim(llm, search, "Tesla delivered 1.8M vehicles")
    assert verdict == "corroborated"


def test_check_single_claim_search_fails():
    llm = MagicMock()
    search = MagicMock()
    search._run.side_effect = Exception("network error")
    verdict = _check_single_claim(llm, search, "some claim")
    assert verdict is None


def test_spot_check_claims_all_searches_fail():
    llm = _make_llm_returning({"selected": ["claim1"]})
    search = MagicMock()
    search._run.side_effect = Exception("network error")
    result = spot_check_claims(llm, search, ["claim1", "claim2"])
    assert result is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_fact_checker.py -v`
Expected: FAIL -- module doesn't exist

- [ ] **Step 4: Implement fact_checker.py**

Create `src/evolution/graders/fact_checker.py`:
```python
"""Factual spot-check via web search.

Selects the most verifiable claims from a set, searches for each,
and compares search results against the claims. Returns an aggregated
spot-check score that can be blended into the claim_verification grader.
"""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from src.agent.prompts import CLAIM_SELECTION_PROMPT, FACT_CHECK_PROMPT

logger = logging.getLogger(__name__)

VERDICT_SCORES = {
    "corroborated": 1.0,
    "inconclusive": 0.5,
    "contradicted": 0.0,
}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _select_verifiable_claims(
    llm: BaseChatModel,
    claims: list[str],
) -> list[str]:
    """Select 2-3 most objectively verifiable claims."""
    claims_str = "\n".join(f"- {c}" for c in claims)
    prompt = CLAIM_SELECTION_PROMPT.format(claims=claims_str)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        selected = parsed.get("selected", [])
        return [c for c in selected if isinstance(c, str) and c.strip()]
    except Exception as exc:
        logger.warning("Claim selection failed: %s", exc)
        return claims[:2]  # fallback: take first 2


def _check_single_claim(
    llm: BaseChatModel,
    search_tool: BaseTool,
    claim: str,
) -> str | None:
    """Search for a claim and check if evidence supports it.

    Returns verdict string or None if search failed.
    """
    try:
        search_results = search_tool._run(claim)
    except Exception as exc:
        logger.warning("Search failed for claim '%s': %s", claim[:50], exc)
        return None

    if not search_results or "no results" in search_results.lower():
        return None

    prompt = FACT_CHECK_PROMPT.format(
        claim=claim,
        search_results=search_results[:3000],
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        verdict = parsed.get("verdict", "inconclusive")
        return verdict if verdict in VERDICT_SCORES else "inconclusive"
    except Exception as exc:
        logger.warning("Fact-check LLM failed for claim '%s': %s", claim[:50], exc)
        return None


def _compute_spot_check_score(verdicts: list[str]) -> float | None:
    """Compute aggregated spot-check score from verdicts.

    Returns None if no verdicts (all searches failed).
    """
    if not verdicts:
        return None
    scores = [VERDICT_SCORES.get(v, 0.5) for v in verdicts]
    return round(sum(scores) / len(scores), 3)


def spot_check_claims(
    llm: BaseChatModel,
    search_tool: BaseTool,
    claims: list[str],
) -> float | None:
    """Run factual spot-check on a list of claims.

    Returns spot-check score (0-1) or None if check could not be performed.
    """
    if not claims:
        return None

    selected = _select_verifiable_claims(llm, claims)
    if not selected:
        return None

    # Process sequentially to respect search API rate limits
    verdicts: list[str] = []
    for claim in selected:
        verdict = _check_single_claim(llm, search_tool, claim)
        if verdict is not None:
            verdicts.append(verdict)

    return _compute_spot_check_score(verdicts)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_fact_checker.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent/prompts.py src/evolution/graders/fact_checker.py tests/unit/test_fact_checker.py
git commit -m "feat: add factual spot-check via web search"
```

---

### Task 6: Update State + Parallel Analyzer Graph

**Files:**
- Modify: `src/evolution/state.py:29-37`
- Modify: `src/evolution/analyzer.py`
- Create: `tests/unit/test_analyzer_parallel.py`

- [ ] **Step 1: Write tests for updated classification and parallel analyzer**

Create `tests/unit/test_analyzer_parallel.py`:
```python
"""Tests for parallel analyzer graph with 4 graders."""

import typing

from src.evolution.analyzer import classify_trajectory
from src.evolution.state import AnalyzerState, GraderResult


def _gr(name: str, score: float, passed: bool) -> GraderResult:
    return GraderResult(name=name, score=score, passed=passed, reasoning="test")


def test_classify_4_graders_all_pass():
    results = [
        _gr("tc", 0.9, True), _gr("eff", 0.8, True),
        _gr("q", 0.85, True), _gr("cv", 0.7, True),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "successful"
    assert avg >= 0.75


def test_classify_4_graders_2_pass_high_avg():
    results = [
        _gr("tc", 0.8, True), _gr("eff", 0.3, False),
        _gr("q", 0.8, True), _gr("cv", 0.5, False),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "partial"


def test_classify_4_graders_1_pass_low_avg():
    results = [
        _gr("tc", 0.8, True), _gr("eff", 0.2, False),
        _gr("q", 0.3, False), _gr("cv", 0.2, False),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "failed"


def test_classify_4_graders_claim_verification_included_in_avg():
    results = [
        _gr("tc", 0.9, True), _gr("eff", 0.9, True),
        _gr("q", 0.9, True), _gr("cv", 0.0, False),
    ]
    classification, avg = classify_trajectory(results)
    # avg = (0.9+0.9+0.9+0.0)/4 = 0.675
    assert 0.67 <= avg <= 0.68


def test_analyzer_state_has_claim_verification_field():
    hints = typing.get_type_hints(AnalyzerState)
    assert "claim_verification" in hints
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_analyzer_parallel.py -v`
Expected: FAIL -- `claim_verification` not in AnalyzerState

- [ ] **Step 3: Update AnalyzerState in state.py**

In `src/evolution/state.py`, add `claim_verification` to `AnalyzerState`:
```python
class AnalyzerState(TypedDict):
    """State for the trajectory analyzer LangGraph workflow."""

    trajectory: Trajectory
    task_completion: GraderResult
    efficiency: GraderResult
    quality: GraderResult
    claim_verification: GraderResult
    classification: str
    average_score: float
```

- [ ] **Step 4: Restructure analyzer.py for parallel grading + 4th grader**

Replace `src/evolution/analyzer.py` entirely. Key changes:
- Import `multi_judge_task_completion` and `multi_judge_quality` instead of originals
- Import `grade_claims` from claim_verification and `spot_check_claims` from fact_checker
- Add `search_tool: BaseTool | None = None` parameter to `build_analyzer_graph` and `analyze_trajectory`
- Add `node_grade_claims` node that runs claim extraction, optional spot-check, and `grade_claims`
- Restructure edges: all 4 graders start from START, all feed into classify
- Update `node_classify` to include all 4 grader results
- Update `initial_state` to include `claim_verification` default
- Update `AnalysisResult` construction to include 4 grader results

See spec Section 2 (parallel layout) and Section 3 (claim verification integration) for full details.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_analyzer_parallel.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/evolution/state.py src/evolution/analyzer.py tests/unit/test_analyzer_parallel.py
git commit -m "feat: parallel 4-grader analyzer with claim verification"
```

---

### Task 7: Update Orchestrator for Search Tool Pass-Through

**Files:**
- Modify: `src/evolution/orchestrator.py`

- [ ] **Step 1: Create search tool in build_orchestrator_graph**

In `src/evolution/orchestrator.py`, inside `build_orchestrator_graph`, after `trace_fetcher = TraceFetcher(settings)`:
```python
from src.tools.search import create_search_tool
try:
    search_tool = create_search_tool(settings)
except Exception:
    search_tool = None
    logger.warning("Search tool unavailable for fact-checking -- spot-check disabled")
```

- [ ] **Step 2: Pass search_tool to analyze_trajectory**

In `node_analyze`, update the call:
```python
analysis = analyze_trajectory(llm, traj, search_tool=search_tool)
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/evolution/orchestrator.py
git commit -m "feat: pass search tool to analyzer for factual spot-checking"
```

---

### Task 8: Pairwise Comparison for Prompt Optimization

**Files:**
- Modify: `src/evolution/prompt_optimizer.py`
- Modify: `src/agent/prompts.py`
- Create: `tests/unit/test_pairwise.py`

- [ ] **Step 1: Add pairwise prompt to prompts.py**

Add to `src/agent/prompts.py`:
```python
# --- Pairwise Comparison Prompt ---

PAIRWISE_COMPARISON_PROMPT = (
    "You are comparing two research outputs for the same task. Determine which "
    "output is better overall (more complete, accurate, well-sourced, and clear).\n\n"
    "## Task\n{task}\n\n"
    "## Output A\n{output_a}\n\n"
    "## Output B\n{output_b}\n\n"
    'Respond as JSON: {"winner": "A" or "B", '
    '"confidence": "high|medium|low", "reasoning": "brief explanation"}'
)
```

- [ ] **Step 2: Write tests for pairwise functions**

Create `tests/unit/test_pairwise.py`:
```python
"""Tests for pairwise comparison in prompt optimization."""

import json
from unittest.mock import MagicMock

from src.evolution.prompt_optimizer import (
    pairwise_compare,
    _select_tasks_for_mini_scoring,
)
from src.evolution.state import AnalysisResult, GraderResult


def _make_llm_returning(data: dict) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(data))
    return llm


def _make_analysis(task: str, classification: str, score: float) -> AnalysisResult:
    return AnalysisResult(
        run_id="test",
        task=task,
        classification=classification,
        average_score=score,
        grader_results=[GraderResult(name="tc", score=score, passed=score >= 0.75, reasoning="")],
        output=f"output for {task}",
        tool_calls=[],
    )


def test_pairwise_compare_returns_winner():
    llm = _make_llm_returning({"winner": "A", "confidence": "high", "reasoning": "better"})
    result = pairwise_compare(llm, "task", "output old", "output new", "old", "new")
    assert result["winner"] in ("old", "new")
    assert result["confidence"] == "high"


def test_select_tasks_prioritizes_failures():
    analyses = [
        _make_analysis("fail1", "failed", 0.2),
        _make_analysis("fail2", "partial", 0.4),
        _make_analysis("ok1", "successful", 0.9),
    ]
    selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
    tasks = [s["task"] for s in selected]
    assert "fail1" in tasks
    assert "fail2" in tasks


def test_select_tasks_fills_from_successful_if_few_failures():
    analyses = [
        _make_analysis("fail1", "failed", 0.3),
        _make_analysis("ok1", "successful", 0.8),
        _make_analysis("ok2", "successful", 0.9),
    ]
    selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
    assert len(selected) == 2


def test_select_tasks_handles_empty():
    selected = _select_tasks_for_mini_scoring([], max_tasks=2)
    assert selected == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_pairwise.py -v`
Expected: FAIL -- functions don't exist

- [ ] **Step 4: Add pairwise functions to prompt_optimizer.py**

Add to `src/evolution/prompt_optimizer.py`:

```python
import random

from src.agent.prompts import PAIRWISE_COMPARISON_PROMPT


def _select_tasks_for_mini_scoring(
    analyses: list[AnalysisResult],
    max_tasks: int = 2,
) -> list[AnalysisResult]:
    """Select tasks for mini-scoring, prioritizing failures."""
    if not analyses:
        return []

    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    failed.sort(key=lambda a: a["average_score"])

    selected = failed[:max_tasks]

    if len(selected) < max_tasks:
        successful = [a for a in analyses if a["classification"] == "successful"]
        successful.sort(key=lambda a: a["average_score"])
        remaining = max_tasks - len(selected)
        selected.extend(successful[:remaining])

    return selected


def pairwise_compare(
    llm: BaseChatModel,
    task: str,
    output_a: str,
    output_b: str,
    label_a: str,
    label_b: str,
) -> dict[str, str]:
    """Compare two outputs for the same task. Returns winner label + confidence.

    Randomly swaps A/B position to counter position bias.
    """
    if random.random() < 0.5:
        pos_a_output, pos_b_output = output_a[:3000], output_b[:3000]
        pos_a_label, pos_b_label = label_a, label_b
    else:
        pos_a_output, pos_b_output = output_b[:3000], output_a[:3000]
        pos_a_label, pos_b_label = label_b, label_a

    prompt = PAIRWISE_COMPARISON_PROMPT.format(
        task=task,
        output_a=pos_a_output,
        output_b=pos_b_output,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content.strip())
        ab_winner = parsed.get("winner", "A")
        winner = pos_a_label if ab_winner == "A" else pos_b_label
        return {
            "winner": winner,
            "confidence": parsed.get("confidence", "low"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as exc:
        logger.warning("Pairwise comparison failed: %s", exc)
        return {"winner": label_a, "confidence": "low", "reasoning": f"error: {exc}"}


def validate_candidate_prompt(
    llm: BaseChatModel,
    settings: "Settings",
    prompt_store: "PromptStore",
    memory_store: "MemoryStore",
    candidate_prompt: str,
    analyses: list[AnalysisResult],
) -> bool:
    """Check that a candidate prompt actually improves outputs via pairwise comparison.

    Re-runs 2 tasks with the candidate prompt, then compares old vs new outputs.
    Returns True if the candidate wins majority of comparisons.
    """
    from src.evolution.orchestrator import _run_single_task

    selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
    if len(selected) < 2:
        logger.info("Not enough tasks for pairwise validation, accepting candidate")
        return True

    # Temporarily add candidate prompt as latest version
    temp_version = prompt_store.add_version(candidate_prompt, score=None)

    wins_new = 0
    wins_old = 0

    try:
        for analysis in selected:
            task = analysis["task"]
            old_output = analysis["output"]

            result = _run_single_task(settings, prompt_store, memory_store, task)
            new_output = result.get("output", "")

            if not new_output or result.get("status") == "error":
                wins_old += 1
                continue

            comparison = pairwise_compare(
                llm, task, old_output, new_output, "old", "new"
            )
            logger.info(
                "Pairwise: task='%s' winner=%s confidence=%s",
                task[:50], comparison["winner"], comparison["confidence"],
            )

            if comparison["winner"] == "new":
                wins_new += 1
            else:
                wins_old += 1
    finally:
        pass  # prompt_store doesn't have delete; get_current_prompt picks best-scoring

    logger.info("Pairwise validation: new=%d, old=%d", wins_new, wins_old)
    return wins_new > wins_old
```

- [ ] **Step 5: Update optimize_prompt() signature and add pairwise validation**

Update `optimize_prompt` function signature to accept `settings` and `memory_store` (optional for backwards compat):
```python
def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[AnalysisResult],
    skills_dir: "Path | None" = None,
    settings: "Settings | None" = None,
    memory_store: "MemoryStore | None" = None,
) -> int:
```

After `improved_prompt = generate_improved_prompt(...)` and before `prompt_store.add_version(...)`, add:
```python
    # Pairwise validation: only adopt if candidate beats current
    if settings and memory_store:
        is_better = validate_candidate_prompt(
            llm, settings, prompt_store, memory_store, improved_prompt, analyses,
        )
        if not is_better:
            logger.info("Candidate prompt lost pairwise validation, keeping current")
            return prompt_store.get_latest_version_number()
    else:
        logger.info("Pairwise validation skipped (settings/memory_store not provided)")
```

- [ ] **Step 6: Update orchestrator to pass settings + memory_store**

In `src/evolution/orchestrator.py`, `node_optimize_prompt`, update the call:
```python
new_version = optimize_prompt(
    llm, prompt_store, state["analysis_results"],
    skills_dir=settings.skills_path,
    settings=settings,
    memory_store=memory_store,
)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_pairwise.py -v`
Expected: PASS

- [ ] **Step 8: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/agent/prompts.py src/evolution/prompt_optimizer.py src/evolution/orchestrator.py tests/unit/test_pairwise.py
git commit -m "feat: add pairwise comparison for prompt optimization validation"
```

---

### Task 9: Final Integration Checks

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run ruff linter**

Run: `ruff check src/ tests/`
Expected: No errors (or fix any that appear)

- [ ] **Step 3: Run ruff formatter**

Run: `ruff format src/ tests/`

- [ ] **Step 4: Run mypy type check**

Run: `mypy src/ --ignore-missing-imports`
Expected: No errors

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: fix lint and type errors from reliable grading implementation"
```
