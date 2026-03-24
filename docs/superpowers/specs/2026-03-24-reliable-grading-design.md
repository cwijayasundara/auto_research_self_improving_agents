# Reliable Grading, Prompt Optimization & Skill Learning

**Date:** 2026-03-24
**Status:** Approved
**Approach:** In-place enhancement (Approach A)

## Problem

The evolution engine's grading, prompt optimization, and skill learning have several reliability weaknesses:

- **No ground truth** — LLM graders judge output quality without reference answers
- **Single-judge bias** — one LLM call per grader, no calibration, no inter-rater reliability
- **No A/B testing** — new prompts replace old ones with no evidence they're better
- **No claim verification** — no check that factual claims are consistent or accurate
- **Vague rubrics** — grading prompts don't define what score levels mean

## Constraints

- Must remain fully self-improving (no human-provided ground truth per task)
- Increased LLM cost is acceptable
- Latency increase acceptable (evolution loop only, not normal agent use)
- Changes only affect the evolution pipeline (`src/evolution/`, `src/agent/prompts.py`)

## Terminology

- **Grader**: A top-level scoring component (task_completion, efficiency, quality, claim_verification). Produces one `GraderResult`.
- **Judge**: A perspective-specific LLM call within a multi-judge grader. Multiple judges produce sub-scores that are aggregated into one grader result.

## Design

### 1. Rubric Anchoring (Few-Shot Score Calibration)

**Goal:** Eliminate score ambiguity by defining what each score level looks like.

**Change:** Add 3 anchor examples (high/medium/low) to each LLM grading prompt in `src/agent/prompts.py`. These anchors are research-task-specific (matching this system's domain).

**Task completion anchors:**
- Score 0.9: Directly answers the question with 3+ cited sources, structured sections, no factual errors, thorough coverage of all subtopics asked about.
- Score 0.5: Partially addresses the question but misses key aspects, 1-2 sources, some structure but gaps in analysis, may have minor inaccuracies.
- Score 0.2: Off-topic or superficial, no sources, unstructured, factual errors or hallucinated claims, fails to answer the core question.

**Quality anchors:**
- Score 0.9: Clear logical structure with sections, accurate facts with citations, deep analysis that covers nuances and trade-offs, stays tightly focused on the task.
- Score 0.5: Readable but loosely organized, mostly accurate but some unsourced claims, surface-level analysis, some tangential content.
- Score 0.2: Disorganized or incoherent, factual errors, shallow or repetitive, significant off-topic content or filler.

**Files:** `src/agent/prompts.py`
**New LLM calls:** 0

### 2. Multi-Judge with Different Rubrics

**Goal:** Replace single-judge scoring with 3 perspective judges + median aggregation.

**Task completion judges:**
- **Completeness judge**: Did the output cover all aspects of the question? Score 0-1.
- **Evidence judge**: Are claims backed by cited sources? Score 0-1.
- **Accuracy judge**: Are the facts and reasoning correct? Score 0-1.

**Quality judges:**
- **Structure judge**: Is the output well-organized and readable? Score 0-1.
- **Depth judge**: Is the analysis thorough and insightful? Score 0-1.
- **Relevance judge**: Does the output stay focused, no filler? Score 0-1.

Each judge prompt includes the rubric anchors from Section 1 plus a focused lens. Example for the evidence judge: "Focus primarily on whether claims are supported by cited sources. A well-sourced output with minor structural issues should still score high."

**Aggregation:** `statistics.median()` of 3 judge scores (resistant to one outlier). If max - min > 0.3, append "low_agreement" flag in reasoning string. `passed` threshold remains 0.75 for task_completion and quality.

**Parallelization strategy:**
- The 3 judges within each multi-judge grader run in parallel via `ThreadPoolExecutor(max_workers=3)` **inside the grader function itself** (not at the LangGraph level).
- The LangGraph analyzer graph is restructured: `task_completion`, `efficiency`, and `quality` nodes all receive edges from START and feed into `classify`. This gives us LangGraph-level parallelism between graders + thread-level parallelism within multi-judge graders.
- Claim verification (Section 3) also runs in parallel with the other graders from START.

**Updated LangGraph layout:**
```
START ──┬── grade_task_completion ──┐
        ├── grade_efficiency ───────┤
        ├── grade_quality ──────────┤
        └── grade_claims ───────────┘
                                    └── classify ── END
```

All four grader nodes run concurrently. Within task_completion and quality nodes, 3 judges run concurrently via ThreadPoolExecutor.

**Error handling for parallel judges:** If a judge call fails (LLM timeout, JSON parse error), log the error and exclude that judge from the median. If 2+ judges fail within a grader, fall back to a single retry with the original (non-perspective) prompt. If that also fails, return `GraderResult(score=0.5, passed=False, reasoning="grader_error")`.

**Efficiency grader:** Unchanged (rule-based, no LLM, no multi-judge).

**Files:**
- New: `src/evolution/graders/multi_judge.py`
- Modified: `src/evolution/analyzer.py` (parallel graph layout, swap grader calls)
- Modified: `src/agent/prompts.py` (6 new perspective prompts with anchors)

**New LLM calls per task:** +4 (was 1 per LLM grader, now 3 each)

### 3. Claim-Level Verification (New 4th Grader)

**Goal:** Check internal consistency and source alignment of factual claims.

**Flow:**
1. **Extract claims** — LLM call extracts up to 10 key factual claims as a JSON list. Prompt instructs: "Only extract claims that appear verbatim or are clearly stated in the output. Do not infer or fabricate claims." If the output contains fewer than 3 factual claims, extract what exists.
2. **Verify claims** — Separate LLM call receives the extracted claims AND the full output text. Checks each claim for:
   - Does it have a cited source in the output?
   - Is it contradicted by other claims in the output?
   - Is it suspiciously specific without a source (likely hallucinated)?
   - Per-claim verdict: `supported`, `unsupported`, `contradicted`
3. **Internal consistency score** — `supported_count / total_claims`. Falls back to 0.5 if no claims extracted (output too short or non-factual).
4. **Spot-check enrichment** (Section 5) — If available, blends with factual spot-check score:
   ```
   final_score = 0.6 * internal_consistency_score + 0.4 * spot_check_score
   ```
   If spot-check fails or is unavailable: `final_score = internal_consistency_score` (weight shifts entirely to internal consistency, no penalty).

**Why 2 LLM calls?** Separating extraction from verification prevents the LLM from cherry-picking only claims it can verify.

**Pass threshold:** >= 0.6

**JSON parsing:** Reuse the existing `_parse_grader_response()` pattern from `task_completion.py`. On parse failure, return `GraderResult(score=0.5, passed=False, reasoning="parse_error")`.

**Updated AnalyzerState:**
```python
class AnalyzerState(TypedDict):
    trajectory: Trajectory
    task_completion: GraderResult
    efficiency: GraderResult
    quality: GraderResult
    claim_verification: GraderResult  # NEW
    classification: str
    average_score: float
```

`AnalysisResult.grader_results` remains `list[GraderResult]` — now contains 4 items instead of 3. No schema change needed.

**Updated classification logic (4 graders):**
```python
SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5
MIN_PASS_COUNT = 2      # unchanged — 2 of 4 must pass
MIN_AVERAGE_SCORE = 0.6  # unchanged

# Logic is unchanged, just more voters:
# "successful" = pass_count >= 2 AND avg_score >= 0.75
# "partial"    = (pass_count >= 2 AND avg_score >= 0.6) OR avg_score >= 0.5
# "failed"     = everything else
```

Rationale for keeping MIN_PASS_COUNT=2: With 4 graders having different pass thresholds (task_completion=0.75, quality=0.75, efficiency=0.5, claim_verification=0.6), requiring 3 passes would be too strict. The average_score threshold (0.6/0.75) already guards against low overall quality.

**Files:**
- New: `src/evolution/graders/claim_verification.py`
- Modified: `src/evolution/analyzer.py` (add 4th grader node in parallel layout)
- Modified: `src/evolution/state.py` (add `claim_verification` to `AnalyzerState`)
- Modified: `src/agent/prompts.py` (`CLAIM_EXTRACTION_PROMPT`, `CLAIM_VERIFICATION_PROMPT`)

**New LLM calls per task:** +2

### 4. Pairwise Comparison for Prompt Optimization

**Goal:** Only adopt a new prompt if it demonstrably produces better outputs than the current one.

**Current flow:**
```
analyze failures -> generate improved prompt -> save it -> move on
```

**New flow:**
```
analyze failures -> generate candidate prompt -> re-run 2-3 tasks with candidate
-> pairwise compare old vs new outputs -> keep only if it wins
```

**Steps:**
1. **Generate candidate prompt** — Same metaprompt approach as today.
2. **Mini evaluation** — Select tasks for re-run:
   - Take failed/partial tasks sorted by ascending score (worst first)
   - Pick up to 2 tasks. If fewer than 2 failed/partial, fill from successful tasks (lowest scores first)
   - Minimum 2 tasks for a meaningful comparison
   - Re-run agent with candidate prompt on selected tasks. Uses same timeout and search tools as normal runs.
3. **Pairwise comparison** — For each task, send both outputs to LLM:
   - Randomly assign old/new to positions A/B (coin flip per comparison)
   - Prompt asks: "Which output is better? Respond with winner (A or B), confidence (high/medium/low), reasoning."
   - Single comparison per task (no repeated trials — position bias is handled by random assignment across tasks)
4. **Decision:**
   - New wins majority of comparisons -> adopt candidate prompt
   - Tie -> keep current prompt (conservative, prevent regression)
   - Old wins majority -> keep current prompt
   - Log all comparison results (task, winner, confidence, reasoning) for debugging

**Files:**
- Modified: `src/evolution/prompt_optimizer.py`
  - New: `pairwise_evaluate(llm, task, output_a, output_b, label_a, label_b) -> dict`
  - New: `validate_prompt_improvement(llm, settings, prompt_store, memory_store, candidate_prompt, analyses) -> bool`
  - Modified: `optimize_prompt()` calls `validate_prompt_improvement` before saving
- Modified: `src/agent/prompts.py` (`PAIRWISE_COMPARISON_PROMPT`)

**New calls per cycle:** 2 agent runs + 2 LLM comparisons

### 5. Factual Spot-Check via Search

**Goal:** Verify a sample of factual claims against external reality using web search.

**Flow:** Builds on Section 3's claim extraction. This is NOT a separate grader — it enriches the claim_verification grader's score.

1. **Select verifiable claims** — LLM picks 2-3 most objectively verifiable claims from the extracted claims (prefer claims with numbers, dates, or named entities over subjective statements).
2. **Search each claim** — Use existing `src/tools/search.py` to query each claim as a search query. Collect the top 2-3 search results per claim. Respect existing rate-limit handling in search.py (`_is_quota_error` + DuckDuckGo fallback).
3. **Compare** — One LLM call per claim with the claim text + search snippets. Returns: `corroborated`, `contradicted`, or `inconclusive`.
4. **Score** — corroborated=1.0, inconclusive=0.5, contradicted=0.0. `spot_check_score = mean(per_claim_scores)`.

**Failure handling:**
- If ALL searches fail (rate limits, network): spot-check is skipped entirely. `claim_verification.score = internal_consistency_score` (no blending, no penalty).
- If SOME searches fail: score only the claims that got search results. If fewer than 1 claim got results, skip spot-check entirely.
- Search failures are logged but never cause the grading pipeline to error.

**Rate limiting:** The fact checker processes claims sequentially (not parallel) to avoid hitting search API rate limits. With 2-3 claims per task and 7-10 tasks per batch, this is ~20 searches per cycle — within typical Tavily free-tier limits.

**Files:**
- New: `src/evolution/graders/fact_checker.py`
- Modified: `src/evolution/graders/claim_verification.py` (import + blend spot-check score)
- Modified: `src/agent/prompts.py` (`CLAIM_SELECTION_PROMPT`, `FACT_CHECK_PROMPT`)

**New calls per task:** 1 LLM (selection) + 2-3 searches + 2-3 LLM (comparison) = ~7 calls

## Prompt Specifications

All new prompts to be added to `src/agent/prompts.py`:

### Perspective Judge Prompts (Section 2)

Each perspective prompt follows this structure:
```
You are evaluating a research agent's output from the perspective of [PERSPECTIVE].

## Task
{task}

## Agent Output
{output}

## Scoring Guide
- 0.9: [high anchor for this perspective]
- 0.5: [medium anchor]
- 0.2: [low anchor]

Focus primarily on [PERSPECTIVE FOCUS]. Other quality dimensions are handled
by other judges — your job is only [PERSPECTIVE].

Respond as JSON: {"score": float 0.0-1.0, "reasoning": "brief explanation"}
```

**6 perspective prompts:**
1. `TC_COMPLETENESS_PROMPT` — focus: coverage of all aspects of the question
2. `TC_EVIDENCE_PROMPT` — focus: whether claims are backed by cited sources
3. `TC_ACCURACY_PROMPT` — focus: factual correctness and reasoning quality
4. `Q_STRUCTURE_PROMPT` — focus: organization, readability, formatting
5. `Q_DEPTH_PROMPT` — focus: thoroughness of analysis, nuance, trade-offs
6. `Q_RELEVANCE_PROMPT` — focus: staying on-topic, no filler or tangents

### Claim Verification Prompts (Section 3)

**CLAIM_EXTRACTION_PROMPT:**
```
Extract the key factual claims from this research output. Only extract claims
that appear verbatim or are clearly stated in the text. Do not infer or
fabricate claims. Extract up to 10 claims.

## Agent Output
{output}

Respond as JSON: {"claims": ["claim 1", "claim 2", ...]}
```

**CLAIM_VERIFICATION_PROMPT:**
```
Verify each claim against the source output. For each claim, determine:
- Is it supported by a cited source in the output?
- Is it contradicted by other claims in the output?
- Is it suspiciously specific without any source?

## Claims
{claims}

## Full Output
{output}

Respond as JSON: {"verdicts": [{"claim": "...", "verdict": "supported|unsupported|contradicted", "reasoning": "..."}]}
```

### Pairwise Comparison Prompt (Section 4)

**PAIRWISE_COMPARISON_PROMPT:**
```
You are comparing two research outputs for the same task. Determine which
output is better overall (more complete, accurate, well-sourced, and clear).

## Task
{task}

## Output A
{output_a}

## Output B
{output_b}

Respond as JSON: {"winner": "A" or "B", "confidence": "high|medium|low", "reasoning": "brief explanation"}
```

### Fact-Check Prompts (Section 5)

**CLAIM_SELECTION_PROMPT:**
```
From these claims, select the 2-3 most objectively verifiable ones.
Prefer claims with specific numbers, dates, percentages, or named entities.
Avoid subjective or opinion-based claims.

## Claims
{claims}

Respond as JSON: {"selected": ["claim 1", "claim 2"]}
```

**FACT_CHECK_PROMPT:**
```
Does the search evidence support, contradict, or not address this claim?

## Claim
{claim}

## Search Results
{search_results}

Respond as JSON: {"verdict": "corroborated|contradicted|inconclusive", "reasoning": "brief explanation"}
```

## File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/agent/prompts.py` | Modified | Anchored rubrics, 6 perspective prompts, claim/pairwise/fact-check prompts |
| `src/evolution/graders/multi_judge.py` | New | Multi-judge wrapper with ThreadPoolExecutor parallel execution |
| `src/evolution/graders/claim_verification.py` | New | Claim extraction + verification + spot-check blending |
| `src/evolution/graders/fact_checker.py` | New | Factual spot-check via web search |
| `src/evolution/analyzer.py` | Modified | Parallel LangGraph layout, swap to multi-judge, add 4th grader |
| `src/evolution/state.py` | Modified | Add `claim_verification: GraderResult` to `AnalyzerState` |
| `src/evolution/prompt_optimizer.py` | Modified | Add pairwise validation before saving new prompts |
| `src/evolution/orchestrator.py` | Modified | Pass settings to analyzer for search tool access |

## Implementation Order

1. **Rubric anchoring** (standalone, no deps)
2. **Multi-judge** (depends on #1 for anchored prompts)
3. **Claim verification** (standalone new grader)
4. **Factual spot-check** (depends on #3 for claim extraction)
5. **Pairwise comparison** (depends on working grading pipeline for mini-eval)

Each step should be tested independently before moving to the next. Steps 1-4 modify the grading pipeline; step 5 modifies the prompt optimizer. The grading pipeline changes should be verified end-to-end before adding pairwise comparison.
