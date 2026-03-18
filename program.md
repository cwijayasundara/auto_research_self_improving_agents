# Two-Speed Self-Improving Agent

This is a combined autoresearch + self-evolving agent experiment. The system has two optimization loops:

1. **Inner loop** (automatic): prompt optimization, skill extraction (from successes AND failures), reflective memory — runs via `python -m src evolve`
2. **Outer loop** (you, the coding agent): structural code changes — new tools, model swaps, architecture changes

## Setup

1. **Create a branch**: `git checkout -b autoresearch/<tag>` from main.
2. **Read the key files**:
   - `program.md` — this file (your instructions)
   - `evolution_state/` — inner loop's learned state (failures, hypotheses, skills, plateau reports)
   - `src/agent/deep_agent.py` — the agent you can modify
   - `src/agent/prompts.py` — prompt templates
   - `src/tools/search.py` — tools available to the agent
   - `src/evolution/orchestrator.py` — the inner-loop pipeline
3. **Verify setup**: Check `.env` has the required API keys.
4. **Run baseline**: `python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3 > eval.log 2>&1`
5. **Initialize results.tsv** with the baseline.

## The Two-Speed Loop

The outer loop runs for a **bounded number of iterations** controlled by `MAX_OUTER_LOOP_ITERATIONS`
(default: 5, configurable via env var). Each iteration consists of one inner-loop evolution run
followed by one structural change attempt. The loop also stops early if a score target is reached
or consecutive iterations show no improvement.

```
MAX_OUTER_LOOP_ITERATIONS = 5  (set in .env or src/config/settings.py)
```

LOOP (up to MAX_OUTER_LOOP_ITERATIONS times):

### Phase 1: Inner Loop (automatic self-evolution)

1. Run inner loop: `python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 5 > eval.log 2>&1`
2. The inner loop will automatically:
   - Run the agent on all tasks
   - Grade with 3 evaluators (task completion, efficiency, quality)
   - Reflect and store memories (episodic + semantic)
   - Extract skills from successes (score >= 0.70)
   - Create defensive skills from failures (the "antibody" pattern)
   - Optimize the system prompt using failure analysis + skill awareness
   - Persist state to `evolution_state/` for you to read
   - Detect plateau and stop when improvement < 5% for 2 cycles
3. Parse results: `grep "score=" eval.log | tail -5`

### Phase 2: Outer Loop (your structural changes)

When the inner loop plateaus (check `evolution_state/plateau_report.md`):

1. **Read the evolution state**:
   - `evolution_state/failures.md` — what the inner loop couldn't fix
   - `evolution_state/hypotheses.md` — what's been tried and outcomes
   - `evolution_state/skills_summary.md` — all learned skills (success + defensive)
   - `evolution_state/plateau_report.md` — why it stopped and what to try next

2. **Decide what to change** based on the failure analysis:
   - If failures are **capability gaps** → add new tools to `src/tools/`
   - If failures are **architectural** → restructure `src/agent/deep_agent.py`
   - If failures are **model limitations** → swap model in `.env`
   - If failures are **search quality** → improve search tool or add new search tools
   - If all scores are high → expand `tasks/research_tasks.json` with harder examples

3. **Make ONE structural change**, git commit

4. **Re-run inner loop** (Phase 1) to see if the structural change + inner evolution improves scores

5. **Keep or discard**:
   - If improved: keep the commit, log to `results.tsv`
   - If equal/worse: `git reset --hard` to previous best

### Early stopping

Stop the outer loop early (before reaching MAX_OUTER_LOOP_ITERATIONS) if:
- **Score target reached**: avg_score >= 0.95 for 2 consecutive iterations
- **No progress**: 3 consecutive outer-loop iterations where the best score doesn't improve
- **All experiments exhausted**: you've tried tools, architecture, model, and search changes without improvement

## What you CAN modify

- `src/agent/deep_agent.py` — agent construction, model selection, tool wiring
- `src/agent/prompts.py` — prompt templates (though inner loop also optimizes these)
- `src/agent/subagents.py` — sub-agent definitions
- `src/tools/` — add new tools, improve existing ones
- `src/config/settings.py` — configuration options
- `src/evolution/prompt_optimizer.py` — prompt validation guardrails (e.g. adding new drift patterns)
- `tasks/research_tasks.json` — expand the evaluation dataset
- `.env` — model selection, API keys, tuning parameters

## What you CANNOT modify

- `src/evolution/orchestrator.py` — the inner-loop pipeline is fixed
- `src/evolution/graders/` — grading functions are ground truth
- `src/evolution/analyzer.py` — analysis pipeline is fixed
- `src/evolution/failure_skill_creator.py` — failure skill creation is fixed
- `src/evolution/skill_extractor.py` — skill extraction is fixed

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	overall_score	skills_count	failure_skills	prompt_version	status	description
```

## Decision Protocol

```
iteration = 0
consecutive_no_improvement = 0
consecutive_high_score = 0

WHILE iteration < MAX_OUTER_LOOP_ITERATIONS:
  1. Run inner evolution: python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 5
  2. Read evolution_state/*.md
  3. If plateau was reached:
     - If failures are prompt-related → inner loop handles it (skip)
     - If failures are capability gaps → add new tools
     - If failures are architectural → refactor agent structure
     - If failures are model limitations → swap model
     - If all scores are high → expand dataset with harder examples
  4. Make ONE structural change, commit, re-run inner loop
  5. If improved: keep, log to results.tsv
     If not: git reset --hard, log as discarded
  6. Update counters:
     - If score improved: consecutive_no_improvement = 0
     - Else: consecutive_no_improvement += 1
     - If avg_score >= 0.95: consecutive_high_score += 1
     - Else: consecutive_high_score = 0
  7. STOP EARLY if:
     - consecutive_no_improvement >= 3
     - consecutive_high_score >= 2
  8. iteration += 1

DONE — print final results summary from results.tsv
```

## Autonomous Operation

Once the experiment loop has begun, do NOT pause to ask the human if you should continue.
Work autonomously until one of the stopping conditions is met (max iterations, early stopping,
or manual interruption). The loop is bounded — it WILL terminate.

## Simplicity criterion

All else equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

## Known risks: prompt drift

The inner loop's prompt optimizer can drift into generating prompts that break
autonomous operation — e.g. prompts that tell the agent to ask the user for
input, present menu options, or request clarification. This has happened before
and caused the agent to stop producing reports.

**Guardrails in place** (in `src/evolution/prompt_optimizer.py`):
- `validate_prompt_autonomy()` checks every generated prompt against a set of
  regex patterns that detect user-interaction language
- If violations are found, the optimizer retries (up to 2 times), then falls
  back to the current prompt rather than accepting a drifted prompt
- The `METAPROMPT_TEMPLATE` explicitly instructs the LLM to never add user
  interaction to the generated prompt

**If the agent starts asking the user for input again**:
1. Check the latest prompt version in `prompts/` — look for interaction patterns
2. The fix is in the prompt, not the code. Reset to a known-good prompt version
3. If new drift patterns appear, add them to `_AUTONOMY_VIOLATION_PATTERNS` in
   `src/evolution/prompt_optimizer.py`
