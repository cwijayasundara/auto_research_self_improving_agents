"""Quick test script for the new grading components.

Run: python3 test_graders.py
"""

import json
import logging

from src.config.settings import Settings, configure_logging, export_langsmith_env

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

settings = Settings()
export_langsmith_env(settings)

from src.agent.deep_agent import create_llm

llm = create_llm(settings)

# --- Sample data ---
TASK = "What are the latest advances in quantum computing in 2024-2025?"
OUTPUT = """# Quantum Computing Advances 2024-2025

## Executive Summary
Quantum computing made significant strides in 2024-2025, particularly in error correction.
Google's Willow chip demonstrated that increasing qubit count can reduce errors, a key milestone.

## Key Findings
1. Google's Willow processor achieved below-threshold quantum error correction with 105 qubits
2. IBM released its 1,121-qubit Condor processor in late 2023, followed by Heron in 2024
3. Microsoft announced a topological qubit breakthrough in February 2025
4. Error correction moved from theory to repeated experimental demonstration
5. Surface codes remained the dominant architecture for fault-tolerant designs

## Detailed Analysis
The most significant advance was Google's demonstration that logical error rates decrease
as you add more physical qubits — the opposite of what happens without error correction.
This was published in Nature in December 2024.

IBM's roadmap shifted toward "utility-scale" quantum computing, focusing on circuits that
can produce useful results despite noise, rather than purely error-corrected computation.

Several startups including IonQ, Quantinuum, and PsiQuantum made progress on alternative
hardware: trapped ions, photonic systems, and neutral atoms respectively.

## Sources
- Google AI Blog: Willow chip announcement (December 2024)
- Nature paper on below-threshold QEC (2024)
- IBM Quantum roadmap 2024
- Based on available knowledge for startup developments
"""

print("=" * 70)
print("TESTING GRADING COMPONENTS")
print("=" * 70)

# ---- 1. Multi-Judge Task Completion ----
print("\n--- 1. Multi-Judge Task Completion ---")
from evoagent.graders.multi_judge import MultiJudgeGrader
from src.agent.prompts import TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT, TASK_COMPLETION_PROMPT

tc_grader = MultiJudgeGrader(llm, name="task_completion", judge_prompts=[TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT], fallback_prompt=TASK_COMPLETION_PROMPT)
tc_result = tc_grader.grade(TASK, OUTPUT)
print(f"  Score: {tc_result.score}")
print(f"  Passed: {tc_result.passed}")
print(f"  Reasoning: {tc_result.reasoning[:200]}")

# ---- 2. Multi-Judge Quality ----
print("\n--- 2. Multi-Judge Quality ---")
from src.agent.prompts import Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT, QUALITY_PROMPT

q_grader = MultiJudgeGrader(llm, name="quality", judge_prompts=[Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT], fallback_prompt=QUALITY_PROMPT)
q_result = q_grader.grade(TASK, OUTPUT)
print(f"  Score: {q_result.score}")
print(f"  Passed: {q_result.passed}")
print(f"  Reasoning: {q_result.reasoning[:200]}")

# ---- 3. Efficiency (rule-based) ----
print("\n--- 3. Efficiency ---")
from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.core.types import TrajectoryMetrics

metrics = TrajectoryMetrics(
    total_tokens=5000, total_steps=8, latency_seconds=25.0, tool_call_count=3
)
eff_result = EfficiencyGrader().grade("test", OUTPUT, metrics=metrics)
print(f"  Score: {eff_result.score}")
print(f"  Passed: {eff_result.passed}")
print(f"  Reasoning: {eff_result.reasoning}")

# ---- 4. Claim Verification ----
print("\n--- 4. Claim Verification (internal consistency) ---")
from src.evolution.graders.claim_verification import _extract_claims, _verify_claims, grade_claims

claims = _extract_claims(llm, OUTPUT)
print(f"  Extracted {len(claims)} claims:")
for c in claims:
    print(f"    - {c[:80]}")

verdicts = _verify_claims(llm, claims, OUTPUT)
print(f"\n  Verification verdicts:")
for v in verdicts:
    print(f"    [{v.get('verdict', '?')}] {v.get('claim', '?')[:60]}")

cv_result = grade_claims(llm, TASK, OUTPUT)
print(f"\n  Final score: {cv_result.score}")
print(f"  Passed: {cv_result.passed}")

# ---- 5. Fact Checker (spot-check via search) ----
print("\n--- 5. Fact Checker (spot-check via search) ---")
from src.evolution.graders.fact_checker import _select_verifiable_claims, spot_check_claims

try:
    from src.tools.search import create_search_tool

    search_tool = create_search_tool(settings)
    print("  Search tool created successfully")

    if claims:
        selected = _select_verifiable_claims(llm, claims)
        print(f"  Selected {len(selected)} verifiable claims:")
        for c in selected:
            print(f"    - {c[:80]}")

        spot_score = spot_check_claims(llm, search_tool, claims)
        print(f"\n  Spot-check score: {spot_score}")

        # Now test blended claim verification
        cv_blended = grade_claims(llm, TASK, OUTPUT, spot_check_score=spot_score)
        print(f"  Blended claim verification score: {cv_blended.score}")
        print(f"  Reasoning: {cv_blended.reasoning[:200]}")
except Exception as e:
    print(f"  Search tool failed: {e}")
    print("  (This is OK — spot-check is optional, claim verification still works without it)")

# ---- 6. Full Classification ----
print("\n--- 6. Full Classification ---")
from src.evolution.analyzer import classify_trajectory

all_results = [tc_result, eff_result, q_result, cv_result]
classification, avg_score = classify_trajectory(all_results)
print(f"  Classification: {classification}")
print(f"  Average score: {avg_score}")
print(f"  Individual scores: {[f'{r.name}={r.score}' for r in all_results]}")

print("\n" + "=" * 70)
print("ALL GRADING COMPONENTS TESTED")
print("=" * 70)
