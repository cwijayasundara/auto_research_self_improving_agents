"""Evolution loop: propose -> run -> grade -> accept/reject."""

try:
    from langgraph.graph import StateGraph as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.evolution requires langgraph. "
        "Install with: pip install evoagent[evolution]"
    ) from None

from evoagent.evolution.analyzer import analyze_trajectory, classify_trajectory
from evoagent.evolution.sleep_review import run_sleep_review

__all__ = ["analyze_trajectory", "classify_trajectory", "run_sleep_review"]
