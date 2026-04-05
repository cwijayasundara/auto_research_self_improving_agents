"""Pluggable multi-axis grading system."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.graders requires langchain-core. "
        "Install with: pip install evoagent[graders]"
    ) from None

from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.graders.multi_judge import MultiJudgeGrader

__all__ = ["EfficiencyGrader", "MultiJudgeGrader"]
