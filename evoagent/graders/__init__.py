"""Pluggable multi-axis grading system."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.graders requires langchain-core. "
        "Install with: pip install evoagent[graders]"
    ) from None
