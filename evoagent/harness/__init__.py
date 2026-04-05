"""Harness middleware stack for LangChain agents."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.harness requires langchain-core. "
        "Install with: pip install evoagent[harness]"
    ) from None

from evoagent.harness.builder import default_middleware_stack
from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TraceCaptureMiddleware,
)

__all__ = [
    "ContextAssemblyMiddleware",
    "LoopDetectionMiddleware",
    "SelfVerificationMiddleware",
    "TraceCaptureMiddleware",
    "default_middleware_stack",
]
