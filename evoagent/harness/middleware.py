"""Harness middleware re-export for evoagent.

Provides TraceCaptureMiddleware from the src.agent.middleware implementation.
"""

from src.agent.middleware import TraceCaptureMiddleware

__all__ = ["TraceCaptureMiddleware"]
