"""Evolution loop: propose -> run -> grade -> accept/reject."""

try:
    from langgraph.graph import StateGraph as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.evolution requires langgraph. "
        "Install with: pip install evoagent[evolution]"
    ) from None
