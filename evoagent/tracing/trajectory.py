"""Trajectory data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation within a trajectory."""

    name: str
    args: dict = Field(default_factory=dict)
    output: str = ""


class TrajectoryRecord(BaseModel):
    """A complete agent run trajectory."""

    run_id: str
    task: str = ""
    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_tokens: int = 0
    total_steps: int = 0
    latency_seconds: float = 0.0
    tool_call_count: int = 0
    status: str = "completed"

    @property
    def is_successful(self) -> bool:
        return self.status == "completed" and bool(self.output)
