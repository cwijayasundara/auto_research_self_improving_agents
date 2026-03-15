"""Pydantic trajectory data models."""

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation within a trajectory."""

    name: str
    args: dict = Field(default_factory=dict)
    output: str = ""


class TrajectoryMetrics(BaseModel):
    """Efficiency metrics for a trajectory."""

    total_tokens: int = 0
    total_steps: int = 0
    latency_seconds: float = 0.0
    tool_call_count: int = 0


class Trajectory(BaseModel):
    """A complete agent run trajectory parsed from LangSmith traces."""

    run_id: str
    task: str = ""
    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metrics: TrajectoryMetrics = Field(default_factory=TrajectoryMetrics)
    messages: list[dict] = Field(default_factory=list)
    status: str = "completed"
    feedback: dict = Field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.status == "completed" and bool(self.output)
