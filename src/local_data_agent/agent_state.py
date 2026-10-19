"""State is inspectable observations and questions, not private model reasoning."""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    goal: str
    question_params: dict = field(default_factory=dict)
    metric_binding: str | None = None
    context_refs: list = field(default_factory=list)
    plan_version: int = 0
    observed_facts: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    artifact_refs: list = field(default_factory=list)
    validation_status: str = "NOT_CHECKED"
    budget: dict = field(default_factory=lambda: {"requests": 16, "repairs_per_step": 2})
    stop_reason: str | None = None

