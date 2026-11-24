from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class PlanStep:
    id: str
    description: str
    requires_approval: bool = False


@dataclass(frozen=True, slots=True)
class GoalPlan:
    goal: str
    steps: tuple[PlanStep, ...]
    note: str = "Deterministic starter plan; richer reasoning providers are a future milestone."
    id: str = field(default_factory=lambda: str(uuid4()))


class Planner:
    """A safe baseline planner; it proposes steps and never executes external actions."""

    def plan(self, goal: str) -> GoalPlan:
        if not goal.strip():
            raise ValueError("A goal is required")
        steps = (
            PlanStep(str(uuid4()), "Clarify the desired outcome and constraints."),
            PlanStep(str(uuid4()), "Gather relevant, permission-scoped context."),
            PlanStep(str(uuid4()), "Prepare a proposed approach and identify risks."),
            PlanStep(str(uuid4()), "Validate the outcome with the owner before consequential action.", True),
        )
        return GoalPlan(goal=goal, steps=steps)
