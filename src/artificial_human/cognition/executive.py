from __future__ import annotations

from dataclasses import dataclass

from artificial_human.brain.planner import GoalPlan, Planner


@dataclass(frozen=True, slots=True)
class ExecutiveState:
    goal: str
    plan: GoalPlan
    approval_required: bool


class ExecutiveControlSystem:
    """Coordinates goal decomposition and retains explicit approval gates."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def prepare(self, goal: str) -> ExecutiveState:
        plan = self.planner.plan(goal)
        return ExecutiveState(
            goal=goal,
            plan=plan,
            approval_required=any(step.requires_approval for step in plan.steps),
        )
