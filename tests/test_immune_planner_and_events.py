import pytest

from artificial_human.brain.planner import Planner
from artificial_human.core.events import EventBus
from artificial_human.immune.engine import ImmuneSystem
from artificial_human.metabolism.governor import BudgetExceeded, BudgetLimits, ResourceGovernor
from artificial_human.organism import ArtificialOrganism


def test_event_bus_delivers_in_registration_order() -> None:
    bus = EventBus()
    received: list[int] = []
    bus.subscribe("pulse", lambda value: received.append(value))
    bus.subscribe("pulse", lambda value: received.append(value * 2))
    assert bus.publish("pulse", 3) == 2
    assert received == [3, 6]


def test_resource_governor_enforces_budget() -> None:
    governor = ResourceGovernor(BudgetLimits(operations=1, tokens=10, wall_seconds=10))
    governor.consume(operations=1)
    with pytest.raises(BudgetExceeded):
        governor.consume(operations=1)


def test_immune_system_detects_anomaly_and_ignores_normal_signal() -> None:
    immune = ImmuneSystem()
    signal, incident = immune.evaluate("api_error_rate", 0.01)
    assert signal is not None and not signal.abnormal
    assert incident is None
    signal, incident = immune.evaluate("api_error_rate", 0.20)
    assert signal is not None and signal.abnormal
    assert incident is not None and incident.severity.value == "warning"


def test_repair_is_a_reviewable_proposal_not_execution() -> None:
    organism = ArtificialOrganism(database_path=":memory:")
    result = organism.evaluate_health("heartbeat_age_seconds", 120)
    assert result["incident"] is not None
    assert result["repair_proposal"]["status"] == "pending_review"
    proposal = organism.regeneration.list()[0]
    assert "No production mutation." in proposal.safety_checks


def test_planner_returns_steps_without_running_them() -> None:
    plan = Planner().plan("Learn my writing preferences")
    assert len(plan.steps) == 4
    assert plan.steps[-1].requires_approval is True
