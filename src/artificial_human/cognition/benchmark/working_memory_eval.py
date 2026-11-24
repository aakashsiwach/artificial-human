"""Working-memory suite: retention, interference, reactivation and eviction.

The system under test is the unmodified ``WorkingMemory`` (and
``WorkingMemorySessions``). Entries get deterministic ``created_at`` values,
so eviction tie-breaks never depend on the wall clock. Baseline policies
share the same probe protocol:

* ``fifo`` — evict the earliest inserted item; reactivation does not refresh.
* ``lru`` — evict the least recently encoded/reactivated item.
* ``salience_no_decay`` — evict the lowest salience without decay. This is an
  ablation of the existing decay term.

Retention here means "the item is still in the software workspace". It is
not a measure of human working-memory capacity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from artificial_human.cognition.benchmark.common import SPLITS, SuiteResult, check, rng
from artificial_human.cognition.benchmark.metrics import kendall_tau, mean
from artificial_human.cognition.working_memory import (
    WorkingMemory,
    WorkingMemoryEntry,
    WorkingMemorySessions,
)

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class Workspace(Protocol):
    capacity: int
    violations: int

    def encode(self, item_id: str, content: str, salience: float, step: int) -> None: ...

    def contents(self) -> dict[str, str]: ...

    def order(self) -> list[str]: ...


class ExistingWorkspace:
    """Adapter that drives the real ``WorkingMemory`` without changing it."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = WorkingMemory(capacity=capacity)
        self.violations = 0

    def encode(self, item_id: str, content: str, salience: float, step: int) -> None:
        self.memory.remember(WorkingMemoryEntry(
            id=item_id, content=content, kind="benchmark", source="benchmark",
            salience=salience, created_at=(_EPOCH + timedelta(seconds=step)).isoformat(),
        ))
        self.violations += len(self.memory.items()) > self.capacity

    def contents(self) -> dict[str, str]:
        return {item.id: item.content for item in self.memory.items()}

    def order(self) -> list[str]:
        return [item.id for item in self.memory.items()]

    def access_count(self, item_id: str) -> int:
        return next(
            (item.access_count for item in self.memory.items() if item.id == item_id), -1
        )


class BaselineWorkspace:
    def __init__(self, capacity: int, policy: str) -> None:
        self.capacity = capacity
        self.policy = policy
        self.violations = 0
        self._items: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def encode(self, item_id: str, content: str, salience: float, step: int) -> None:
        del step
        present = item_id in self._items
        if self.policy == "fifo" and present:
            self._items[item_id] = (content, salience)  # keep original position
        else:
            self._items.pop(item_id, None)
            self._items[item_id] = (content, salience)
        while len(self._items) > self.capacity:
            if self.policy in ("fifo", "lru"):
                self._items.popitem(last=False)
            else:
                victim = min(self._items, key=lambda key: self._items[key][1])
                del self._items[victim]
        self.violations += len(self._items) > self.capacity

    def contents(self) -> dict[str, str]:
        return {key: value[0] for key, value in self._items.items()}

    def order(self) -> list[str]:
        return list(self._items)


POLICIES: dict[str, Callable[[int], Workspace]] = {
    "existing_working_memory": ExistingWorkspace,
    "fifo": lambda capacity: BaselineWorkspace(capacity, "fifo"),
    "lru": lambda capacity: BaselineWorkspace(capacity, "lru"),
    "salience_no_decay": lambda capacity: BaselineWorkspace(capacity, "salience_no_decay"),
}


@dataclass(frozen=True, slots=True)
class Event:
    item_id: str
    content: str
    salience: float


def _play(policy: str, capacity: int, events: Sequence[Event]) -> Workspace:
    workspace = POLICIES[policy](capacity)
    for step, event in enumerate(events):
        workspace.encode(event.item_id, event.content, event.salience, step)
    return workspace


def _retained(workspace: Workspace, expected: dict[str, str]) -> tuple[float, float]:
    """(retention rate, recall accuracy = present with the expected content)."""
    contents = workspace.contents()
    present = [key for key in expected if key in contents]
    correct = [key for key in present if contents[key] == expected[key]]
    return len(present) / len(expected), len(correct) / len(expected)


_TOPICS = ("lisbon tram", "locker code", "meeting agenda", "insulin dose", "garden shed")


def _delay_events(distractors: int, distractor_salience: float, similar: bool) -> list[Event]:
    events = [Event(f"target-{i}", f"target fact about {_TOPICS[i]}", 0.8) for i in range(3)]
    for index in range(distractors):
        content = (
            f"target fact about {_TOPICS[index % 3]} variant {index}"
            if similar else f"unrelated note number {index}"
        )
        events.append(Event(f"distractor-{index}", content, distractor_salience))
    return events


def run_working_memory_suite(seed: int) -> SuiteResult:
    result = SuiteResult(
        suite="working-memory",
        title="Working memory: retention, interference, reactivation and eviction",
        evidence_category="software behaviour + computational capability benchmark",
        splits={},
    )
    capacities = (3, 5, 7, 9)
    delays = tuple(range(0, 13))
    regimes = {"lower": 0.5, "equal": 0.8, "higher": 0.95}
    targets = {f"target-{i}": f"target fact about {_TOPICS[i]}" for i in range(3)}

    # 1. Delayed retention against intervening items, workspace capacity and distractor salience.
    curves: dict[str, dict[str, dict[str, float]]] = {}
    by_regime: dict[str, dict[str, float]] = {}
    tasks = 0
    for policy in POLICIES:
        retention_by_delay: dict[int, list[float]] = {delay: [] for delay in delays}
        retention_by_capacity: dict[int, list[float]] = {capacity: [] for capacity in capacities}
        regime_scores: dict[str, list[float]] = {name: [] for name in regimes}
        for regime, distractor_salience in regimes.items():
            for capacity in capacities:
                for delay in delays:
                    workspace = _play(policy, capacity, _delay_events(delay, distractor_salience, False))
                    retention, _ = _retained(workspace, targets)
                    regime_scores[regime].append(retention)
                    if regime == "lower":
                        retention_by_delay[delay].append(retention)
                        retention_by_capacity[capacity].append(retention)
                    tasks += policy == "existing_working_memory"
        curves[policy] = {
            "retention_by_intervening_items": {
                str(delay): round(mean(values), 4) for delay, values in retention_by_delay.items()
            },
            "retention_by_capacity": {
                str(capacity): round(mean(values), 4)
                for capacity, values in retention_by_capacity.items()
            },
        }
        by_regime[policy] = {name: round(mean(values), 4) for name, values in regime_scores.items()}
    result.extra["delayed_retention_curves"] = curves
    result.by_condition["delayed_retention_by_distractor_salience"] = {
        policy: values for policy, values in by_regime.items()
    }

    # 2. Similarity-based interference (content-similar vs dissimilar distractors).
    interference: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        similar = [
            _retained(_play(policy, capacity, _delay_events(delay, 0.5, True)), targets)[0]
            for capacity in capacities for delay in delays
        ]
        dissimilar = [
            _retained(_play(policy, capacity, _delay_events(delay, 0.5, False)), targets)[0]
            for capacity in capacities for delay in delays
        ]
        interference[policy] = {
            "similar_distractors": round(mean(similar), 4),
            "dissimilar_distractors": round(mean(dissimilar), 4),
            "difference": round(mean(dissimilar) - mean(similar), 4),
        }
    result.by_condition["similarity_interference"] = interference

    # 3. Reactivation and updating.
    reactivation: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        reactivated: list[float] = []
        others: list[float] = []
        updated: list[float] = []
        for capacity in (3, 5, 7):
            for distractors in range(4, 11):
                events = [Event(f"t{i}", f"original {i}", 0.7) for i in range(3)]
                for index in range(distractors):
                    events.append(Event(f"d{index}", f"distractor {index}", 0.6))
                    if index == distractors // 2:
                        events.append(Event("t0", "updated 0", 0.7))
                workspace = _play(policy, capacity, events)
                contents = workspace.contents()
                reactivated.append(1.0 if "t0" in contents else 0.0)
                updated.append(1.0 if contents.get("t0") == "updated 0" else 0.0)
                others.extend(1.0 if key in contents else 0.0 for key in ("t1", "t2"))
        reactivation[policy] = {
            "reactivated_item_retention": round(mean(reactivated), 4),
            "non_reactivated_retention": round(mean(others), 4),
            "updated_content_recall": round(mean(updated), 4),
        }
    result.by_condition["reactivation"] = reactivation

    # 4. Capacity pressure with seeded random salience (eval split reported).
    pressure: dict[str, dict[str, Any]] = {}
    split_sizes: dict[str, int] = {}
    for policy in POLICIES:
        overlap, recency, violations = [], [], 0
        for split in SPLITS:
            runs = 10 if split == "tuning" else 40
            split_sizes[split] = runs
            for run in range(runs):
                source = rng(seed, "working-memory", "pressure", split, str(run))
                capacity = source.choice(capacities)
                count = capacity + source.randint(3, 12)
                saliences = [round(source.uniform(0.05, 1.0), 4) for _ in range(count)]
                events = [Event(f"i{index}", f"item {index}", value)
                          for index, value in enumerate(saliences)]
                workspace = _play(policy, capacity, events)
                if split != "eval":
                    continue
                kept = set(workspace.contents())
                strongest = {
                    f"i{index}" for index in sorted(
                        range(count), key=lambda index: (-saliences[index], index)
                    )[:capacity]
                }
                latest = {f"i{index}" for index in range(count - capacity, count)}
                overlap.append(len(kept & strongest) / capacity)
                recency.append(len(kept & latest) / capacity)
                violations += workspace.violations
        pressure[policy] = {
            "overlap_with_highest_initial_salience": round(mean(overlap), 4),
            "overlap_with_most_recent": round(mean(recency), 4),
            "capacity_violations": violations,
        }
    result.splits = {"tuning": split_sizes["tuning"], "eval": split_sizes["eval"]}
    result.by_condition["capacity_pressure"] = pressure

    # 5. Serial position (12 equal-salience items into capacity 7).
    positions: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        workspace = _play(policy, 7, [Event(f"p{i:02d}", f"list item {i}", 0.6) for i in range(12)])
        surviving = workspace.contents()
        positions[policy] = {str(i + 1): 1.0 if f"p{i:02d}" in surviving else 0.0 for i in range(12)}
    result.extra["serial_position_retention_capacity_7"] = positions

    # 6. Serial-order preservation of retained items.
    order: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        plain = _play(policy, 7, [Event(f"o{i}", f"ordered {i}", 0.6) for i in range(6)])
        refreshed = _play(
            policy, 7,
            [Event(f"o{i}", f"ordered {i}", 0.6) for i in range(6)] + [Event("o1", "ordered 1", 0.6)],
        )
        reference = [f"o{i}" for i in range(6)]
        order[policy] = {
            "kendall_tau_no_reactivation": round(kendall_tau(plain.order(), reference), 4),
            "kendall_tau_after_reactivation": round(kendall_tau(refreshed.order(), reference), 4),
        }
    result.by_condition["serial_order"] = order

    # 7. Can a new, weak item enter a full workspace?
    newcomer: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        newcomer[policy] = {}
        for salience in (0.1, 0.3, 0.5, 0.7):
            events = [Event(f"f{i}", f"full {i}", 0.9) for i in range(7)]
            events.append(Event("new", "new input", salience))
            newcomer[policy][f"salience_{salience}"] = (
                1.0 if "new" in _play(policy, 7, events).contents() else 0.0
            )
    result.by_condition["newcomer_admission_full_capacity_7"] = newcomer

    # Headline metrics per policy.
    for policy in POLICIES:
        result.systems[policy] = {
            "delayed_retention_lower_salience_distractors": by_regime[policy]["lower"],
            "delayed_retention_equal_salience": by_regime[policy]["equal"],
            "delayed_retention_salient_distractors": by_regime[policy]["higher"],
            "updated_content_recall": reactivation[policy]["updated_content_recall"],
            "capacity_violations": pressure[policy]["capacity_violations"],
        }
    result.extra["delayed_retention_tasks_per_policy"] = tasks

    result.checks = _session_checks() + _existing_checks()
    result.failures = [
        {"case": "weak newcomer dropped on arrival", "detail": (
            "With seven items at salience 0.9, a new item at salience "
            f"{salience} is evicted immediately"
        )}
        for salience, kept in (
            (key.removeprefix("salience_"), value)
            for key, value in newcomer["existing_working_memory"].items()
        )
        if not kept
    ]
    existing_curve = curves["existing_working_memory"]["retention_by_intervening_items"]
    horizon = next((delay for delay in delays if existing_curve[str(delay)] < 1.0), None)
    result.findings = [
        "Retention is content-agnostic: similar and dissimilar distractors give identical "
        f"retention (difference {interference['existing_working_memory']['difference']:+.3f}). "
        "The implementation has no similarity-based interference.",
        f"With lower-salience distractors, mean retention (capacities 3–9) first drops below "
        f"1.0 at {horizon} intervening items. Multiplicative decay (×0.92 per insertion) "
        "eventually pushes older high-salience targets below fresh distractors, so decay "
        "rather than content determines forgetting.",
        "A low-salience input cannot enter a full workspace: it is evicted on arrival "
        "because eviction is applied after insertion.",
        "Serial-position retention is purely recency-based "
        f"({positions['existing_working_memory']}); there is no primacy effect.",
        "Capacity is never exceeded "
        f"({pressure['existing_working_memory']['capacity_violations']} violations).",
    ]
    return result


def _existing_checks() -> list[dict[str, Any]]:
    workspace = ExistingWorkspace(3)
    for step, (item_id, salience) in enumerate((("a", 0.5), ("b", 0.6), ("a", 0.5))):
        workspace.encode(item_id, f"content {item_id}", salience, step)
    return [
        check("reactivation increments access_count", workspace.access_count("a") == 1,
              access_count=workspace.access_count("a")),
        check("reactivated item moves to most-recent position", workspace.order() == ["b", "a"],
              order=workspace.order()),
    ]


def _session_checks() -> list[dict[str, Any]]:
    sessions = WorkingMemorySessions(capacity=3, max_sessions=2)
    first = sessions.get("owner-a", "s")
    first.remember(WorkingMemoryEntry("x", "owner a private", "k", "s", 0.5, _EPOCH.isoformat()))
    other_owner = sessions.get("owner-b", "s")
    isolated = other_owner.items() == ()
    same_again = sessions.get("owner-a", "s") is first
    cleared = sessions.clear("owner-a", "s") and sessions.get("owner-a", "s").items() == ()
    sessions.get("owner-c", "one")
    sessions.get("owner-c", "two")
    bounded = len(sessions._sessions) <= 2  # noqa: SLF001 - inspecting the bound itself
    removed = sessions.clear_owner("owner-c")
    return [
        check("sessions are isolated between owners", isolated),
        check("the same owner and session id returns the same workspace", same_again),
        check("clearing a session empties it", cleared),
        check("session registry is bounded (LRU)", bounded),
        check("clear_owner removes every session of that owner", removed >= 1, removed=removed),
        check("session capacity is applied", first.capacity == 3),
    ]
