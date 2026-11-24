"""Integrated multi-turn scenarios through the real cognitive cycle.

Each scenario builds a fresh in-memory organism stack (repository, long-term
memory, learning system, ``CognitiveArchitecture``, working-memory sessions)
and runs consecutive turns. No remote model is used. Scenario E uses a stub
language component only to inspect the evidence passed to it; the stub's
answer is never scored.

A failed check is a measured finding about current behaviour, not an
execution error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from artificial_human.brain.providers.base import ChatMessage, Generation
from artificial_human.brain.service import BrainService
from artificial_human.cognition.architecture import CognitiveArchitecture, CognitiveCycle
from artificial_human.cognition.benchmark.common import SuiteResult, check
from artificial_human.cognition.working_memory import WorkingMemorySessions
from artificial_human.core.models import MemoryKind
from artificial_human.identity.profile import IdentityProfile
from artificial_human.learning.system import LearningSystem
from artificial_human.memory.embeddings import EmbeddingProvider
from artificial_human.memory.repository import MemoryRepository
from artificial_human.memory.search import SearchFilters
from artificial_human.memory.system import LongTermMemory

OWNER = "integration-owner"


@dataclass
class Stack:
    repository: MemoryRepository
    memory: LongTermMemory
    architecture: CognitiveArchitecture
    sessions: WorkingMemorySessions
    consent: str
    clock: dict[str, datetime] = field(default_factory=dict)

    def remember(self, content: str, *, when: str = "2024-01-01T00:00:00+00:00",
                 **fields: Any) -> str:
        self.clock["now"] = datetime.fromisoformat(when)
        fields.setdefault("kind", MemoryKind.SEMANTIC)
        record, _ = self.memory.add(
            owner_id=OWNER, consent_id=self.consent, content=content,
            source=fields.pop("source", "scenario"), **fields,
        )
        return record.id

    def turn(self, text: str, session: str = "main") -> CognitiveCycle:
        return self.architecture.run(
            owner_id=OWNER, query=text, workspace=self.sessions.get(OWNER, session),
        )


def _stack(provider: EmbeddingProvider | None) -> Stack:
    clock = {"now": datetime(2024, 1, 1, tzinfo=UTC)}
    repository = MemoryRepository(":memory:", clock=lambda: clock["now"])
    memory = LongTermMemory(repository, provider)
    architecture = CognitiveArchitecture(
        memory_search=memory.search_engine, learning=LearningSystem(repository)
    )
    consent = repository.grant_consent(OWNER).id
    return Stack(repository, memory, architecture, WorkingMemorySessions(), consent, clock)


def _selected_ids(cycle: CognitiveCycle) -> list[str]:
    return [match.memory.id for match in cycle.memories]


def _wm_contents(cycle: CognitiveCycle, stack: Stack, session: str = "main") -> list[str]:
    return [item.content for item in stack.sessions.get(OWNER, session).items()]


def scenario_a(stack: Stack) -> list[dict[str, Any]]:
    """Retain and recall across unrelated input and a distractor."""
    target = stack.remember("The spare house key is kept under the blue flowerpot by the door.")
    stack.remember("The car key is in the kitchen drawer.")
    stack.remember("Blue paint is needed for the garden fence.")
    before = len(stack.repository.iter_visible(OWNER))
    first = stack.turn("My spare house key is under the blue flowerpot.")
    stack.turn("The quarterly report needs two more charts.")
    stack.turn("Remind me that the dentist appointment moved to Friday.")
    probe = stack.turn("Where can I find the backup key to get into my home?")
    keyword = stack.turn("Where is the spare house key?")
    selected = _selected_ids(probe)
    return [
        check("paraphrased question selects the stored fact", target in selected,
              rank=selected.index(target) + 1 if target in selected else None),
        check("paraphrased question ranks the stored fact first",
              bool(selected) and selected[0] == target),
        check("keyword question selects the stored fact", target in _selected_ids(keyword)),
        check("turn-1 input is still in working memory after 3 turns",
              first.percept.content in _wm_contents(probe, stack)),
        check("conversation turns were not written to long-term memory",
              len(stack.repository.iter_visible(OWNER)) == before),
    ]


def scenario_b(stack: Stack) -> list[dict[str, Any]]:
    """Temporal knowledge: now, previously, and what was known earlier."""
    old = stack.remember(
        "The owner works as a data analyst at Riverbank Insurance.",
        when="2024-02-01T00:00:00+00:00", fact_key="owner.employer",
        valid_from="2023-01-01T00:00:00+00:00",
    )
    stack.clock["now"] = datetime(2025, 6, 3, tzinfo=UTC)
    new, _, _ = stack.memory.supersede(
        owner_id=OWNER, memory_id=old, consent_id=stack.consent,
        content="The owner now works as a research engineer at Northlight Labs.",
        source="scenario", valid_from="2025-06-01T00:00:00+00:00",
    )
    now = _selected_ids(stack.turn("Where does the owner work now?"))
    before = _selected_ids(stack.turn("Where did the owner work previously?"))
    known = stack.memory.search(
        OWNER, "owner works employer", filters=SearchFilters(
            known_at=datetime(2025, 1, 1, tzinfo=UTC)
        ),
    )
    valid = stack.memory.search(
        OWNER, "owner works employer", filters=SearchFilters(
            valid_at=datetime(2024, 1, 1, tzinfo=UTC)
        ),
    )
    return [
        check("'now' selects the current fact first", bool(now) and now[0] == new.id),
        check("'now' excludes the superseded fact", old not in now),
        check("'previously' selects the superseded fact first", bool(before) and before[0] == old),
        check("known_at before the update returns only the old fact",
              [item.memory.id for item in known.results] == [old]),
        check("valid_at 2024 returns only the old fact",
              [item.memory.id for item in valid.results] == [old]),
        check("both versions keep their own provenance",
              [item.id for item in stack.repository.version_chain(OWNER, old)] == [old, new.id]),
    ]


def scenario_c(stack: Stack) -> list[dict[str, Any]]:
    """Attention under high-salience distraction."""
    relevant = {
        stack.remember("Feed the sourdough starter twice a day with equal flour and water."),
        stack.remember("A sourdough starter fed daily should stay at room temperature."),
    }
    distractors = {
        stack.remember("IMPORTANT: the sourdough bakery downtown has closed permanently.",
                       metadata={"salience": 1.0}),
        stack.remember("IMPORTANT: the water bill is due on Friday.", metadata={"salience": 1.0}),
    }
    stack.remember("The cat prefers the sunny windowsill.")
    cycle = stack.turn("How often should I feed my sourdough starter?")
    weights = {
        item_id: weight for item_id, kind, weight, _ in cycle.attended if kind == "memory"
    }
    ranked = sorted(weights, key=lambda item: -weights[item])
    selected = set(_selected_ids(cycle))
    best_relevant = max((weights[item] for item in relevant if item in weights), default=0.0)
    return [
        check("highest-weighted memory is relevant", bool(ranked) and ranked[0] in relevant),
        check("all relevant memories are selected", relevant <= selected),
        check("no high-salience distractor outweighs a relevant memory",
              all(weights.get(item, 0.0) <= best_relevant for item in distractors)),
        check("selected memory precision", True,
              precision=round(len(selected & relevant) / len(selected), 4) if selected else 0.0,
              selected=len(selected)),
    ]


def scenario_d(stack: Stack) -> list[dict[str, Any]]:
    """Working-memory interference over many turns in one session."""
    fact = stack.remember("The garage gate code is 4417.")
    turns = [
        "The garage gate code matters tonight.",
        "Water the tomatoes this evening.",
        "Book a table for Saturday.",
        "The printer needs new toner.",
        "Call the plumber about the leak.",
        "Renew the library books.",
        "The garage gate code matters tonight.",
        "Check the tyre pressure.",
        "What is the garage gate code?",
    ]
    cycles = [stack.turn(text) for text in turns]
    workspace = stack.sessions.get(OWNER, "main")
    items = workspace.items()
    ids = {item.id for item in items}
    fact_entry = next((item for item in items if item.id == f"memory:{fact}"), None)
    retained_turns = [
        index for index, cycle in enumerate(cycles) if f"percept:{cycle.percept.id}" in ids
    ]
    memory_slots = sum(1 for item in items if item.id.startswith("memory:"))
    return [
        check("capacity is never exceeded", len(items) <= workspace.capacity, items=len(items)),
        check("final question retrieves the stored fact", fact in _selected_ids(cycles[-1])),
        check("re-retrieved stored fact is reactivated (access_count > 0)",
              fact_entry is not None and fact_entry.access_count > 0,
              access_count=fact_entry.access_count if fact_entry else None),
        check("an identical repeated input reactivates its earlier entry",
              cycles[0].percept.id == cycles[6].percept.id,
              detail="percept identifiers are unique per turn"),
        check("displacement is recency-ordered (retained turns are the most recent ones)",
              retained_turns == list(range(len(turns) - len(retained_turns), len(turns))),
              retained_turn_indices=retained_turns, memory_slots=memory_slots),
    ]


class _CapturingProvider:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage], *, temperature: float = 0.2,
                 max_tokens: int = 1200) -> Generation:
        self.messages = messages
        return Generation(content="(stub output, not scored)", model="stub")


def scenario_e(stack: Stack) -> list[dict[str, Any]]:
    """Contradiction awareness through the full response path (stub language component)."""
    first = stack.remember("The owner is allergic to peanuts.", source="medical form 2024",
                           fact_key="owner.allergy")
    second = stack.remember("The owner has no food allergies.", source="casual chat note",
                            fact_key="owner.allergy", when="2025-03-01T00:00:00+00:00")
    before = {item.id: item.epistemic_status for item in stack.repository.iter_visible(OWNER)}
    provider = _CapturingProvider()
    service = BrainService(provider, stack.architecture.learning,
                           architecture=stack.architecture)
    response = service.respond(
        owner_id=OWNER, query="Is the owner allergic to peanuts?",
        profile=IdentityProfile(owner_id=OWNER, display_name="Synthetic owner"),
    )
    recalled = {item["memory_id"] for item in response.recalled_memories}
    evidence = provider.messages[1].content if provider.messages else ""
    after = {item.id: item.epistemic_status for item in stack.repository.iter_visible(OWNER)}
    return [
        check("both conflicting memories reach the language component", {first, second} <= recalled),
        check("evidence marks the conflict", "conflicts_with" in evidence and second in evidence),
        check("sources of both claims are preserved",
              "medical form 2024" in evidence and "casual chat note" in evidence),
        check("no claim was promoted, rewritten, or added", before == after),
        check("model is told not to resolve conflicts silently",
              "present the conflict" in provider.messages[0].content),
    ]


def scenario_f(stack: Stack) -> list[dict[str, Any]]:
    """Context switching between sessions and an elliptical multi-turn follow-up."""
    sister = stack.remember("The owner's sister lives in Pune.")
    stack.remember("The owner's colleague lives in Delhi.", when="2024-02-01T00:00:00+00:00")
    stack.turn("Plan the Lisbon trip itinerary.", session="travel")
    stack.turn("Track my blood pressure readings.", session="health")
    stack.turn("Add the tram museum to the itinerary.", session="travel")
    travel = [item.content for item in stack.sessions.get(OWNER, "travel").items()]
    health = [item.content for item in stack.sessions.get(OWNER, "health").items()]
    stack.turn("Tell me about my sister.", session="family")
    follow_up = _selected_ids(stack.turn("Where does she live?", session="family"))
    without_context = _selected_ids(stack.turn("Where does she live?", session="fresh"))
    return [
        check("sessions do not share working memory",
              not any("blood pressure" in text for text in travel)
              and not any("Lisbon" in text for text in health)),
        check("switching back keeps the earlier session context",
              any("Lisbon" in text for text in travel)),
        check("elliptical follow-up ranks the sister's city first",
              bool(follow_up) and follow_up[0] == sister, selected=len(follow_up)),
        check("the preceding turn changes what is retrieved (conversational context is used)",
              follow_up != without_context,
              detail="retrieval receives only the current turn"),
    ]


SCENARIOS: dict[str, tuple[str, Callable[[Stack], list[dict[str, Any]]]]] = {
    "A_retain_and_recall": ("Retain and recall across distraction", scenario_a),
    "B_temporal_knowledge": ("Current, previous and earlier-known facts", scenario_b),
    "C_attention_under_distraction": ("Attention with high-salience distractors", scenario_c),
    "D_working_memory_interference": ("Working-memory interference over turns", scenario_d),
    "E_contradiction_awareness": ("Contradictions preserved through to the model", scenario_e),
    "F_context_switching_multi_turn": ("Session switching and elliptical follow-up", scenario_f),
}


def run_integration_suite(provider: EmbeddingProvider | None = None) -> SuiteResult:
    configs: dict[str, EmbeddingProvider | None] = {"lexical_temporal": None}
    if provider is not None:
        configs["hybrid_temporal"] = provider
    result = SuiteResult(
        suite="integration",
        title="Integrated multi-turn cognitive-cycle scenarios",
        evidence_category="software behaviour in end-to-end scenarios (deterministic, no LLM)",
        splits={"eval": len(SCENARIOS)},
    )
    for config, backend in configs.items():
        passed_total = checks_total = 0
        for name, (description, scenario) in SCENARIOS.items():
            outcomes = scenario(_stack(backend))
            for outcome in outcomes:
                result.checks.append({"scenario": name, "config": config, **outcome})
                if not outcome["passed"]:
                    result.failures.append({"scenario": name, "config": config,
                                            "check": outcome["name"]})
            passed = sum(outcome["passed"] for outcome in outcomes)
            passed_total += passed
            checks_total += len(outcomes)
            result.by_condition.setdefault(name, {})[config] = {
                "description": description, "passed": passed, "checks": len(outcomes),
            }
        result.systems[config] = {
            "checks_passed": passed_total, "checks_total": checks_total,
            "pass_rate": round(passed_total / checks_total, 4),
        }
    result.findings = [
        f"{config}: {values['checks_passed']}/{values['checks_total']} behavioural checks passed."
        for config, values in result.systems.items()
    ] + [
        "Each cycle retrieves with the current turn only, so elliptical follow-ups cannot "
        "use the conversational context held in working memory.",
        "A repeated input creates a new percept entry instead of reactivating the earlier "
        "one, because percept identifiers are unique per turn.",
        "Retrieved memories compete with percepts for the same seven workspace slots, so "
        "the turn-1 input is displaced after a few turns that retrieve several memories.",
        "Lexical retrieval found nothing for 'Where does she live?' because the light "
        "stemmer maps 'lives' to 'liv' but 'live' to 'live'. This is recorded as a "
        "measured limitation and has not been changed in this phase.",
    ]
    return result
