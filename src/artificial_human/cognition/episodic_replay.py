from __future__ import annotations

import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Final, Literal

from artificial_human.memory.temporal import parse_stored_timestamp


_EPISODIC_REPLAY_VERSION: Final[str] = "contextual-episodic-replay-v1"
_CONTEXT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MEMORY_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True, slots=True)
class EpisodicEvent:
    """An opaque reference to one consent-visible episodic memory."""

    memory_id: str
    occurred_at: str


@dataclass(frozen=True, slots=True)
class TemporalReplay:
    """One adjacent event recovered from an episode, relative to a cue."""

    memory_id: str
    cue_memory_id: str
    relation: Literal["before", "after"]
    distance: int
    strength: float


@dataclass(frozen=True, slots=True)
class _Episode:
    events: tuple[EpisodicEvent, ...]


class EpisodicReplayMemory:
    """Bounded, volatile temporal sequences over opaque episodic memory IDs.

    The network stores only memory IDs, timestamps, hashed/non-content context IDs,
    and scalar transition strengths. It never copies memory text, source labels,
    owner IDs, or raw session IDs. Callers must pass only episodic records already
    selected by attention and must resolve replayed IDs again through the same
    owner's active-consent repository before exposing them to the cognitive cycle.

    Each record_episode call represents one co-attended event sequence. Events are
    ordered by event time; replay returns immediate temporal neighbours in either
    direction. This is a computational sequence-retrieval abstraction, not a
    biological hippocampal replay model.
    """

    def __init__(
        self,
        *,
        max_contexts: int = 128,
        max_episodes_per_context: int = 32,
        max_events_per_episode: int = 12,
        max_transitions: int = 1024,
        learning_rate: float = 0.1,
        retrieval_rate: float = 0.08,
        decay: float = 0.995,
    ) -> None:
        for name, value in (
            ("max_contexts", max_contexts),
            ("max_episodes_per_context", max_episodes_per_context),
            ("max_events_per_episode", max_events_per_episode),
            ("max_transitions", max_transitions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0.0 <= retrieval_rate <= 1.0:
            raise ValueError("retrieval_rate must be between 0 and 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.max_contexts = max_contexts
        self.max_episodes_per_context = max_episodes_per_context
        self.max_events_per_episode = max_events_per_episode
        self.max_transitions = max_transitions
        self.learning_rate = learning_rate
        self.retrieval_rate = retrieval_rate
        self.decay = decay
        self._episodes: OrderedDict[str, deque[_Episode]] = OrderedDict()
        self._strengths: dict[tuple[str, str, str], float] = {}
        self._episodes_recorded = 0
        self._replay_events = 0
        self._retrieval_strengthenings = 0
        self._evicted_transitions = 0

    @staticmethod
    def _validate_context(context_id: str) -> None:
        if not isinstance(context_id, str) or not _CONTEXT_PATTERN.fullmatch(context_id):
            raise ValueError("context_id must be a short lowercase non-content identifier")

    @staticmethod
    def _validate_memory_id(memory_id: str) -> None:
        if not isinstance(memory_id, str) or not _MEMORY_ID_PATTERN.fullmatch(memory_id):
            raise ValueError("memory_id must be a 1-64 character opaque identifier")

    @staticmethod
    def _key(context_id: str, cue_id: str, target_id: str) -> tuple[str, str, str]:
        return context_id, cue_id, target_id

    @staticmethod
    def _pair(first: str, second: str) -> tuple[str, str]:
        return (first, second) if first < second else (second, first)

    def record_episode(
        self,
        events: tuple[EpisodicEvent, ...] | list[EpisodicEvent],
        *,
        context_id: str,
    ) -> bool:
        """Store one co-attended, time-ordered episode if at least two events exist."""
        self._validate_context(context_id)
        unique: dict[str, EpisodicEvent] = {}
        for event in events:
            self._validate_memory_id(event.memory_id)
            parse_stored_timestamp(event.occurred_at)
            unique[event.memory_id] = event
        if len(unique) < 2:
            return False

        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda item: (parse_stored_timestamp(item.occurred_at), item.memory_id),
            )[-self.max_events_per_episode :]
        )
        if len(ordered) < 2:
            return False

        if context_id not in self._episodes and len(self._episodes) >= self.max_contexts:
            oldest_context, _ = self._episodes.popitem(last=False)
            self._drop_context_transitions(oldest_context)
        # Forgetting advances only when this context is active.
        for key, strength in tuple(self._strengths.items()):
            if key[0] != context_id:
                continue
            decayed = strength * self.decay
            if decayed < 0.001:
                del self._strengths[key]
            else:
                self._strengths[key] = decayed

        episodes = self._episodes.setdefault(
            context_id, deque(maxlen=self.max_episodes_per_context)
        )
        self._episodes.move_to_end(context_id)
        if len(episodes) == episodes.maxlen:
            self._drop_episode_transitions(context_id, episodes[0])
        episodes.append(_Episode(events=ordered))
        self._episodes_recorded += 1

        # Store bidirectional adjacent transitions, not all-to-all links.
        for first, second in zip(ordered, ordered[1:], strict=False):
            for cue_id, target_id in (
                (first.memory_id, second.memory_id),
                (second.memory_id, first.memory_id),
            ):
                key = self._key(context_id, cue_id, target_id)
                if key not in self._strengths and len(self._strengths) >= self.max_transitions:
                    weakest = min(self._strengths, key=lambda item: (self._strengths[item], item))
                    del self._strengths[weakest]
                    self._evicted_transitions += 1
                current = self._strengths.get(key, 0.0)
                self._strengths[key] = min(
                    1.0, current + self.learning_rate * (1.0 - current)
                )
        return True

    def episodes_for_context_prefix(
        self, context_prefix: str
    ) -> tuple[tuple[str, tuple[EpisodicEvent, ...]], ...]:
        """Return bounded episode snapshots for one hashed owner-context prefix.

        Context identifiers are returned only to in-process consolidation code so
        it can require recurrence across distinct sessions. They are hashes, not
        raw owner IDs, session IDs, prompts, or memory content.
        """
        if (
            not isinstance(context_prefix, str)
            or not context_prefix
            or len(context_prefix) > 64
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", context_prefix)
        ):
            raise ValueError("context_prefix must be a short lowercase non-content prefix")
        return tuple(
            (context_id, episode.events)
            for context_id, episodes in self._episodes.items()
            if context_id.startswith(context_prefix)
            for episode in episodes
        )

    def replay(
        self,
        cue_memory_ids: tuple[str, ...] | list[str],
        *,
        context_id: str,
        blocked_pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        limit: int = 5,
    ) -> tuple[TemporalReplay, ...]:
        """Recover immediate earlier/later neighbours for cues in this context."""
        self._validate_context(context_id)
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        cues = set(cue_memory_ids)
        for cue in cues:
            self._validate_memory_id(cue)
        blocked: set[tuple[str, str]] = set()
        for first, second in blocked_pairs:
            self._validate_memory_id(first)
            self._validate_memory_id(second)
            if first == second:
                raise ValueError("An episodic memory cannot contradict itself")
            blocked.add(self._pair(first, second))

        # Repeated retrieval within a context advances forgetting locally.
        for key, strength in tuple(self._strengths.items()):
            if key[0] != context_id:
                continue
            decayed = strength * self.decay
            if decayed < 0.001:
                del self._strengths[key]
            else:
                self._strengths[key] = decayed

        episodes = self._episodes.get(context_id, ())
        candidates: dict[str, TemporalReplay] = {}
        for episode in reversed(episodes):
            items = episode.events
            for index, event in enumerate(items):
                if event.memory_id not in cues:
                    continue
                neighbours: tuple[tuple[int, Literal["before", "after"]], ...] = (
                    (index - 1, "before"),
                    (index + 1, "after"),
                )
                for neighbour_index, relation in neighbours:
                    if not 0 <= neighbour_index < len(items):
                        continue
                    neighbour = items[neighbour_index]
                    pair = self._pair(event.memory_id, neighbour.memory_id)
                    key = self._key(context_id, event.memory_id, neighbour.memory_id)
                    if pair in blocked:
                        self._strengths.pop(key, None)
                        continue
                    replay_strength = self._strengths.get(key)
                    if replay_strength is None:
                        continue
                    candidate = TemporalReplay(
                        memory_id=neighbour.memory_id,
                        cue_memory_id=event.memory_id,
                        relation=relation,
                        distance=1,
                        strength=round(replay_strength, 6),
                    )
                    current = candidates.get(neighbour.memory_id)
                    if current is None or candidate.strength > current.strength:
                        candidates[neighbour.memory_id] = candidate

        ranked = sorted(
            candidates.values(),
            key=lambda item: (-item.strength, item.distance, item.memory_id),
        )[:limit]
        self._replay_events += len(ranked)
        return tuple(ranked)

    def reinforce_in_context(
        self,
        replayed: tuple[TemporalReplay, ...] | list[TemporalReplay],
        *,
        context_id: str,
    ) -> int:
        """Strengthen only selected transitions in the supplied context."""
        self._validate_context(context_id)
        changed = 0
        for item in replayed:
            key = self._key(context_id, item.cue_memory_id, item.memory_id)
            current = self._strengths.get(key)
            if current is None:
                continue
            updated = current + self.retrieval_rate * (1.0 - current)
            self._strengths[key] = min(1.0, updated)
            self._retrieval_strengthenings += 1
            changed += 1
        return changed

    def _drop_episode_transitions(self, context_id: str, episode: _Episode) -> None:
        selected: set[tuple[str, str, str]] = set()
        for first, second in zip(episode.events, episode.events[1:], strict=False):
            selected.add(self._key(context_id, first.memory_id, second.memory_id))
            selected.add(self._key(context_id, second.memory_id, first.memory_id))
        other_episodes = [
            item for item in self._episodes.get(context_id, ()) if item is not episode
        ]
        for key in selected:
            still_referenced = any(
                any(
                    (left.memory_id, right.memory_id) == (key[1], key[2])
                    or (left.memory_id, right.memory_id) == (key[2], key[1])
                    for left, right in zip(item.events, item.events[1:], strict=False)
                )
                for item in other_episodes
            )
            if not still_referenced:
                self._strengths.pop(key, None)

    def _drop_context_transitions(self, context_id: str) -> None:
        self._strengths = {
            key: value for key, value in self._strengths.items() if key[0] != context_id
        }

    def forget_context(self, context_id: str) -> int:
        """Delete one owner's/session's sequence traces and transition strengths."""
        self._validate_context(context_id)
        removed = self._episodes.pop(context_id, None)
        before = len(self._strengths)
        self._drop_context_transitions(context_id)
        return before - len(self._strengths) + (len(removed) if removed else 0)

    def forget_contexts(self, context_prefix: str) -> int:
        if (
            not isinstance(context_prefix, str)
            or not context_prefix
            or len(context_prefix) > 64
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", context_prefix)
        ):
            raise ValueError("context_prefix must be a short lowercase non-content prefix")
        selected = [key for key in self._episodes if key.startswith(context_prefix)]
        removed = 0
        for context_id in selected:
            removed += self.forget_context(context_id)
        return removed

    def reset(self) -> None:
        self._episodes.clear()
        self._strengths.clear()
        self._episodes_recorded = 0
        self._replay_events = 0
        self._retrieval_strengthenings = 0
        self._evicted_transitions = 0

    def snapshot(self) -> dict[str, object]:
        strengths = tuple(self._strengths.values())
        return {
            "version": _EPISODIC_REPLAY_VERSION,
            "context_count": len(self._episodes),
            "episode_count": sum(len(items) for items in self._episodes.values()),
            "transition_count": len(self._strengths),
            "episodes_recorded": self._episodes_recorded,
            "replay_events": self._replay_events,
            "retrieval_strengthenings": self._retrieval_strengthenings,
            "evicted_transitions": self._evicted_transitions,
            "mean_transition_strength": (
                round(sum(strengths) / len(strengths), 6) if strengths else 0.0
            ),
            "max_transitions": self.max_transitions,
            "learning_rate": self.learning_rate,
            "retrieval_rate": self.retrieval_rate,
            "decay": self.decay,
            "state_storage": "volatile_process_memory",
            "retains_memory_content": False,
            "retains_owner_or_session_ids": False,
        }
