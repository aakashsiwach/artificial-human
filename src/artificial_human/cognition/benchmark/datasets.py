"""Deterministic, synthetic task datasets for the cognitive evaluation suites.

All material is authored for this project (Apache-2.0). It contains no
personal data and no proprietary psychological test items. Generated tasks
use seeded RNG streams derived separately per suite, split and condition, so
tuning and evaluation tasks never share a random stream.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from artificial_human.cognition.architecture import CognitiveArchitecture
from artificial_human.cognition.attention import AttentionCandidate
from artificial_human.cognition.benchmark.common import SPLITS, Split, rng

# --------------------------------------------------------------------- attention

ATTENTION_CONDITIONS: tuple[str, ...] = (
    "clean_relevance",
    "salience_informative",
    "salient_distractors",
    "tied_relevance",
    "cross_source_normalisation",
)
ATTENTION_SET_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64)
ATTENTION_TASKS_PER_CELL: dict[Split, int] = {"tuning": 2, "eval": 6}


@dataclass(frozen=True, slots=True)
class AttentionTask:
    """A candidate set with hidden ground-truth usefulness grades (0, 1, 2).

    ``relevance`` and ``salience`` are the observable inputs the attention
    system receives; ``grades`` are the generator's ground truth.
    """

    id: str
    split: Split
    condition: str
    candidates: tuple[AttentionCandidate, ...]
    grades: dict[str, int]
    k: int


def _clip(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def _grade(random_source: random.Random) -> int:
    draw = random_source.random()
    return 2 if draw < 0.15 else 1 if draw < 0.35 else 0


def generate_attention_tasks(seed: int) -> list[AttentionTask]:
    tasks: list[AttentionTask] = []
    for split in SPLITS:
        for condition in ATTENTION_CONDITIONS:
            for size in ATTENTION_SET_SIZES:
                for index in range(ATTENTION_TASKS_PER_CELL[split]):
                    source = rng(seed, "attention", split, condition, str(size), str(index))
                    tasks.append(_attention_task(source, split, condition, size, index))
    return tasks


def _attention_task(
    source: random.Random, split: Split, condition: str, size: int, index: int
) -> AttentionTask:
    random_source = source
    labels = [f"c{number:03d}" for number in range(size)]
    random_source.shuffle(labels)
    grades = {label: _grade(random_source) for label in labels}
    if not any(grades.values()):
        grades[labels[0]] = 2
    gauss = random_source.gauss
    uniform = random_source.uniform
    candidates: list[AttentionCandidate] = []
    if condition == "cross_source_normalisation":
        candidates = _cross_source_candidates(random_source, labels, grades)
    else:
        for label in labels:
            grade = grades[label]
            if condition == "clean_relevance":
                relevance, salience = _clip(0.2 + 0.3 * grade + gauss(0, 0.1)), _clip(uniform(0, 1))
            elif condition == "salience_informative":
                relevance = _clip(0.35 + 0.15 * grade + gauss(0, 0.2))
                salience = _clip(0.15 + 0.35 * grade + gauss(0, 0.1))
            elif condition == "salient_distractors":
                if grade == 0 and random_source.random() < 0.3:
                    relevance, salience = _clip(0.55 + gauss(0, 0.05)), _clip(uniform(0.9, 1.0))
                elif grade == 0:
                    relevance, salience = _clip(0.15 + gauss(0, 0.1)), _clip(uniform(0, 0.3))
                else:
                    relevance = _clip(0.45 + 0.15 * grade + gauss(0, 0.05))
                    salience = _clip(uniform(0, 0.2))
            elif condition == "tied_relevance":
                relevance, salience = 0.5, _clip(0.1 + 0.35 * grade + gauss(0, 0.08))
            else:
                raise ValueError(f"unknown attention condition {condition!r}")
            candidates.append(AttentionCandidate(label, "memory", relevance, salience))
    k = min(5, max(2, size // 4))
    return AttentionTask(
        id=f"attention-{split}-{condition}-{size}-{index}", split=split, condition=condition,
        candidates=tuple(candidates), grades=grades, k=k,
    )


def _cross_source_candidates(
    source: random.Random, labels: Sequence[str], grades: dict[str, int]
) -> list[AttentionCandidate]:
    """Memories and learning examples normalised per source, exactly as the cycle does.

    Learning examples are weak (grade 0) lexical matches. The architecture's
    per-source max-normalisation still gives the best of them relevance 1.0.
    """
    gauss = source.gauss
    uniform = source.uniform
    memories: list[tuple[str, float]] = []
    examples: list[tuple[str, float]] = []
    for position, label in enumerate(labels):
        if position % 4 == 3:  # one quarter of candidates are learning examples
            grades[label] = 0
            examples.append((label, max(0.05, uniform(0.1, 0.6))))
        else:
            memories.append((label, max(0.05, 0.3 + 1.5 * grades[label] + gauss(0, 0.2))))
    if not any(grades[label] for label, _ in memories):
        grades[memories[0][0]] = 2
        memories[0] = (memories[0][0], 3.3)
    normalise = CognitiveArchitecture._normalise_scores
    by_label = {label: index for index, label in enumerate(labels)}
    candidates = [
        AttentionCandidate(label, "memory", round(relevance, 4))
        for label, relevance in normalise(memories)
    ] + [
        AttentionCandidate(label, "learning_example", round(relevance, 4))
        for label, relevance in normalise(examples)
    ]
    return sorted(candidates, key=lambda item: by_label[item.reference_id])


# -------------------------------------------------------------------- prediction


@dataclass(frozen=True, slots=True)
class PredictionItem:
    """An input judged against an active context.

    ``unexpected`` is the author-assigned ground truth: 1 when the input
    conveys information not entailed by, or inconsistent with, the context
    (unrelated content, a different word sense, a contradiction, or a
    script violation); 0 when it repeats, paraphrases, or is a typical
    continuation of the context.
    """

    id: str
    split: Split
    category: str
    context: tuple[str, ...]
    text: str
    unexpected: int


_PREDICTION_SOURCE: dict[str, tuple[int, list[tuple[tuple[str, ...], str]]]] = {
    "exact_repetition": (0, [
        (("The hippocampus supports episodic memory.",), "The hippocampus supports episodic memory."),
        (("My sister lives in Pune.",), "My sister lives in Pune."),
        (("The meeting starts at nine tomorrow.",), "The meeting starts at nine tomorrow."),
        (("Water boils at one hundred degrees Celsius.",), "Water boils at one hundred degrees Celsius."),
        (("The train to Lisbon leaves from platform four.",), "The train to Lisbon leaves from platform four."),
        (("Insulin lowers blood glucose.",), "Insulin lowers blood glucose."),
    ]),
    "near_repetition": (0, [
        (("The hippocampus supports episodic memory.",), "The hippocampus supports episodic memories."),
        (("My sister lives in Pune.",), "My sister lives in Pune now."),
        (("The meeting starts at nine tomorrow.",), "Tomorrow the meeting starts at nine."),
        (("Water boils at one hundred degrees Celsius.",), "Water boils at 100 degrees Celsius."),
        (("The train to Lisbon leaves from platform four.",), "The Lisbon train leaves from platform four."),
        (("Insulin lowers blood glucose.",), "Insulin lowers the blood glucose level."),
    ]),
    "paraphrase": (0, [
        (("The hippocampus supports episodic memory.",), "Remembering personal events depends on the hippocampus."),
        (("My sister lives in Pune.",), "Pune is where my sister has her home."),
        (("The meeting starts at nine tomorrow.",), "Tomorrow's meeting begins at 9 o'clock."),
        (("Water boils at one hundred degrees Celsius.",), "At one hundred degrees Celsius, water starts to boil."),
        (("The car would not start this morning.",), "This morning the engine refused to turn over."),
        (("She finished the marathon in under four hours.",), "Her marathon time was below four hours."),
    ]),
    "low_overlap_related": (0, [
        (("The patient was given insulin after the meal.",), "Her blood sugar dropped afterwards."),
        (("We packed the tent and sleeping bags.",), "Camping trip preparations are complete."),
        (("The orchestra tuned their violins.",), "A concert is about to begin."),
        (("It has rained heavily for three days.",), "The rivers are flooding."),
        (("I booked a flight to Tokyo.",), "Japan travel plans are confirmed."),
        (("The chef sharpened her knives.",), "Dinner preparation is underway in the kitchen."),
    ]),
    "unrelated": (1, [
        (("The hippocampus supports episodic memory.",), "Neptune has fourteen known moons."),
        (("My sister lives in Pune.",), "Photosynthesis converts sunlight into chemical energy."),
        (("The meeting starts at nine tomorrow.",), "Volcanoes release sulfur dioxide."),
        (("Water boils at one hundred degrees Celsius.",), "Chess grandmasters study openings."),
        (("The train to Lisbon leaves from platform four.",), "Octopuses have three hearts."),
        (("Insulin lowers blood glucose.",), "The library closes on Sundays."),
    ]),
    "shared_vocabulary_unrelated": (1, [
        (("The bank approved my loan application.",), "We had a picnic on the river bank."),
        (("The bat flew out of the cave at dusk.",), "He bought a new cricket bat."),
        (("She will train the new engineers.",), "The train was delayed by snow."),
        (("The python escaped from the zoo enclosure.",), "Python is my favourite programming language."),
        (("Turn the key to open the lock.",), "The key point of the lecture was memory consolidation."),
        (("The match was cancelled because of rain.",), "He lit the candle with a match."),
    ]),
    "contradiction": (1, [
        (("My sister lives in Pune.",), "My sister does not live in Pune."),
        (("The meeting starts at nine tomorrow.",), "The meeting starts at eleven tomorrow."),
        (("Water boils at one hundred degrees Celsius.",), "Water boils at fifty degrees Celsius."),
        (("The shop is open on Sundays.",), "The shop is closed on Sundays."),
        (("The user is allergic to peanuts.",), "The user is not allergic to peanuts."),
        (("The project deadline is in March.",), "The project deadline is in June."),
    ]),
    "contextually_expected": (0, [
        (("We sat down at the restaurant.", "The waiter brought the menus."), "We ordered pasta and a salad."),
        (("I arrived at the airport early.", "I checked in my suitcase."), "Then I went through security screening."),
        (("The surgeon washed her hands.", "The patient was anaesthetised."), "She made the first incision."),
        (("The students entered the exam hall.", "Invigilators handed out papers."), "Everyone began writing answers."),
        (("I opened the laptop.", "I logged into the build server."), "The test suite started running."),
        (("We pitched the tent by the lake.", "The sun began to set."), "We lit a campfire for dinner."),
    ]),
    "contextually_unexpected": (1, [
        (("We sat down at the restaurant.", "The waiter brought the menus."), "The waiter then performed a magic trick with the menus."),
        (("I arrived at the airport early.", "I checked in my suitcase."), "Then a penguin checked in at the airport ahead of me."),
        (("The surgeon washed her hands.", "The patient was anaesthetised."), "The patient sat up and asked the surgeon for coffee."),
        (("The students entered the exam hall.", "Invigilators handed out papers."), "The invigilators handed out ice cream instead of papers."),
        (("I opened the laptop.", "I logged into the build server."), "The build server began playing opera music."),
        (("We pitched the tent by the lake.", "The sun began to set."), "A herd of zebras walked through the tent by the lake."),
    ]),
}
# The first two items of every category are tuning items; the rest are eval.
PREDICTION_TUNING_PER_CATEGORY = 2


def prediction_items() -> list[PredictionItem]:
    items: list[PredictionItem] = []
    for category, (label, entries) in _PREDICTION_SOURCE.items():
        for index, (context, text) in enumerate(entries):
            split: Split = "tuning" if index < PREDICTION_TUNING_PER_CATEGORY else "eval"
            items.append(PredictionItem(
                id=f"prediction-{category}-{index + 1}", split=split, category=category,
                context=context, text=text, unexpected=label,
            ))
    return items


def validate_split_isolation(ids_by_split: dict[str, Sequence[str]]) -> None:
    """Raise if any task identifier appears in more than one split."""
    seen: dict[str, str] = {}
    for split, ids in ids_by_split.items():
        for item in ids:
            if item in seen and seen[item] != split:
                raise ValueError(f"task {item!r} appears in both {seen[item]} and {split}")
            seen[item] = split
