"""Human digit-span reference (NEMAR on003838) and a software comparison.

Source: Pavlov, Kasanov, Kosachenko, Kotyusov & Busch (2022), "Pupillometry and
electroencephalography in the digit span task", Scientific Data 9, 325.
NEMAR on003838 v1.0.0 (DOI 10.82901/nemar.on003838), CC0; mirrors OpenNeuro
ds003838 v1.0.6. Only the behavioural files ``sub-*_task-memory_beh.tsv`` are
read. Auditory digit sequences of 5, 9 or 13 digits were followed by serial
recall; ``triggerCorrect`` flags each serial position as correctly recalled.

Scientific constraint: human serial recall and the software workspace are
different systems. Human results describe behaviour; they are kept separate
from software metrics and are **not** a target to reproduce or a test of
biological equivalence. The comparison only asks which qualitative features
(load effects, primacy, recency) the current software shows on identical
sequences.

Privacy: participant identifiers are used only in memory to form the
participant-level split. They never appear in outputs, which are aggregates
across participants.
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artificial_human.cognition.benchmark.external import ExternalDatasetError, sha256_manifest
from artificial_human.cognition.benchmark.metrics import mean
from artificial_human.cognition.benchmark.working_memory_eval import ExistingWorkspace

COLUMNS = ("participant_id", "task", "condition", "trial", "sequence", "response",
           "NCorrect", "partialScore", "triggerCorrect")
FILE_PATTERN = "*_task-memory_beh.tsv"
TUNING_SHARE = 0.3


@dataclass(frozen=True, slots=True)
class Trial:
    participant: str  # used only for splitting, never reported
    load: int
    sequence: str
    correct: tuple[int, ...]
    missing_response: bool


def load_digit_span(directory: Path) -> tuple[list[Trial], dict[str, Any]]:
    """Read and validate local behavioural TSVs. Never downloads anything."""
    if not directory.is_dir():
        raise ExternalDatasetError(
            f"Digit-span directory not found: {directory}. Download the behavioural files "
            "manually (see docs/benchmarks/EXTERNAL_DATASETS.md)."
        )
    files = sorted(directory.glob(FILE_PATTERN))
    if not files:
        raise ExternalDatasetError(f"No {FILE_PATTERN} files in {directory}")
    trials: list[Trial] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise ExternalDatasetError(f"{path.name}: unexpected columns")
            for line, row in enumerate(reader, start=2):
                try:
                    load = int(row["condition"])
                    flags = tuple(int(flag) for flag in row["triggerCorrect"])
                    partial = int(row["partialScore"])
                except ValueError as error:
                    raise ExternalDatasetError(f"{path.name}:{line}: non-numeric field") from error
                sequence = row["sequence"]
                if row["task"] != "remember":
                    raise ExternalDatasetError(f"{path.name}:{line}: unexpected task")
                if not sequence.isdigit() or len(sequence) != load or len(flags) != load:
                    raise ExternalDatasetError(f"{path.name}:{line}: length mismatch")
                if any(flag not in (0, 1) for flag in flags) or sum(flags) != partial:
                    raise ExternalDatasetError(f"{path.name}:{line}: inconsistent correctness")
                trials.append(Trial(
                    participant=row["participant_id"], load=load, sequence=sequence,
                    correct=flags, missing_response=row["response"].strip() in ("", "n/a"),
                ))
    meta = {
        "files": len(files),
        "manifest_sha256": sha256_manifest(files),
        "participants": len({trial.participant for trial in trials}),
        "trials": len(trials),
        "loads": sorted({trial.load for trial in trials}),
        "missing_responses": sum(trial.missing_response for trial in trials),
    }
    return trials, meta


def participant_split(participant: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:digit-span:{participant}".encode()).digest()
    return "tuning" if int.from_bytes(digest[:8], "big") / 2**64 < TUNING_SHARE else "eval"


def _position_curves(trials: Sequence[Trial]) -> dict[int, list[float]]:
    """Mean correctness per serial position for each load (participant-weighted)."""
    by_participant: dict[str, dict[int, list[tuple[int, ...]]]] = {}
    for trial in trials:
        by_participant.setdefault(trial.participant, {}).setdefault(trial.load, []).append(
            trial.correct)
    curves: dict[int, list[float]] = {}
    for load in sorted({trial.load for trial in trials}):
        participant_curves = [
            [mean([float(flags[i]) for flags in loads[load]]) for i in range(load)]
            for loads in by_participant.values() if load in loads
        ]
        curves[load] = [mean([curve[i] for curve in participant_curves]) for i in range(load)]
    return curves


def _participant_accuracy(trials: Sequence[Trial]) -> dict[int, list[float]]:
    by_participant: dict[tuple[str, int], list[float]] = {}
    for trial in trials:
        by_participant.setdefault((trial.participant, trial.load), []).append(
            sum(trial.correct) / trial.load)
    result: dict[int, list[float]] = {}
    for (_, load), values in by_participant.items():
        result.setdefault(load, []).append(mean(values))
    return result


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _shape(curve: Sequence[float]) -> dict[str, float]:
    third = max(1, len(curve) // 3)
    return {
        "first_third": round(mean(curve[:third]), 4),
        "middle": round(mean(curve[third:len(curve) - third]) if len(curve) > 2 * third else 0.0, 4),
        "last_third": round(mean(curve[-third:]), 4),
    }


def software_curves(trials: Sequence[Trial], capacity: int = 7) -> dict[int, list[float]]:
    """Present each sequence to the real WorkingMemory (equal salience); probe every position."""
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for trial in trials:
        workspace = ExistingWorkspace(capacity)
        for position, digit in enumerate(trial.sequence):
            workspace.encode(f"position-{position}", digit, 0.6, position)
        contents = workspace.contents()
        flags = [
            1.0 if contents.get(f"position-{position}") == digit else 0.0
            for position, digit in enumerate(trial.sequence)
        ]
        totals = sums.setdefault(trial.load, [0.0] * trial.load)
        for index, flag in enumerate(flags):
            totals[index] += flag
        counts[trial.load] = counts.get(trial.load, 0) + 1
    return {load: [value / counts[load] for value in totals] for load, totals in sums.items()}


def _fifo_curve(load: int, capacity: int) -> list[float]:
    """Recency store: keeps the last ``capacity`` items."""
    return [1.0 if position >= load - capacity else 0.0 for position in range(load)]


def _primacy_curve(load: int, capacity: int) -> list[float]:
    """Primacy store: keeps the first ``capacity`` items."""
    return [1.0 if position < capacity else 0.0 for position in range(load)]


def _rmse(predicted: dict[int, list[float]], observed: dict[int, list[float]]) -> float:
    errors = [
        (p - o) ** 2
        for load in observed for p, o in zip(predicted[load], observed[load], strict=True)
    ]
    return math.sqrt(mean(errors))


def run_digit_span(directory: Path, *, seed: int) -> dict[str, Any]:
    trials, meta = load_digit_span(directory)
    split = {participant: participant_split(participant, seed)
             for participant in {trial.participant for trial in trials}}
    tuning = [trial for trial in trials if split[trial.participant] == "tuning"]
    evaluation = [trial for trial in trials if split[trial.participant] == "eval"]
    if not tuning or not evaluation:
        raise ExternalDatasetError("participant split produced an empty partition")
    loads = sorted({trial.load for trial in trials})

    human_eval = _position_curves(evaluation)
    human_tuning = _position_curves(tuning)
    accuracy = _participant_accuracy(evaluation)
    human = {
        str(load): {
            "participants": len(accuracy[load]),
            "mean_proportion_correct": round(mean(accuracy[load]), 4),
            "sd_across_participants": round(statistics.stdev(accuracy[load]), 4)
            if len(accuracy[load]) > 1 else None,
            "p10": round(_quantile(accuracy[load], 0.1), 4),
            "median": round(_quantile(accuracy[load], 0.5), 4),
            "p90": round(_quantile(accuracy[load], 0.9), 4),
            "serial_position_curve": [round(value, 4) for value in human_eval[load]],
            "curve_shape": _shape(human_eval[load]),
        }
        for load in loads
    }
    software = software_curves(evaluation)
    software_summary = {
        str(load): {
            "mean_proportion_correct": round(mean(software[load]), 4),
            "serial_position_curve": [round(value, 4) for value in software[load]],
            "curve_shape": _shape(software[load]),
        }
        for load in loads
    }
    # Reference models fitted on tuning participants only, scored on held-out participants.
    capacities = range(1, max(loads) + 1)

    def fit(curve: Any) -> int:
        errors = {capacity: _rmse({load: curve(load, capacity) for load in loads}, human_tuning)
                  for capacity in capacities}
        return min(errors, key=lambda capacity: (errors[capacity], capacity))

    best_capacity, best_primacy = fit(_fifo_curve), fit(_primacy_curve)
    flat = {load: [mean(human_tuning[load])] * load for load in loads}
    models = {
        "existing_working_memory_capacity_7": _rmse(software, human_eval),
        f"recency_store_capacity_{best_capacity}_fitted_on_tuning": _rmse(
            {load: _fifo_curve(load, best_capacity) for load in loads}, human_eval),
        f"primacy_store_capacity_{best_primacy}_fitted_on_tuning": _rmse(
            {load: _primacy_curve(load, best_primacy) for load in loads}, human_eval),
        "flat_per_load_mean_fitted_on_tuning": _rmse(flat, human_eval),
    }
    return {
        "dataset": meta | {
            "tuning_participants": sum(1 for value in split.values() if value == "tuning"),
            "eval_participants": sum(1 for value in split.values() if value == "eval"),
            "tuning_trials": len(tuning), "eval_trials": len(evaluation),
        },
        "human_reference_eval_participants": human,
        "software_on_identical_sequences": software_summary,
        "serial_position_rmse_vs_held_out_humans": {
            name: round(value, 4) for name, value in models.items()
        },
        "fitted_recency_capacity": best_capacity,
        "fitted_primacy_capacity": best_primacy,
        "note": (
            "Human serial recall and the software workspace are different systems. RMSE "
            "compares only serial-position curves on the same sequences; a low value would "
            "not imply a shared mechanism."
        ),
    }
