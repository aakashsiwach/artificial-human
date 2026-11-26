def schema_event_log_suite(
    path: Path, seed: int, attribute_keys: tuple[str, ...], max_cases: int,
) -> SuiteResult:
    from artificial_human.cognition.benchmark.external.schema_event_log import (
        run_schema_event_log_suite,
    )

    return run_schema_event_log_suite(
        path, seed=seed, attribute_keys=attribute_keys, max_cases=max_cases,
    )


"""Wrap external-dataset results as ``SuiteResult`` objects for the shared runner/report."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from artificial_human.cognition.benchmark.common import SuiteResult, check
from artificial_human.memory.embeddings import EmbeddingProvider


def longmemeval_suite(
    path: Path, provider: EmbeddingProvider | None, limit: int | None,
    notify: Callable[[str], None],
) -> SuiteResult:
    from artificial_human.cognition.benchmark.external.longmemeval import run_longmemeval

    outcome = run_longmemeval(path, provider=provider, limit=limit, progress=notify)
    summary = outcome["summary"]
    result = SuiteResult(
        suite="longmemeval",
        title="LongMemEval Oracle: evidence retrieval on independently authored data",
        evidence_category="computational capability benchmark (external, MIT, fully held out)",
        splits={"eval": summary["questions"]},
    )
    for variant, entry in summary["variants"].items():
        result.systems[variant] = entry["overall"] | {
            "abstention_rejection_rate": entry["abstention"]["rejection_rate"],
        }
        for category, values in entry["by_category"].items():
            result.by_condition.setdefault(category, {})[variant] = values
    result.checks = [
        check("no cross-owner leakage", summary["owner_leaks"] == 0, leaks=summary["owner_leaks"]),
        check("isolation probe owner sees nothing", summary["isolation_probe_results"] == 0),
    ]
    result.failures = outcome["missed_at_depth_20"]["examples"]
    result.extra = {
        "dataset": outcome["dataset"], "summary": summary,
        "primary_variant": outcome["primary_variant"],
        "failure_count": outcome["missed_at_depth_20"]["count"],
    }
    primary = summary["variants"][outcome["primary_variant"]]["overall"]
    result.findings = [
        f"{name}: turn recall@5 {entry['overall']['turn_recall@5']:.3f}, recall@20 "
        f"{entry['overall']['turn_recall@20']:.3f}, MRR {entry['overall']['turn_mrr']:.3f}; "
        f"abstention rejection rate {entry['abstention']['rejection_rate']}."
        for name, entry in summary["variants"].items()
    ] + [
        f"The primary variant ({outcome['primary_variant']}) found no evidence turn in its top "
        f"20 for {outcome['missed_at_depth_20']['count']} of {primary['questions']} scored "
        "questions.",
        "Session-level metrics are uninformative on the Oracle file: every history session is "
        "an evidence session, so session precision is 1 by construction.",
        "This measures evidence retrieval only, not answer accuracy.",
    ]
    return result


def digit_span_suite(directory: Path, seed: int) -> SuiteResult:
    from artificial_human.cognition.benchmark.external.digit_span import run_digit_span

    outcome = run_digit_span(directory, seed=seed)
    meta = outcome["dataset"]
    result = SuiteResult(
        suite="digit-span",
        title="Human digit-span reference (NEMAR on003838) vs software on identical sequences",
        evidence_category="human behavioural comparison (external, CC0); not biological fidelity",
        splits={"tuning": meta["tuning_participants"], "eval": meta["eval_participants"]},
    )
    result.systems = {
        name: {"serial_position_rmse_vs_held_out_humans": value}
        for name, value in outcome["serial_position_rmse_vs_held_out_humans"].items()
    }
    software = outcome["software_on_identical_sequences"]
    for load, values in outcome["human_reference_eval_participants"].items():
        result.by_condition[f"load_{load}"] = {
            "humans_held_out": {key: values[key] for key in (
                "participants", "mean_proportion_correct", "sd_across_participants",
                "p10", "median", "p90")} | values["curve_shape"],
            "software_existing_working_memory": {
                "mean_proportion_correct": software[load]["mean_proportion_correct"],
            } | software[load]["curve_shape"],
        }
    result.checks = [check(
        "participants do not appear in both splits",
        meta["tuning_participants"] + meta["eval_participants"] == meta["participants"],
    )]
    result.extra = outcome
    longest = str(max(int(load) for load in outcome["human_reference_eval_participants"]))
    human = outcome["human_reference_eval_participants"][longest]["curve_shape"]
    soft = software[longest]["curve_shape"]
    result.findings = [
        f"At load {longest}, held-out humans show a primacy gradient (first third "
        f"{human['first_third']:.3f}, last third {human['last_third']:.3f}). The software "
        f"workspace shows the reverse (first third {soft['first_third']:.3f}, last third "
        f"{soft['last_third']:.3f}) because it keeps the most recent items.",
        "Among reference models fitted on tuning participants only, a primacy store (capacity "
        f"{outcome['fitted_primacy_capacity']}) describes held-out serial-position curves better "
        f"than a recency store (best capacity {outcome['fitted_recency_capacity']}). This "
        "describes behaviour; it is not a claim about mechanism.",
        outcome["note"],
    ]
    return result


def schema_event_log_suite(
    path: Path, seed: int, attribute_keys: tuple[str, ...], max_cases: int,
) -> SuiteResult:
    from artificial_human.cognition.benchmark.external.schema_event_log import (
        run_schema_event_log_suite,
    )

    return run_schema_event_log_suite(
        path, seed=seed, attribute_keys=attribute_keys, max_cases=max_cases,
    )
