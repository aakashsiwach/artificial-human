"""Markdown rendering of a cognitive evaluation report.

Static text explains purpose, background, definitions and limitations. Every
number is taken from the report produced by ``runner.run_cognitive_evaluation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PURPOSE = """\
This report measures the cognitive subsystems that exist on `main` today:
attention, working memory, the lexical prediction/novelty signal, long-term
memory recall, controlled associative cue completion, context-scoped source monitoring,
temporal episodic replay, and the integrated multi-turn cycle. It establishes a
reproducible baseline *before* new mechanisms are introduced, as Stage 1 of the
[Research Charter](../RESEARCH_CHARTER.md) requires. No algorithm was changed to
improve these numbers.

Four kinds of evidence are kept separate:

1. **Software correctness tests** (`pytest`) show that the code does what it
   specifies.
2. **Computational capability benchmarks** (this report) measure behaviour on
   defined, synthetic tasks against baselines.
3. **Human behavioural comparisons** are qualitative here. The report notes where
   software behaviour resembles or departs from well-replicated human findings,
   but no human data were collected or fitted.
4. **Biological fidelity** is **not** assessed. Nothing in this report shows that a
   component works like a brain region or circuit."""

_BACKGROUND = """\
The full research record (phenomena, hypotheses, competing accounts, and what
each benchmark cannot establish) is in [docs/COGNITIVE_EVALUATION.md](../COGNITIVE_EVALUATION.md).
In brief:

- **Attention.** Selection among competing representations is shaped by
  goal-relevance and stimulus salience (biased competition: Desimone & Duncan,
  1995; saliency models: Itti & Koch, 2001). Salient distractors can capture
  selection against current goals (Theeuwes, 2010), and the top-down/bottom-up
  dichotomy is itself debated (Awh, Belopolsky & Theeuwes, 2012). The software
  combines two supplied numbers linearly; it does not model competition dynamics.
- **Working memory.** Capacity is limited (Miller, 1956; about four chunks in
  Cowan, 2001). Forgetting in short-term memory is better explained by
  interference than by time-based decay (Lewandowsky, Oberauer & Brown, 2009;
  contrast Berman, Jonides & Lewis, 2009). Free recall shows primacy and recency
  (Murdock, 1962). Oberauer et al. (2018) list benchmark findings that models
  should explain. The software is a bounded store with salience decay.
- **Prediction and novelty.** Predictive-processing accounts (Rao & Ballard,
  1999; Friston, 2005) have mixed empirical support (Hodson, Mehta & Smith, 2024).
  Neural responses differ for semantic incongruity (Kutas & Hillyard, 1980), and
  hippocampal novelty/mismatch signals are an active research area (Kumaran &
  Maguire, 2007). The software computes token-set overlap only.
- **Long-term memory.** The memory benchmark addresses episodic and semantic
  recall (Tulving, 1972), temporal organisation (Howard & Kahana, 2002) and source
  monitoring (Johnson, Hashtroudi & Lindsay, 1993). See
  [docs/MEMORY.md](../MEMORY.md)."""

_DEFINITIONS = """\
| Suite | Tasks | Ground truth |
| --- | --- | --- |
| Attention | Seeded candidate sets (4–64 candidates) in five conditions: clean relevance; informative salience; salient distractors; tied relevance; cross-source normalisation (memories and learning examples max-normalised separately, using the architecture's own function) | Generator-assigned usefulness grades 0/1/2; the system sees only noisy relevance and salience |
| Working memory | Delayed retention (3 targets, 0–12 distractors, capacities 3/5/7/9, distractor salience lower/equal/higher); similar vs dissimilar distractors; reactivation with updated content; seeded capacity pressure; serial position; serial order; newcomer admission; session isolation | Whether each encoded item remains in the workspace with the expected content |
| Prediction | 54 authored context/input pairs in 9 categories: exact repetition, near repetition, paraphrase, low-overlap related, unrelated, shared vocabulary with a different sense, contradiction, contextually expected, contextually unexpected | Author label `unexpected` (unrelated, other word sense, contradiction, script violation) vs expected |
| Memory | Existing synthetic memory benchmark (46 memories, two synthetic owners) plus source-attribution queries | Graded relevance per query, expected source, empty set for irrelevant queries |
| Associative memory | Nine fixed source-fact pairs; each pair is co-activated during training, then tested with a cue that directly retrieves only its source fact | Whether the declared target fact is retrieved in the top 5 |
| Associative context | Nine source-fact pairs trained in separate contexts, then queried in the same context, a different context, and a shared global graph | Same-context target recall@5 and cross-context leakage@5 |
| Associative dynamics | Nine paired contradiction and distractor-interference probes over opaque record IDs | Contradiction suppression rate and target recall after a capacity-limited distractor sequence |
| Episodic replay | Nine four-event sequences; cue each middle event and retrieve its immediate temporal neighbours | Exact neighbour recall@2, MRR, false replay rate, cross-context leakage |
| Integration | Six multi-turn scenarios through `CognitiveArchitecture` (A–F) | Scenario-specific behavioural checks |"""

_BASELINES = """\
- **Attention.** `existing_attention` is the unmodified `AttentionSystem.select`
  (0.8 × relevance + 0.2 × salience, ties by id). Compared with relevance-only,
  salience-only, and input-order (deterministic) ranking. Ablations re-apply the
  formula with weights 0.6/0.4 and 0.5/0.5.
- **Working memory.** `existing_working_memory` is the unmodified `WorkingMemory`
  (decay × 0.92 per insertion, evict lowest salience after insertion).
  Compared with FIFO, LRU, and salience eviction without decay (ablation of decay).
- **Prediction.** `existing_prediction` is the unmodified `PredictionSystem`.
  Compared with content-word overlap (stop words removed, light stemming) and
  token Jaccard distance. With a local embedding model, an
  `embedding_distance_reference` (1 − maximum cosine to the context) is
  reported as a non-lexical reference, not as a proposed component.
- **Memory.** Lexical, semantic, hybrid and hybrid + temporal retrieval, plus
  ablations, as defined by the memory benchmark.
- **Associative memory.** The learned co-activation graph is compared with direct
  keyword retrieval and a random undirected graph with the same edge budget and
  mean link strength. The random graph is sampled uniformly over source-fact pairs.
- **Episodic replay.** Adjacent-event cue completion is compared with a no-replay
  baseline and a wrong owner/session context control. The test uses fixed
  project-authored event sequences and opaque identifiers."""

_METRICS = """\
- **Precision@K / Recall@K / MRR / nDCG@K.** Standard ranked-retrieval
  definitions, shared with the memory benchmark (`memory/benchmark/metrics.py`).
  nDCG uses exponential gain 2^grade − 1.
- **Pairwise ranking accuracy.** Share of candidate pairs with different grades
  that are ordered correctly in the full ranking.
- **Selection consistency.** Identical output on repeated runs, and an
  unchanged top-k set when the input order is shuffled.
- **Retention rate.** Share of probed items still present in the workspace.
  **Recall accuracy** also requires the expected (possibly updated) content.
- **Capacity violations.** Number of steps at which the workspace held more items
  than its capacity.
- **AUROC.** Probability that a randomly chosen unexpected input receives a
  higher novelty score than a randomly chosen expected input (ties count ½).
- **False / missed novelty rate.** False-positive and false-negative rates at the
  threshold that maximised balanced accuracy on the **tuning** split.
- **Novelty separation.** Mean novelty for unexpected inputs minus mean
  novelty for expected inputs.
- **Temporal answer accuracy / source attribution accuracy / contradiction
  recall.** Top result is a best-graded answer / top result comes from the
  expected source / all conflicting memories appear in the top 5.
- **Associative target recall@5 / MRR.** Share of held-out cues that retrieve the
  declared target in the top 5 / mean reciprocal rank of that target.
- **Cross-context leakage@5.** Share of cues that retrieve a target learned in a
  different context; lower is better.
- **Contradiction suppression rate.** Share of explicit conflict probes in which
  a previously learned positive edge is deleted and remains unavailable for recall.
- **Post-interference target recall@1.** Share of cue probes that recover a target
  relation after a fixed distractor sequence under a one-edge budget.
- **Episodic neighbour recall@2 / replay MRR.** Share of cues that recover both
  immediate temporal neighbours in the top two / mean reciprocal rank of expected
  neighbours.
- **False replays per query.** Returned records that are not adjacent to a cue in
  the stored event sequence.
- **Cross-context replay leakage@2.** Share of cues that recover an event from a
  sequence stored under another owner/session context; lower is better.
- **False associations per query.** Mean count of top-5 facts that are neither
  the cue source nor the declared target."""

_LIMITATIONS = """\
- All datasets are synthetic, English-only and written by this project. They test
  defined behaviours of the software, not human-like cognition.
- Attention ground truth comes from the same generator that produces the
  observable scores, so results describe the formula's behaviour under stated
  input distributions. Overall averages depend on the chosen condition mix.
- Working-memory probes check presence in a software store. Humans recall by
  content and show similarity-based interference; this software does neither.
- Prediction labels are author judgements on 54 short items (18 tuning, 36 eval).
  Confidence intervals would be wide; small differences are not meaningful.
- The integration scenarios are few and hand-designed. A pass shows that one
  specified behaviour occurred, not that the behaviour generalises.
- Associative-memory probes use nine project-authored pairs from a small
  source-defined corpus. They test controlled cue completion, not naturalistic
  episodic recall, source monitoring, or human memory. The random-link control
  is a graph-density control, not a fitted biological baseline.
- No externally authored dataset is included. Candidates were reviewed but not
  adopted, because of non-commercial licence terms, real user conversations
  among their sources, or mixed per-source licences (see
  [docs/COGNITIVE_EVALUATION.md](../COGNITIVE_EVALUATION.md#external-datasets)).
- Runtime figures are single-run wall-clock times on the generating machine.
  Peak memory is Python allocations traced by `tracemalloc`."""

_QUESTIONS = """\
1. Should attention normalise candidates from different sources on a common
   scale, and how should salience be gated so salient distractors cannot win?
2. Should working-memory forgetting be driven by interference (content
   similarity, overlap of features) rather than multiplicative decay, and can
   that reproduce primacy as well as recency?
3. Should retrieval use the active working-memory context (conversational state)
   so elliptical follow-ups can be resolved, and how would that be evaluated?
4. Should repeated inputs reactivate existing workspace entries (content-based
   identity) instead of creating new percepts?
5. What representation would let the prediction signal detect contradictions
   and script violations? A semantic distance alone does not (see the reference
   results), so this points towards explicit propositional or world-state models.
6. Which externally authored, permissively licensed evaluation material could
   replace or supplement the project-authored datasets?
7. These baselines are the comparison point for Stage 2 (associative memory and
   memory dynamics). Each new mechanism must improve a named metric here without
   regressing others."""


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(rows: Mapping[str, Mapping[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| System | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---" for _ in columns) + " |",
    ]
    for name, values in rows.items():
        lines.append(
            f"| {name} | " + " | ".join(_fmt(values.get(column)) for column in columns) + " |"
        )
    return lines


def _attention_generalisation(data: Mapping[str, Any] | None) -> list[str]:
    if not data:
        return []
    lines = [
        "**Generalisation to held-out conditions**", "",
        f"Weight tuned on tuning-split tasks of {', '.join(data['train_conditions'])} "
        f"({data['tuning_tasks']} tasks): relevance weight **{data['tuned_relevance_weight']}**. "
        f"Tested on eval-split tasks of the unseen conditions {', '.join(data['held_out_conditions'])} "
        f"({data['held_out_eval_tasks']} tasks).", "",
        "| System | In-distribution nDCG@k | Held-out nDCG@k | Gap | Salient distractors selected / task "
        "| Weak learning examples selected / task | Highly relevant missed / task |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in data["systems"].items():
        failures = values["held_out_failure_categories_per_task"]
        lines.append(
            f"| {name} | {_fmt(values['in_distribution_ndcg@k'])} | {_fmt(values['held_out_ndcg@k'])} "
            f"| {_fmt(values['generalisation_gap'])} | {_fmt(failures['salient_distractor_selected'])} "
            f"| {_fmt(failures['weak_learning_example_selected'])} "
            f"| {_fmt(failures['highly_relevant_missed'])} |"
        )
    return lines + [""]


def _wm_generalisation(data: Mapping[str, Any] | None) -> list[str]:
    if not data:
        return []
    lines = [
        f"**Generalisation grid**: {data['cells']} cells × {data['repetitions_per_cell']} seeded "
        "repetitions (capacity × intervening items × distractor-salience distribution × event "
        "ordering). Mean target retention per factor level:", "",
    ]
    for factor, levels in data["factors"].items():
        lines += [f"*{factor.replace('_', ' ')}*", "",
                  "| Policy | " + " | ".join(levels) + " |",
                  "| --- | " + " | ".join("---" for _ in levels) + " |"]
        for policy, by_factor in data["mean_retention_by_factor"].items():
            lines.append(f"| {policy} | " + " | ".join(
                _fmt(by_factor[factor][level]) for level in levels) + " |")
        lines.append("")
    return lines + [
        f"In {data['cells_where_a_baseline_retains_more']} of {data['cells']} cells at least one "
        "baseline retains more targets than the existing policy.", "",
    ]


def _calibration(data: Mapping[str, Any] | None, note: str | None) -> list[str]:
    if not data:
        return []
    lines = [
        "**Score calibration** (a logistic mapping is fitted on the tuning split, frozen, then "
        "applied to the eval split)", "",
        "| System | Mapping | Brier | ECE | Cal. intercept | Cal. slope | AUROC | False novelty @0.5 "
        "| Missed novelty @0.5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in data.items():
        for label, key in (("uncalibrated", "uncalibrated_score_as_probability"),
                           ("logistic", "logistic_calibrated")):
            if key not in values:
                continue
            item = values[key]
            lines.append(
                f"| {name} | {label} | {_fmt(item['brier_score'])} | "
                f"{_fmt(item['expected_calibration_error'])} | {_fmt(item['calibration_intercept'])} "
                f"| {_fmt(item['calibration_slope'])} | {_fmt(item['auroc'])} "
                f"| {_fmt(item['false_novelty_rate_at_0.5'])} | {_fmt(item['missed_novelty_rate_at_0.5'])} |"
            )
        base = values["base_rate_reference"]
        lines.append(f"| {name} | base rate {base['probability']:.3f} | {_fmt(base['brier_score'])} "
                     "| n/a | n/a | n/a | 0.500 | n/a | n/a |")
    existing = data.get("existing_prediction", {})
    bins = existing.get("logistic_calibrated", {}).get("reliability_bins", [])
    if bins:
        lines += ["", "Reliability bins (existing, logistic-calibrated): " + "; ".join(
            f"[{item['lower']:.1f}, {item['upper']:.1f}) n={item['count']}"
            + (f", predicted {item['mean_predicted']:.2f}, observed {item['observed_rate']:.2f}"
               if item["count"] else "")
            for item in bins)]
    return lines + (["", note] if note else []) + [""]


def _external_section(suite: Mapping[str, Any]) -> list[str]:
    lines = [f"### {suite['title']}", "", f"*Evidence category: {suite['evidence_category']}.*", ""]
    extra = suite["extra"]
    if suite["suite"] == "longmemeval":
        meta = extra["dataset"]
        lines += [
            f"Source file `{meta['file']}` (sha256 `{meta['sha256']}`, {meta['bytes']} bytes); "
            f"{meta['questions_evaluated']} of {meta['questions_in_file']} questions evaluated; "
            f"{meta['sessions']} sessions, {meta['turns']} turns. All questions are held out; "
            "nothing was tuned on them.", "",
            "**Turn-level evidence retrieval** (non-abstention questions with labelled turns)", "",
        ]
        summary = extra["summary"]
        columns = ["questions", "turn_recall@1", "turn_recall@5", "turn_recall@10",
                   "turn_recall@20", "turn_precision@5", "turn_ndcg@10", "turn_mrr"]
        lines += _table({name: entry["overall"] for name, entry in summary["variants"].items()},
                        columns) + [""]
        for grouping in ("by_category", "by_history_sessions", "by_history_turns",
                         "by_evidence_position", "by_evidence_sessions",
                         "by_days_since_latest_evidence"):
            lines += [f"**Turn MRR / recall@10 {grouping.replace('_', ' ')}**", ""]
            variants = list(summary["variants"])
            groups = sorted({key for entry in summary["variants"].values() for key in entry[grouping]})
            lines += ["| Group | n | " + " | ".join(variants) + " |",
                      "| --- | --- | " + " | ".join("---" for _ in variants) + " |"]
            for group in groups:
                first = summary["variants"][variants[0]][grouping].get(group, {})
                cells = []
                for variant in variants:
                    values = summary["variants"][variant][grouping].get(group, {})
                    cells.append(f"{_fmt(values.get('turn_mrr'))} / {_fmt(values.get('turn_recall@10'))}")
                lines.append(f"| {group} | {first.get('questions', 0)} | " + " | ".join(cells) + " |")
            lines.append("")
        depth = summary["variants"][extra["primary_variant"]]["turn_recall_by_depth"]
        lines += [f"**Turn recall by retrieval depth ({extra['primary_variant']})**: " + ", ".join(
            f"@{k} {_fmt(v)}" for k, v in depth.items() if int(k) in (1, 2, 3, 5, 10, 15, 20)), ""]
        lines += ["**Session level and abstention**", "",
                  "| Variant | Session recall@1 | Session recall@5 | Session precision@1 | Abstention questions | Rejection rate | Mean results returned |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, entry in summary["variants"].items():
            session, abstention = entry["session_level"], entry["abstention"]
            lines.append(f"| {name} | {_fmt(session.get('session_recall@1'))} | "
                         f"{_fmt(session.get('session_recall@5'))} | {_fmt(session.get('session_precision@1'))} "
                         f"| {abstention['questions']} | {_fmt(abstention['rejection_rate'])} "
                         f"| {_fmt(abstention['mean_results_returned'])} |")
        lines += ["", f"Owner leaks: {summary['owner_leaks']}; isolation-probe results: "
                  f"{summary['isolation_probe_results']}; questions without turn labels: "
                  f"{summary['questions_without_turn_labels']}.", ""]
    elif suite["suite"] == "digit-span":
        meta = extra["dataset"]
        lines += [
            f"{meta['files']} behavioural files (manifest sha256 `{meta['manifest_sha256']}`), "
            f"{meta['participants']} participants, {meta['trials']} trials, loads {meta['loads']}, "
            f"{meta['missing_responses']} empty responses. Participant-level split: "
            f"{meta['tuning_participants']} tuning / {meta['eval_participants']} eval participants. "
            "Only aggregates are reported.", "",
            "| Load | Humans: mean (SD) | Humans p10 / median / p90 | Humans first / middle / last third "
            "| Software: mean | Software first / middle / last third |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        humans = extra["human_reference_eval_participants"]
        software = extra["software_on_identical_sequences"]
        for load, values in humans.items():
            shape, soft = values["curve_shape"], software[load]
            lines.append(
                f"| {load} | {_fmt(values['mean_proportion_correct'])} ({_fmt(values['sd_across_participants'])}) "
                f"| {_fmt(values['p10'])} / {_fmt(values['median'])} / {_fmt(values['p90'])} "
                f"| {_fmt(shape['first_third'])} / {_fmt(shape['middle'])} / {_fmt(shape['last_third'])} "
                f"| {_fmt(soft['mean_proportion_correct'])} | {_fmt(soft['curve_shape']['first_third'])} / "
                f"{_fmt(soft['curve_shape']['middle'])} / {_fmt(soft['curve_shape']['last_third'])} |"
            )
        lines += ["", "**Held-out human serial-position curves** (mean proportion correct by position)", ""]
        for load, values in humans.items():
            lines.append(f"- Load {load}: humans " + ", ".join(_fmt(v) for v in values["serial_position_curve"])
                         + "; software " + ", ".join(_fmt(v) for v in software[load]["serial_position_curve"]))
        lines += ["", "**Serial-position RMSE against held-out participants** (lower = closer)", ""]
        lines += _table({name: {"rmse": value} for name, value in
                         extra["serial_position_rmse_vs_held_out_humans"].items()}, ["rmse"]) + [""]
    if suite["checks"]:
        lines += ["**Checks**", ""] + [
            f"- {'PASS' if item['passed'] else 'FAIL'}: {item['name']}" for item in suite["checks"]
        ] + [""]
    if suite["findings"]:
        lines += ["**Findings**", ""] + [f"- {finding}" for finding in suite["findings"]] + [""]
    return lines


def _suite_section(suite: Mapping[str, Any]) -> list[str]:
    lines = [f"### {suite['title']}", "", f"*Evidence category: {suite['evidence_category']}.*", ""]
    if suite["systems"]:
        columns = sorted({key for values in suite["systems"].values() for key in values})
        lines += _table(suite["systems"], columns) + [""]
    name = suite["suite"]
    if name == "attention":
        lines += ["**nDCG@k by condition**", ""]
        systems = list(suite["systems"])
        lines += ["| Condition | " + " | ".join(systems) + " |",
                  "| --- | " + " | ".join("---" for _ in systems) + " |"]
        for condition, values in suite["by_condition"].items():
            lines.append(f"| {condition} | " + " | ".join(
                _fmt(values[system]["ndcg@k"]) for system in systems) + " |")
        lines += ["", "**nDCG@k by candidate-set size**", ""]
        sizes = list(next(iter(suite["extra"]["ndcg_by_set_size"].values())))
        lines += ["| System | " + " | ".join(sizes) + " |",
                  "| --- | " + " | ".join("---" for _ in sizes) + " |"]
        for system, values in suite["extra"]["ndcg_by_set_size"].items():
            lines.append(f"| {system} | " + " | ".join(_fmt(values[s]) for s in sizes) + " |")
        consistency = suite["extra"]["consistency"]
        lines += ["", f"Consistency: repeat-run identical {_fmt(consistency['repeat_run_identical'])}, "
                  f"top-k invariant to input order {_fmt(consistency['topk_permutation_invariance'])}.", ""]
        lines += _attention_generalisation(suite["extra"].get("generalisation"))
    elif name == "working-memory":
        curves = suite["extra"]["delayed_retention_curves"]
        delays = list(next(iter(curves.values()))["retention_by_intervening_items"])
        lines += ["**Retention vs intervening items** (lower-salience distractors, mean over capacities)", "",
                  "| Policy | " + " | ".join(delays) + " |",
                  "| --- | " + " | ".join("---" for _ in delays) + " |"]
        for policy, values in curves.items():
            lines.append(f"| {policy} | " + " | ".join(
                _fmt(values["retention_by_intervening_items"][d]) for d in delays) + " |")
        capacities = list(next(iter(curves.values()))["retention_by_capacity"])
        lines += ["", "**Retention vs workspace capacity** (lower-salience distractors, mean over 0–12 distractors)", "",
                  "| Policy | " + " | ".join(capacities) + " |",
                  "| --- | " + " | ".join("---" for _ in capacities) + " |"]
        for policy, values in curves.items():
            lines.append(f"| {policy} | " + " | ".join(
                _fmt(values["retention_by_capacity"][c]) for c in capacities) + " |")
        for condition in ("reactivation", "capacity_pressure", "serial_order",
                          "similarity_interference", "newcomer_admission_full_capacity_7"):
            rows = suite["by_condition"][condition]
            columns = list(next(iter(rows.values())))
            lines += ["", f"**{condition.replace('_', ' ')}**", ""] + _table(rows, columns)
        positions = suite["extra"]["serial_position_retention_capacity_7"]
        order = list(next(iter(positions.values())))
        lines += ["", "**Serial position** (12 equal-salience items into capacity 7; 1 = retained)", "",
                  "| Policy | " + " | ".join(order) + " |",
                  "| --- | " + " | ".join("---" for _ in order) + " |"]
        for policy, values in positions.items():
            lines.append(f"| {policy} | " + " | ".join(_fmt(values[p]) for p in order) + " |")
        lines.append("")
        lines += _wm_generalisation(suite["extra"].get("generalisation_grid"))
    elif name == "prediction":
        systems = list(suite["systems"])
        lines += ["**Mean novelty by category**", "",
                  "| Category | Label | " + " | ".join(systems) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in systems) + " |"]
        for category, values in suite["by_condition"].items():
            label = next(iter(values.values()))["label"]
            lines.append(f"| {category} | {label} | " + " | ".join(
                _fmt(values[s]["mean_novelty"]) for s in systems) + " |")
        extra = suite["extra"]
        lines += ["", f"Case/punctuation invariance: {_fmt(extra['case_and_punctuation_invariance'])}. "
                  f"Empty context comparable: {_fmt(extra['empty_context_comparable'])} "
                  f"(novelty {_fmt(extra['empty_context_novelty'])}).", ""]
        lines += _calibration(extra.get("calibration"), extra.get("calibration_note"))
    elif name == "memory":
        extra = suite["extra"]
        model = extra["embedding_model"]
        lines += [
            f"Dataset `{extra['dataset']['name']}` v{extra['dataset']['version']} "
            f"(sha256 `{extra['dataset']['sha256'][:12]}`); embedding model: "
            + (f"`{model['model_id']}`" if model else "none (lexical variants only)")
            + f"; semantic threshold {_fmt(extra['semantic_threshold'])}"
            + (" (tuned on the tuning split)" if extra["threshold_tuned_on"] else "") + ".",
            "",
        ]
        if extra["skipped_variants"]:
            lines += ["Skipped variants: " + ", ".join(extra["skipped_variants"]) + ".", ""]
        systems = list(suite["systems"])
        lines += ["**MRR by category** (rej = rejection rate)", "",
                  "| Category | " + " | ".join(systems) + " |",
                  "| --- | " + " | ".join("---" for _ in systems) + " |"]
        for category, values in suite["by_condition"].items():
            cells = []
            for system in systems:
                entry = values.get(system, {})
                cells.append(_fmt(entry.get("mrr")) if entry.get("mrr") is not None
                             else f"rej {_fmt(entry.get('rejection_rate'))}")
            lines.append(f"| {category} | " + " | ".join(cells) + " |")
        lines.append("")
    elif name == "associative-memory":
        graph = suite["extra"]["training_graph"]
        control = suite["extra"]["random_link_control"]
        lines += [
            f"Training graph: {graph['edge_count']} edges across {graph['source_fact_count']} "
            f"source facts; mean link strength {_fmt(graph['mean_link_strength'])}.",
            f"Random-link control: {control['repetitions']} "
            f"seeded graphs with the same edge budget and mean link strength.",
            "",
            "| Probe | System | Target rank | Target recall@5 | Target MRR | False associations |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for probe, results in suite["by_condition"].items():
            for system, values in results.items():
                lines.append(
                    f"| {probe} | {system} | {_fmt(values.get('target_rank'))} "
                    f"| {_fmt(values.get('target_recall@5'))} "
                    f"| {_fmt(values.get('target_mrr'))} "
                    f"| {_fmt(values.get('false_associations'))} |"
                )
        lines.append("")
    elif name == "associative-context":
        lines += [
            f"Contexts: {suite['extra']['context_count']} trained; "
            f"{suite['extra']['contexts_removed']} removed during cleanup.",
            "",
            "| Probe | Same-context recall@5 | Different-context leakage@5 | Shared-global recall@5 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for probe, systems in suite["by_condition"].items():
            same = systems["same_context"]
            cross = systems["different_context"]
            shared = systems["shared_global_control"]
            lines.append(
                f"| {probe} | {_fmt(same['target_recall@5'])} "
                f"| {_fmt(cross['target_recall@5'])} "
                f"| {_fmt(shared['target_recall@5'])} |"
            )
        lines.append("")
    elif name == "episodic-replay":
        systems = suite["systems"]
        lines += [
            "| Probe | Expected neighbours | Recalled neighbours | Exact recall@2 | MRR | False replays |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for probe, values in suite["by_condition"].items():
            replay = values["temporal_replay"]
            lines.append(
                f"| {probe} | {', '.join(replay['expected_neighbours'])} "
                f"| {', '.join(replay['recalled_neighbours'])} "
                f"| {_fmt(replay['exact_neighbour_recall@2'])} "
                f"| {_fmt(replay['mean_reciprocal_rank'])} "
                f"| {_fmt(replay['false_replays'])} |"
            )
        lines += [
            "",
            f"No-replay baseline neighbour recall@2: "
            f"{_fmt(systems['no_replay_baseline']['exact_neighbour_recall@2'])}; "
            f"wrong-context leakage@2: "
            f"{_fmt(systems['wrong_context_control']['cross_context_leakage@2'])}.",
            "",
        ]
    elif name == "associative-dynamics":
        systems = suite["systems"]
        contradiction = systems["contradiction_suppression"]
        lines += [
            f"Explicit contradiction suppression rate: "
            f"{_fmt(contradiction['suppression_rate'])}; "
            f"contradiction recall@1: {_fmt(contradiction['contradiction_recall@1'])}.",
            "",
            "| Policy | Target recall after distractors@1 | Mean evicted links |",
            "| --- | ---: | ---: |",
        ]
        for policy, values in suite["systems"].items():
            if policy == "contradiction_suppression":
                continue
            lines.append(
                f"| {policy} | {_fmt(values['target_recall_after_distractors@1'])} "
                f"| {_fmt(values['mean_evicted_links'])} |"
            )
        lines.append("")
    if suite["checks"] and name != "integration":
        lines += ["**Checks**", ""] + [
            f"- {'PASS' if item['passed'] else 'FAIL'}: {item['name']}" for item in suite["checks"]
        ] + [""]
    if suite["findings"]:
        lines += ["**Findings**", ""] + [f"- {finding}" for finding in suite["findings"]] + [""]
    return lines


def _integration_section(suite: Mapping[str, Any]) -> list[str]:
    configs = list(suite["systems"])
    lines = [
        "| Scenario | " + " | ".join(configs) + " |",
        "| --- | " + " | ".join("---" for _ in configs) + " |",
    ]
    for scenario, values in suite["by_condition"].items():
        lines.append(f"| {scenario} — {next(iter(values.values()))['description']} | " + " | ".join(
            f"{values[c]['passed']}/{values[c]['checks']}" for c in configs) + " |")
    lines += ["", "| Scenario | Check | " + " | ".join(configs) + " |",
              "| --- | --- | " + " | ".join("---" for _ in configs) + " |"]
    keyed: dict[tuple[str, str], dict[str, bool]] = {}
    for item in suite["checks"]:
        keyed.setdefault((item["scenario"], item["name"]), {})[item["config"]] = item["passed"]
    for (scenario, name), outcomes in keyed.items():
        lines.append(f"| {scenario[:1]} | {name} | " + " | ".join(
            "pass" if outcomes.get(c) else "FAIL" for c in configs) + " |")
    lines += ["", "**Findings**", ""] + [f"- {finding}" for finding in suite["findings"]] + [""]
    return lines


def _failures(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for name, suite in report["suites"].items():
        failures = suite["failures"]
        if not failures:
            continue
        total = suite["extra"].get("failure_count", len(failures))
        shown = min(len(failures), 10)
        lines += [f"**{name}** (showing {shown} of {total})", ""]
        for item in failures[:10]:
            lines.append("- " + "; ".join(f"{key}: {value}" for key, value in item.items()))
        lines.append("")
    return lines or ["No failure cases were recorded.", ""]


def render_markdown(report: Mapping[str, Any]) -> str:
    suites: Mapping[str, Any] = report["suites"]
    model = report["embedding_model"]
    lines = [
        "# Cognitive evaluation report",
        "",
        f"- Generated: {report['generated_at']} · seed `{report['seed']}` · Python "
        f"{report['environment']['python']} ({report['environment']['platform']})",
        "- Embedding backend: " + (
            f"`{model['model_id']}` ({model['backend']}, {model['locality']})" if model
            else "none. Runs fully offline; semantic variants are skipped."
        ),
        f"- Suites: {', '.join(report['suites_run'])}",
        "",
        "## 1. Evaluation purpose", "", _PURPOSE, "",
        "## 2. Scientific background", "", _BACKGROUND, "",
        "## 3. Task and dataset definitions", "", _DEFINITIONS, "",
        "## 4. Evaluation splits", "",
        "| Suite | Tuning | Eval | What the tuning split is used for |",
        "| --- | --- | --- | --- |",
    ]
    uses = {
        "attention": "reserved; nothing is tuned",
        "working-memory": "reserved (capacity-pressure runs); nothing is tuned",
        "prediction": "novelty threshold per system (balanced accuracy)",
        "memory": "semantic similarity threshold",
        "associative-memory": "none (fixed source-fact probes; no tuning)",
        "associative-context": "none (fixed source-fact/context probes; no tuning)",
        "associative-dynamics": "none (fixed contradiction/interference probes; no tuning)",
        "episodic-replay": "none (fixed temporal-sequence probes; no tuning)",
        "integration": "none (fixed scenarios)",
        "longmemeval": "none: every external question is held out",
        "digit-span": "fitting human reference models only (participant-level split)",
    }
    for name, suite in suites.items():
        lines.append(f"| {name} | {suite['splits'].get('tuning', 0)} | "
                     f"{suite['splits'].get('eval', 0)} | {uses[name]} |")
    lines += [
        "", "Generated tasks use RNG streams derived from (seed, suite, split, condition), "
        "so tuning and eval tasks never share a stream. Reported metrics use the eval split only.",
        "",
        "## 5. Baselines", "", _BASELINES, "",
        "## 6. Metric definitions", "", _METRICS, "",
        "## 7. Results by subsystem", "",
    ]
    for name in ("attention", "working-memory", "prediction", "memory",
                 "associative-memory", "associative-context", "associative-dynamics",
                 "episodic-replay"):
        if name in suites:
            lines += _suite_section(suites[name])
    lines += ["## 8. Integrated multi-turn results", ""]
    lines += _integration_section(suites["integration"]) if "integration" in suites else [
        "Not run.", ""]
    lines += ["## 9. Failure cases", ""] + _failures(report)
    lines += ["## 10. Ablations", ""]
    if "attention" in suites:
        lines += ["**Attention weight sensitivity** (eval split, all conditions)", ""]
        lines += _table(suites["attention"]["extra"]["ablations"],
                        ["ndcg@k", "precision@k", "recall@k", "mrr", "pairwise_accuracy"]) + [""]
    if "working-memory" in suites:
        lines += ["Working memory: `salience_no_decay` removes the decay term; its rows "
                  "appear in every working-memory table above.", ""]
    if "memory" in suites:
        lines += ["Memory: the `ablation_*` rows in the memory table remove the lexical or "
                  "semantic channel, the similarity threshold, or contradiction expansion.", ""]
    lines += [
        "## 11. Runtime and resource use", "",
        "| Suite | Wall-clock (s) | Peak traced Python memory (KiB) |",
        "| --- | --- | --- |",
    ] + [
        f"| {name} | {suite['runtime_seconds']:.3f} | {suite['peak_memory_kib']:.0f} |"
        for name, suite in suites.items()
    ] + [
        "", f"Total: {report['total_runtime_seconds']:.3f} s. "
        f"Network used by the evaluation: {_fmt(report['network_used_by_evaluation'])}.", "",
        "## 12. Reproducibility", "",
        "```",
        "python -m pip install -e \".[dev]\"",
        f"artificial-human cognition-eval --suite all --seed {report['seed']} --output-dir docs/benchmarks",
        "# optional local semantic variants (downloads model weights once):",
        "python -m pip install -e \".[dev,semantic]\"",
        f"artificial-human cognition-eval --suite all --seed {report['seed']} "
        "--embedding-backend fastembed --output-dir docs/benchmarks",
        "```",
        "",
        "Apart from timing, memory and timestamps, results are a deterministic function "
        "of the seed, the code, and (for semantic variants) the embedding model version.",
        "",
        "## 13. Limitations", "", _LIMITATIONS, "",
        "## 14. Research questions for the next phase", "", _QUESTIONS, "",
    ]
    external = [name for name in ("longmemeval", "digit-span") if name in suites]
    if external:
        lines += [
            "## 15. Independently authored datasets (opt-in)", "",
            "These suites read locally downloaded external data (see "
            "[EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md)). Raw data are never committed, and "
            "outputs contain no conversation text, question text, or participant identifiers.", "",
            "```",
            "artificial-human cognition-eval --suite longmemeval --suite digit-span \\",
            "  --longmemeval-path external-data/longmemeval/longmemeval_oracle.json \\",
            "  --digit-span-dir external-data/nemar/on003838/beh --report-stem external-evaluation \\",
            "  --output-dir docs/benchmarks",
            "```", "",
        ]
        for name in external:
            lines += _external_section(suites[name])
    return "\n".join(lines)
