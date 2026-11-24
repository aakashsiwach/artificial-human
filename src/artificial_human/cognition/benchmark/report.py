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
memory recall, and the integrated multi-turn cycle. It establishes a
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
  ablations, as defined by the memory benchmark."""

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
  expected source / all conflicting memories appear in the top 5."""

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
        "integration": "none (fixed scenarios)",
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
    for name in ("attention", "working-memory", "prediction", "memory"):
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
    return "\n".join(lines)
