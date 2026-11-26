# External evaluation datasets: governance record

This file records every independently authored dataset considered for Artificial Human's evaluation, whether it was adopted, and why. Last reviewed **2026-11-24**.

Rules that apply to every external dataset:

- **Raw data are never committed.** Local copies live under `external-data/`, which is git-ignored.
- **Nothing is downloaded automatically.** Ordinary CLI runs and CI never fetch external data. External suites run only when a local path is passed explicitly.
- **External questions and participants are held out.** No retrieval threshold, weight, or mapping is tuned on them. Digit Span reference models are fitted only on a participant-level tuning partition.
- **Outputs are aggregates.** Reports contain no conversation text, question text, answers, or participant identifiers. Failure listings use dataset question IDs and categories only.
- **No LLM judge or remote API** is used anywhere in evaluation.
- **Provenance is pinned:** source revision, licence evidence, and the SHA-256 of the evaluated local file(s).

## Summary

| Dataset | Status | Reason |
| --- | --- | --- |
| LongMemEval, Oracle file (`longmemeval_oracle.json`) | **Adopted** (opt-in suite `longmemeval`) | MIT; evidence-only histories made of simulated sessions; turn-level evidence labels allow retrieval evaluation without an LLM |
| LongMemEval `_s` / `_m` and `custom_history` | Rejected | Filler sessions from ShareGPT/UltraChat (real user conversations); far larger than needed |
| NEMAR on003838, Digit Span (behavioural files only) | **Adopted as a human behavioural reference** (opt-in suite `digit-span`) | CC0; documented trial-level serial-recall data |
| NEMAR on004574, Cross-modal Oddball | Research candidate, not adopted | Undocumented behavioural columns; mostly clinical population; task not comparable to text novelty |
| BPI Challenge 2013 incidents (XES) | Research candidate, not adopted | Official challenge page describes a real-life Volvo IT log and identifies the XES file/DOI, but that page does not state a licence; do not use until provider terms are verified |
| LoCoMo (Maharana et al., 2024) | Rejected (earlier review) | Reported CC BY-NC 4.0, which is incompatible with Apache-2.0 redistribution; licence not verified at source |
| STS Benchmark (Cer et al., 2017) | Rejected (earlier review) | Mixed per-source licences, some requiring provider agreements |

---

## 1. LongMemEval — Oracle variant (adopted)

| Field | Record |
| --- | --- |
| Official name | LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory |
| Authors / citation | Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu. ICLR 2025. arXiv:2410.10813 |
| Source | https://github.com/xiaowu0162/LongMemEval (code) · https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (data) |
| Version | Hugging Face dataset revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` (last modified 2025-09-19), the "cleaned" release |
| DOI | None issued; cite arXiv:2410.10813 |
| Licence and evidence | **MIT.** The dataset card front matter declares `license: mit`. The GitHub repository licence API reports `MIT License` (copyright 2024 Di Wu). |
| File used | `longmemeval_oracle.json` only: 15,388,478 bytes, SHA-256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` (matches the Hugging Face LFS hash) |
| Size and modality | Text. 500 questions (133 temporal-reasoning, 133 multi-session, 78 knowledge-update, 70 single-session-user, 56 single-session-assistant, 30 single-session-preference; 30 are abstention questions, ids ending `_abs`). 948 sessions, 10,960 turns. 1–6 sessions per question (median 2) and 2–32 turns per session (median 12). |
| Provenance of content | According to the authors' README, Oracle histories contain **only evidence sessions**, which are simulated user sessions generated from synthetic user attributes. The ShareGPT/UltraChat filler sessions occur only in the `_s`/`_m` haystacks and in `custom_history`, which are not used. |
| Fields used | `question_id`, `question_type`, `question` (as the retrieval query, never output), `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions[].role/content/has_answer`, `answer_session_ids` |
| Fields excluded | `answer` (answer accuracy is not evaluated; no LLM judge) |
| Validation performed | JSON list of objects with all required fields; known `question_type` values; non-empty ids; equal lengths for session ids, dates and sessions; unique session ids; `answer_session_ids` ⊆ haystack ids; parseable dates (`YYYY/MM/DD (Day) HH:MM`, read as UTC because no offset is given); roles `user`/`assistant`; boolean `has_answer`; unique question ids. The adapter refuses any other file name. |
| Caveats | Every Oracle history session is an evidence session (0 non-evidence sessions), so **session-level retrieval is degenerate** (precision 1 by construction). The primary metric is turn-level retrieval of `has_answer` turns; 854 of 948 evidence sessions contain labelled turns, and every non-abstention question has at least one. Dates have no timezone. Content is English. Answers are not evaluated. |
| Privacy | Simulated conversations about synthetic personas. No content leaves the local process; outputs contain no text. |

### Download and run

```bash
mkdir -p external-data/longmemeval
curl -L -o external-data/longmemeval/longmemeval_oracle.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json"
sha256sum external-data/longmemeval/longmemeval_oracle.json
# expect 821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c

artificial-human cognition-eval --suite longmemeval \
  --longmemeval-path external-data/longmemeval/longmemeval_oracle.json \
  --report-stem external-evaluation --output-dir docs/benchmarks
# add --embedding-backend fastembed for semantic variants (local model only)
```

### How it is evaluated

Each question is ingested into its own fresh in-memory repository under a unique synthetic owner. Ingestion uses the normal consent-gated write API: one episodic memory per non-empty turn, with `occurred_at` set to the session date, source `longmemeval:<session_id>`, and session id, turn index and role as metadata. Sessions are ingested in chronological order. Only after the full history is ingested does the question run as a query, with `reference_time` set to the question date.

Metrics: turn-level Recall@{1,5,10,20}, Precision@5, nDCG@10, MRR, and recall against depth 1–20. Session-level Recall/Precision/MRR are reported with the caveat above. For abstention questions, the rejection rate (zero results) is reported. Breakdowns: question category, number of history sessions, history length in turns, position of the first evidence turn (early/middle/late third), single- vs multi-session evidence, and days between the latest evidence session and the question.

Isolation and cleanup: a probe search under a different owner must return nothing. Each repository is erased and closed after its question.

---

## 2. NEMAR on003838 — Digit Span (adopted as a human behavioural reference)

| Field | Record |
| --- | --- |
| Official name | EEG, pupillometry, ECG and photoplethysmography, and behavioral data in the digit span task and rest |
| Authors / citation | Yuri G. Pavlov, Dauren Kasanov, Alexandra I. Kosachenko, Alexander I. Kotyusov. Primary data paper: Pavlov, Kasanov, Kosachenko, Kotyusov & Busch (2022), *Pupillometry and electroencephalography in the digit span task*, Scientific Data 9, 325, doi:10.1038/s41597-022-01414-2 |
| Source | https://nemar.org/dataset/on003838 · mirror https://github.com/nemarDatasets/on003838 (commit `38c300f48ee869df1e5bdb032d7861bcf28ecc7c`) · source dataset OpenNeuro ds003838 v1.0.6 (doi:10.18112/openneuro.ds003838.v1.0.6) |
| Version and DOI | NEMAR v1.0.0, **doi:10.82901/nemar.on003838** (published 2026-06-22). BIDS 1.1.1. |
| Licence and evidence | **CC0.** `dataset_description.json` reads `"License": "CC0"`; the NEMAR page's JSON-LD reads `license: https://creativecommons.org/publicdomain/zero/1.0/`. |
| Files used | `sub-*/beh/sub-*_task-memory_beh.tsv` only: 86 files, 525,493 bytes. Manifest SHA-256 (sorted `name<TAB>sha256` lines) `d1d9524b3887864617e2c9b52af7f8a0edbd325e6c47d24373b5114448e97367`. Column definitions were taken from `sub-013_task-memory_beh.json`. |
| Size and modality | Behaviour: 86 participants, 9,276 memory trials (3,092 each at loads 5, 9 and 13), 96–108 trials per participant, 16 empty responses. Auditory digit sequences (2 s stimulus onset asynchrony) followed by serial recall. EEG, ECG, PPG and pupillometry exist in the dataset but were **not downloaded**. |
| Fields used | `condition` (load), `sequence`, `triggerCorrect` (per-position correctness), `partialScore` (consistency check), `task` (must be `remember`), `response` (only to count empty responses), `participant_id` (in memory only, for the participant-level split; never output) |
| Fields excluded | `trial`, `NCorrect`; all of `participants.tsv` (age, sex, handedness, NASA-TLX; not downloaded); all physiological recordings |
| Validation performed | Exact column set; numeric fields; sequence of digits whose length equals the load; per-position flags in {0,1}; `sum(triggerCorrect) == partialScore` (holds for all 9,276 trials) |
| Split | Participant-level: `sha256(seed:digit-span:participant)` assigns about 30% of participants to tuning (31) and the rest to eval (55) with the default seed. No participant is in both. |
| Caveats | Human serial recall involves perception, rehearsal, chunking and output interference. The software workspace has none of these. Human results are a behavioural reference, not a target or a test of biological equivalence. Strict serial-position scoring favours early positions. |
| Privacy | De-identified CC0 data. Only behavioural files are read, and only aggregates across participants are reported. |

### Download and run

```bash
mkdir -p external-data/nemar/on003838/beh
# For each sub-XXX listed in the repository tree (86 participants):
curl -L -o external-data/nemar/on003838/beh/sub-XXX_task-memory_beh.tsv \
  "https://raw.githubusercontent.com/nemarDatasets/on003838/38c300f48ee869df1e5bdb032d7861bcf28ecc7c/sub-XXX/beh/sub-XXX_task-memory_beh.tsv"
# The participant list is available from
# https://api.github.com/repos/nemarDatasets/on003838/git/trees/38c300f48ee869df1e5bdb032d7861bcf28ecc7c?recursive=1
# (keep paths matching sub-*/beh/sub-*_task-memory_beh.tsv only).

artificial-human cognition-eval --suite digit-span \
  --digit-span-dir external-data/nemar/on003838/beh \
  --report-stem external-evaluation --output-dir docs/benchmarks
```

### How it is evaluated

1. Human reference (eval participants): mean proportion correct per load, SD, 10th/50th/90th percentiles across participants, serial-position curves, and first/middle/last-third accuracy.
2. Software on identical sequences: each eval trial's digits are presented in order to the unmodified `WorkingMemory` (capacity 7, equal salience), and every position is probed.
3. Reference models fitted on **tuning participants only** and scored by serial-position RMSE on held-out participants: a recency store and a primacy store (each with its best capacity on tuning), and a flat per-load mean. The existing software is scored without fitting.

---

## 3. NEMAR on004574 — Cross-modal Oddball (research candidate, not adopted)

| Field | Record |
| --- | --- |
| Official name | Cross-modal Oddball Task |
| Authors / reference | Arun Singh, Rachel Cole, Arturo Espinoza, Jan R. Wessel, Jim Cavanagh, Nandakumar Narayanan; preprint doi:10.1101/2022.07.26.22278079 |
| Source | https://nemar.org/dataset/on004574 · mirror https://github.com/nemarDatasets/on004574 · OpenNeuro ds004574 v1.0.0 |
| Version and DOI | NEMAR v1.0.0, **doi:10.82901/nemar.on004574**. BIDS v1.2.1. |
| Licence and evidence | **CC0**: `dataset_description.json` reads `"License": "CC0"`. |
| Inspected | `dataset_description.json`, `README.md`, `CHANGES`, `participants.json` (column definitions only), `task-Oddball_events.json`, and the header and rows of a single participant's `beh.tsv` (inspected locally, not retained in the repository) |
| Behavioural data found | Per trial: `TrialNum`, `Block`, `Odd: Visual`, `Odd: Audio`, `Odd: Haptic`, `CueDirection`, `RT`, `RespDirection`, `Acc`, `TrialTime`, `ITI` (240 trials in the inspected file). Oddball pre-cues precede a directional go cue. |
| Why not adopted | (1) The `beh` columns have **no BIDS sidecar**, so their definitions (for example the haptic column, which the README does not mention, and RT units) cannot be verified. (2) 98 of 146 participants have Parkinson's disease. Interpreting the data needs the group label, which is health data (`participants.tsv`: diagnosis, MoCA, UPDRS), and the project should not process it. (3) The measured effect, reaction-time cost of an oddball pre-cue on a later go response, reflects orienting/distraction in a sensorimotor task, which has no defensible mapping to a lexical text-novelty score. |
| Status | Kept as a candidate for future perceptual-novelty work, which would need multimodal perception and its own consent and privacy design. No data from it are used. |

---

## 4. BPI Challenge 2013 — incident-management XES log (candidate; not yet adopted)

| Field | Record |
| --- | --- |
| Official name | BPI Challenge 2013 — Incident Management Log |
| Publisher / provenance | The official challenge page says the event log was prepared from Volvo IT Belgium's VINST incident/problem-management system and describes it as a real-life event log. |
| Source | [Official BPI Challenge 2013 page](https://ais.win.tue.nl/bpi/2013/challenge.html) · incident-log DOI: [10.4121/500573e6-accc-4b0c-9576-aa5468b10cee](https://doi.org/10.4121/500573e6-accc-4b0c-9576-aa5468b10cee) |
| Format | The challenge page identifies the incident, open-problem, and closed-problem log files as XES; source CSV is also listed. |
| Licence and evidence | **Not verified.** The official challenge page identifies the dataset and DOI but does not state a licence. Secondary mirrors describe reuse terms inconsistently. The project therefore does not redistribute, download, or claim this dataset is approved for use. Verify the current provider terms and any organisational/privacy restrictions before running the opt-in adapter. |
| Status | Research candidate only. No benchmark results from this dataset are claimed. |

### Generic local XES schema-evaluation adapter

The `schema-event-log` suite accepts a local `.xes` or `.xes.gz` file. It does not download data, infer activities from free text, or write approved schemas into persistent memory.

- The XES `concept:name` activity labels are normalized into bounded event-type slugs.
- Trace/case IDs are salted with the local file SHA-256 and used only as hashed training contexts; they are not reported.
- Event attributes are excluded by default. Optional attributes require an explicit allowlist, and sensitive/resource-like keys are rejected.
- Files are capped at 100 MiB, 250,000 events, 20,000 cases, and 128 events per case. The evaluation selects at most 500 cases by default.
- Cases are ordered by their first event timestamp and split chronologically 80/20. Earlier cases generate proposals; later cases measure recurrence coverage.
- A shuffled-within-case order is reported as a simple null control. Metrics are case support, coverage, shuffled coverage, coverage lift, and activity-vocabulary overlap.
- Typical XES logs do not contain gold schema labels. These metrics are **not** schema precision, semantic correctness, or causal understanding. A single dataset does not demonstrate cross-domain transfer.
- Output contains counts, ratios, split metadata, and the dataset SHA-256 only. External files remain local and must be covered by a separately verified data licence/terms review.

After obtaining a dataset and reviewing its terms:

```bash
artificial-human cognition-eval --suite schema-event-log \
  --schema-event-log-path external-data/workflow.xes.gz \
  --schema-event-log-max-cases 500 \
  --report-stem external-schema --output-dir docs/benchmarks
```

No external file is downloaded automatically and the suite is not run in CI.

---
## 5. Earlier rejections (recorded in COGNITIVE_EVALUATION.md)

- **LoCoMo** (Maharana et al., ACL 2024): reported as CC BY-NC 4.0 by secondary sources. Non-commercial terms are incompatible with this Apache-2.0 project, and the licence was not verified at source.
- **STS Benchmark** (Cer et al., 2017): scores are CC BY-SA 4.0, but source texts carry separate licences, some requiring agreements.

## Local layout and ignore rules

```
external-data/                      # git-ignored (.gitignore: external-data/)
  longmemeval/longmemeval_oracle.json
  nemar/on003838/beh/sub-*_task-memory_beh.tsv
  # optional: operator-supplied XES/XES.GZ files (no standard filename enforced)
```

Intermediate files are not written. Reports go only where `--output-dir` points, and contain aggregates only.
