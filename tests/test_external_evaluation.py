"""External-dataset adapters, calibration and generalisation.

All fixtures here are tiny, project-authored, synthetic files written to a
temporary directory. They mimic the documented schemas of the external
datasets but contain none of their data.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from artificial_human.cli import main
from artificial_human.cognition.benchmark.attention_eval import (
    HELD_OUT_CONDITIONS,
    TRAIN_CONDITIONS,
    attention_generalisation,
)
from artificial_human.cognition.benchmark.datasets import generate_attention_tasks, prediction_items
from artificial_human.cognition.benchmark.external import (
    ExternalDatasetError,
    sha256_file,
    sha256_manifest,
)
from artificial_human.cognition.benchmark.external import longmemeval as lme
from artificial_human.cognition.benchmark.external.digit_span import (
    load_digit_span,
    participant_split,
    run_digit_span,
)
from artificial_human.cognition.benchmark.metrics import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_logistic,
    reliability_bins,
)
from artificial_human.cognition.benchmark.prediction_eval import (
    calibration_report,
    existing_novelty,
)
from artificial_human.cognition.benchmark.runner import resolve_suites, run_cognitive_evaluation
from artificial_human.cognition.benchmark.working_memory_eval import (
    working_memory_generalisation,
)
from artificial_human.memory.repository import MemoryRepository

SECRET_PHRASE = "zebra-umbrella-quartz"  # must never appear in any output


# ------------------------------------------------------------------ fixtures


def _turn(role: str, content: str, has_answer: bool = False) -> dict[str, object]:
    return {"role": role, "content": content, "has_answer": has_answer}


def write_longmemeval_fixture(directory: Path) -> Path:
    """Three synthetic questions in the LongMemEval Oracle schema."""
    data = [
        {
            "question_id": "fixture_q1", "question_type": "single-session-user",
            "question": "Which colour did I paint the garden shed?", "answer": "green",
            "question_date": "2024/05/20 (Mon) 10:00",
            "haystack_session_ids": ["answer_q1_1"],
            "haystack_dates": ["2024/05/01 (Wed) 09:00"],
            "haystack_sessions": [[
                _turn("user", f"I painted the garden shed green last weekend {SECRET_PHRASE}.", True),
                _turn("assistant", "Green sounds lovely for a garden shed."),
                _turn("user", "I also bought new tulip bulbs."),
                _turn("assistant", "Tulips should go in during autumn."),
            ]],
            "answer_session_ids": ["answer_q1_1"],
        },
        {
            "question_id": "fixture_q2", "question_type": "multi-session",
            "question": "How many marathons have I run in total?", "answer": "two",
            "question_date": "2024/09/01 (Sun) 12:00",
            "haystack_session_ids": ["answer_q2_2", "answer_q2_1"],
            "haystack_dates": ["2024/08/01 (Thu) 08:00", "2024/03/10 (Sun) 08:00"],
            "haystack_sessions": [
                [_turn("user", "I finished my second marathon in Berlin today.", True),
                 _turn("assistant", "Congratulations on the Berlin marathon!")],
                [_turn("user", "My first marathon was in Lisbon this spring.", True),
                 _turn("assistant", "Lisbon is a beautiful marathon course."),
                 _turn("user", "My knee feels fine now.")],
            ],
            "answer_session_ids": ["answer_q2_2", "answer_q2_1"],
        },
        {
            "question_id": "fixture_q3_abs", "question_type": "temporal-reasoning",
            "question": "When did I adopt my parrot?", "answer": "never mentioned",
            "question_date": "2024/06/01 (Sat) 12:00",
            "haystack_session_ids": ["answer_q3_1"],
            "haystack_dates": ["2024/05/05 (Sun) 18:00"],
            "haystack_sessions": [[
                _turn("user", "I adopted a cat named Miso.", True),
                _turn("assistant", "Miso is a cute name for a cat."),
            ]],
            "answer_session_ids": ["answer_q3_1"],
        },
    ]
    path = directory / "longmemeval_oracle.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


COLUMNS = "participant_id\ttask\tcondition\ttrial\tsequence\tresponse\tNCorrect\tpartialScore\ttriggerCorrect"


def write_digit_span_fixture(directory: Path, participants: int = 12) -> Path:
    """Synthetic participants with a primacy gradient (early positions correct)."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(participants):
        participant = f"sub-9{index:02d}"
        rows = [COLUMNS]
        for trial, (load, sequence) in enumerate(((5, "31415"), (9, "271828182"),
                                                   (13, "1123581321345"))):
            kept = min(load, 4 + index % 3)
            flags = "1" * kept + "0" * (load - kept)
            response = sequence[:kept]
            rows.append("\t".join([participant, "remember", str(load), str(trial + 1), sequence,
                                   response, str(kept), str(kept), flags]))
        (directory / f"{participant}_task-memory_beh.tsv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
    return directory


# ------------------------------------------------------------------ metrics


def test_calibration_metrics() -> None:
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([0.5, 0.5], [0, 1]) == 0.25
    bins = reliability_bins([0.05, 0.95, 0.97], [0, 1, 1], bins=10)
    assert bins[0]["count"] == 1 and bins[9]["count"] == 2
    assert bins[9]["observed_rate"] == 1.0
    assert expected_calibration_error([0.05, 0.95, 0.97], [0, 1, 1]) == pytest.approx(0.0433, abs=1e-3)
    with pytest.raises(ValueError):
        reliability_bins([1.2], [1])
    with pytest.raises(ValueError):
        brier_score([], [])
    parameters = fit_logistic([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])  # separable: ridge keeps it finite
    assert parameters is not None and parameters[1] > 0
    assert calibration_slope_intercept([0.2, 0.8], [1, 1]) is None
    intercept, slope = calibration_slope_intercept(
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6], [0, 0, 1, 1, 1, 0, 0, 1]) or (None, None)
    assert intercept is not None and slope is not None


def test_prediction_calibration_is_fitted_on_tuning_only() -> None:
    items = prediction_items()
    tuning = [item for item in items if item.split == "tuning"]
    evaluation = [item for item in items if item.split == "eval"]
    report = calibration_report(existing_novelty, tuning, evaluation)
    expected = fit_logistic([existing_novelty(item) for item in tuning],
                            [item.unexpected for item in tuning])
    assert expected is not None
    assert report["logistic_parameters"] == {
        "intercept": round(expected[0], 4), "slope": round(expected[1], 4)}
    assert report["logistic_mapping_fitted_on"] == "tuning"
    calibrated, raw = report["logistic_calibrated"], report["uncalibrated_score_as_probability"]
    assert calibrated["auroc"] == raw["auroc"]  # monotone mapping keeps the ranking
    assert "base_rate_reference" in report
    assert sum(item["count"] for item in calibrated["reliability_bins"]) == len(evaluation)


# ----------------------------------------------------------- generalisation


def test_attention_generalisation_tunes_only_on_training_conditions() -> None:
    tasks = generate_attention_tasks(3)
    report = attention_generalisation(tasks)
    assert report["train_conditions"] == list(TRAIN_CONDITIONS)
    assert set(report["held_out_conditions"]).isdisjoint(TRAIN_CONDITIONS)
    assert report["tuning_tasks"] == sum(
        1 for t in tasks if t.split == "tuning" and t.condition in TRAIN_CONDITIONS)
    tuned = report["tuned_relevance_weight"]
    best_key, _ = max(report["tuning_ndcg_by_weight"].items(),
                      key=lambda item: (item[1], float(item[0])))
    assert float(best_key) == tuned
    # Removing every held-out task must not change the tuned weight (no leakage).
    without_held_out = [t for t in tasks if t.condition not in HELD_OUT_CONDITIONS]
    assert attention_generalisation(without_held_out)["tuned_relevance_weight"] == tuned
    assert attention_generalisation(tasks) == report


def test_working_memory_grid_is_complete_and_deterministic() -> None:
    first = working_memory_generalisation(5)
    assert first["cells"] == 4 * 6 * 3 * 3
    assert first == working_memory_generalisation(5)
    existing = first["mean_retention_by_factor"]["existing_working_memory"]
    assert existing["delay"]["0"] >= existing["delay"]["16"]
    fifo = first["mean_retention_by_factor"]["fifo"]["distractor_salience"]
    assert len(set(fifo.values())) == 1  # FIFO ignores salience entirely


# -------------------------------------------------------------- LongMemEval


def test_longmemeval_adapter_scores_turn_level_evidence(tmp_path, monkeypatch) -> None:
    path = write_longmemeval_fixture(tmp_path)
    closed: list[bool] = []
    original = MemoryRepository.close

    def tracking_close(self: MemoryRepository) -> None:
        closed.append(True)
        original(self)

    monkeypatch.setattr(MemoryRepository, "close", tracking_close)
    result = lme.run_longmemeval(path)
    summary = result["summary"]
    assert closed == [True, True, True]  # every per-question repository was released
    assert summary["questions"] == 3 and summary["abstention_questions"] == 1
    assert summary["owner_leaks"] == 0 and summary["isolation_probe_results"] == 0
    lexical = summary["variants"]["lexical"]
    assert lexical["overall"]["questions"] == 2
    assert lexical["overall"]["turn_recall@20"] == 1.0
    assert lexical["by_category"]["multi-session"]["questions"] == 1
    assert lexical["by_evidence_sessions"]["multi-session evidence"]["questions"] == 1
    assert lexical["session_level"]["session_precision@1"] == 1.0
    assert lexical["abstention"]["questions"] == 1
    assert result["dataset"]["sha256"] == sha256_file(path)
    serialised = json.dumps(result)
    for forbidden in (SECRET_PHRASE, "garden shed", "Which colour", "Miso"):
        assert forbidden not in serialised  # no conversation or question text in outputs


def test_longmemeval_validation_rejects_malformed_data(tmp_path) -> None:
    path = write_longmemeval_fixture(tmp_path)
    original = json.loads(path.read_text(encoding="utf-8"))

    def broken(mutate) -> None:
        data = json.loads(json.dumps(original))
        mutate(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ExternalDatasetError):
            lme.load_longmemeval(path)

    broken(lambda d: d[0].pop("answer_session_ids"))
    broken(lambda d: d[0].__setitem__("question_date", "20 May 2024"))
    broken(lambda d: d[0].__setitem__("answer_session_ids", ["not-in-haystack"]))
    broken(lambda d: d[0].__setitem__("question_type", "telepathy"))
    broken(lambda d: d[0]["haystack_sessions"][0][0].__setitem__("role", "narrator"))
    broken(lambda d: d[0]["haystack_dates"].append("2024/05/01 (Wed) 09:00"))
    broken(lambda d: d.append(d[0]))  # duplicate question id
    other = tmp_path / "longmemeval_s_cleaned.json"
    other.write_text("[]", encoding="utf-8")
    with pytest.raises(ExternalDatasetError, match="Only"):
        lme.load_longmemeval(other)
    with pytest.raises(ExternalDatasetError, match="not found"):
        lme.load_longmemeval(tmp_path / "missing" / "longmemeval_oracle.json")


# -------------------------------------------------------------- digit span


def test_digit_span_reference_uses_participant_split_and_reports_aggregates(tmp_path) -> None:
    directory = write_digit_span_fixture(tmp_path / "beh")
    trials, meta = load_digit_span(directory)
    assert meta["participants"] == 12 and meta["trials"] == 36
    assert meta["manifest_sha256"] == sha256_manifest(sorted(directory.glob("*.tsv")))
    result = run_digit_span(directory, seed=7)
    assert result["dataset"]["tuning_participants"] + result["dataset"]["eval_participants"] == 12
    assert participant_split("sub-900", 7) == participant_split("sub-900", 7)
    software = result["software_on_identical_sequences"]
    assert software["13"]["serial_position_curve"][:6] == [0.0] * 6  # pure recency
    assert software["5"]["mean_proportion_correct"] == 1.0
    humans = result["human_reference_eval_participants"]["13"]["curve_shape"]
    assert humans["first_third"] > humans["last_third"]  # fixture has a primacy gradient
    assert result["fitted_primacy_capacity"] >= 4
    assert "sub-9" not in json.dumps(result)  # no participant identifiers in outputs
    del trials


def test_digit_span_validation(tmp_path) -> None:
    directory = write_digit_span_fixture(tmp_path / "beh", participants=2)
    target = next(directory.glob("*.tsv"))
    lines = target.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].rsplit("\t", 1)[0] + "\t11111"  # flags no longer match partialScore
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ExternalDatasetError, match="inconsistent"):
        load_digit_span(directory)
    target.write_text("wrong\theader\n", encoding="utf-8")
    with pytest.raises(ExternalDatasetError, match="columns"):
        load_digit_span(directory)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExternalDatasetError, match="No"):
        load_digit_span(empty)
    with pytest.raises(ExternalDatasetError, match="not found"):
        load_digit_span(tmp_path / "missing")


# ------------------------------------------------------------ runner / CLI


def test_external_suites_are_opt_in() -> None:
    assert "longmemeval" not in resolve_suites(["all"])
    assert "longmemeval" in resolve_suites(["all"], available_external=["longmemeval"])
    with pytest.raises(ValueError, match="--longmemeval-path"):
        resolve_suites(["longmemeval"])


def test_cli_external_suites(tmp_path, capsys) -> None:
    path = write_longmemeval_fixture(tmp_path)
    directory = write_digit_span_fixture(tmp_path / "beh")
    assert main(["cognition-eval", "--suite", "longmemeval"]) == 2
    assert main(["cognition-eval", "--suite", "longmemeval", "--suite", "digit-span",
                 "--longmemeval-path", str(path), "--digit-span-dir", str(directory),
                 "--output-dir", str(tmp_path / "out"), "--report-stem", "external-evaluation"]) == 0
    report = (tmp_path / "out" / "external-evaluation-report.md").read_text(encoding="utf-8")
    assert "## 15. Independently authored datasets" in report
    assert SECRET_PHRASE not in report and "sub-9" not in report
    assert main(["cognition-eval", "--suite", "longmemeval",
                 "--longmemeval-path", str(tmp_path / "nope" / "longmemeval_oracle.json")]) == 1
    capsys.readouterr()


def test_offline_evaluation_makes_no_network_calls(tmp_path, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    report = run_cognitive_evaluation(
        ["all"], seed=2,
        longmemeval_path=write_longmemeval_fixture(tmp_path),
        digit_span_dir=write_digit_span_fixture(tmp_path / "beh"),
    )
    assert report["suites_run"][-2:] == ["longmemeval", "digit-span"]
    assert report["network_used_by_evaluation"] is False


def test_repository_close_releases_in_memory_database() -> None:
    repository = MemoryRepository(":memory:")
    repository.grant_consent("owner")
    repository.close()
    repository.close()  # idempotent
    assert repository._keeper is None  # noqa: SLF001 - verifying release
