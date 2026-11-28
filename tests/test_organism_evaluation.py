from __future__ import annotations

import pytest

from artificial_human.neural.organism_evaluation import run_organism_evaluation


def test_organism_evaluation_reports_fixed_seed_baselines_and_ablations() -> None:
    report = run_organism_evaluation(
        seeds=(17,),
        integration_ticks=5,
        neural_benchmark_steps=1_000,
    )

    assert report["evaluation"] == "artificial-human-organism-evaluation-v1"
    assert report["fixed_seeds"] == [17]
    assert set(report["neural_microcircuit"]) == {
        "recurrent_with_inhibition",
        "feed_forward_with_inhibition",
        "disconnected",
        "recurrent_inhibition_off",
        "matched_size_fixed_rate_reference",
    }
    assert report["action_selection"]["random_accuracy"] >= 0.0
    assert report["action_selection"]["fixed_policy_accuracy"] >= 0.0
    assert len(report["adaptive_control_sensitivity"]) == 3
    assert len(report["embodied_closed_loop"]["standard"]["runs"]) == 1
    assert len(report["embodied_closed_loop"]["noisier_delayed_feedback"]["runs"]) == 1
    assert len(report["organism_integration_ablations"]) == 6
    assert report["resources"]["total_wall_seconds"] >= 0.0
    assert "consciousness" in " ".join(report["limitations"])


def test_organism_evaluation_rejects_unbounded_or_ambiguous_configuration() -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_organism_evaluation(seeds=(17, 17))
    with pytest.raises(ValueError, match="seeds"):
        run_organism_evaluation(seeds=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integration_ticks"):
        run_organism_evaluation(integration_ticks=0)
    with pytest.raises(ValueError, match="neural_benchmark_steps"):
        run_organism_evaluation(neural_benchmark_steps=0)
