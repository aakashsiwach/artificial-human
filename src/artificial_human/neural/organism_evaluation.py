"""Reproducible fixed-seed evaluation across neural and organism subsystems.

The runner composes existing task-specific benchmarks; it does not introduce a
new cognitive model or combine their scores into a claim of intelligence.
"""
from __future__ import annotations

import platform
import tracemalloc
from dataclasses import asdict
from time import perf_counter

from artificial_human.development.program import run_developmental_comparison
from artificial_human.neural.action_selection import run_action_selection_comparison
from artificial_human.neural.adaptive_control import run_adaptive_control_comparison
from artificial_human.neural.benchmark import benchmark_neural_simulation
from artificial_human.neural.body_environment import run_closed_loop_task
from artificial_human.neural.circuits import run_fixed_rate_reference, run_microcircuit_demo
from artificial_human.neural.organism_loop import run_integration_ablation_comparison


_DEFAULT_SEEDS = (17, 23, 41)
_SENSITIVITY = ((0.0, 0), (0.02, 2), (0.05, 5))


def _validate_seeds(seeds: tuple[int, ...]) -> None:
    if (
        not isinstance(seeds, tuple)
        or not 1 <= len(seeds) <= 8
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a tuple of 1..8 unique integers")


def _microcircuit_summary(
    *,
    recurrent: bool = True,
    inhibition_enabled: bool = True,
    connected: bool = True,
) -> dict[str, object]:
    result = run_microcircuit_demo(
        recurrent=recurrent,
        inhibition_enabled=inhibition_enabled,
        connected=connected,
    )
    trace = result["trace"]
    spike_count = sum(len(step["spikes"]) for step in trace)
    final_step = trace[-1] if trace else {}
    return {
        "neuron_count": result["neuron_count"],
        "synapse_count": result["synapse_count"],
        "total_spikes": spike_count,
        "events_scheduled": result["events_scheduled"],
        "events_delivered": result["events_delivered"],
        "events_dropped": result["events_dropped"],
        "pending_events": result["pending_events"],
        "final_membrane_potentials_mV": final_step.get("membrane_potentials_mV", {}),
    }


def _closed_loop_summary(
    *, seed: int, sensor_noise_std: float, feedback_delay_steps: int
) -> dict[str, object]:
    result = run_closed_loop_task(
        max_steps=240,
        feedback_delay_steps=feedback_delay_steps,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
    )
    return {
        "seed": result.seed,
        "steps": result.steps,
        "success": result.success,
        "disturbance_shift_tick": result.disturbance_shift_tick,
        "initial_distance": result.initial_distance,
        "final_distance": result.final_distance,
        "final_position": result.final_position,
        "final_velocity": result.final_velocity,
        "action_counts": dict(result.action_counts),
        "mean_abs_prediction_error": result.mean_abs_prediction_error,
        "mean_abs_prediction_error_after_shift": result.mean_abs_prediction_error_after_shift,
        "max_abs_prediction_error": result.max_abs_prediction_error,
        "learned_disturbance_bias": result.learned_disturbance_bias,
        "sensory_retained_count": result.sensory_retained_count,
        "neural_spike_count": result.neural_spike_count,
    }


def _summarize_runs(runs: tuple[dict[str, object], ...]) -> dict[str, object]:
    successes = sum(bool(run["success"]) for run in runs)
    count = len(runs)
    return {
        "runs": list(runs),
        "successes": successes,
        "failure_count": count - successes,
        "success_rate": round(successes / count, 8),
        "mean_final_distance": round(
            sum(float(run["final_distance"]) for run in runs) / count, 8
        ),
        "mean_abs_prediction_error": round(
            sum(float(run["mean_abs_prediction_error"]) for run in runs) / count, 8
        ),
        "mean_post_shift_prediction_error": round(
            sum(float(run["mean_abs_prediction_error_after_shift"]) for run in runs) / count,
            8,
        ),
    }


def run_organism_evaluation(
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    integration_ticks: int = 60,
    neural_benchmark_steps: int = 20_000,
) -> dict[str, object]:
    """Run deterministic, offline baselines, ablations, sensitivity and resource checks.

    Wall-clock and Python-heap measurements are descriptive observations. They
    are deliberately not correctness thresholds because runner hardware varies.
    """
    _validate_seeds(seeds)
    if (
        isinstance(integration_ticks, bool)
        or not isinstance(integration_ticks, int)
        or not 1 <= integration_ticks <= 1_000
    ):
        raise ValueError("integration_ticks must be an integer in [1, 1000]")
    if (
        isinstance(neural_benchmark_steps, bool)
        or not isinstance(neural_benchmark_steps, int)
        or not 1 <= neural_benchmark_steps <= 1_000_000
    ):
        raise ValueError("neural_benchmark_steps must be an integer in [1, 1000000]")

    overall_started = perf_counter()
    suite_seconds: dict[str, float] = {}

    started = perf_counter()
    neural_performance = benchmark_neural_simulation(steps=neural_benchmark_steps)
    suite_seconds["neural_performance"] = round(perf_counter() - started, 6)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()

        started = perf_counter()
        microcircuits = {
            "recurrent_with_inhibition": _microcircuit_summary(),
            "feed_forward_with_inhibition": _microcircuit_summary(recurrent=False),
            "disconnected": _microcircuit_summary(connected=False),
            "recurrent_inhibition_off": _microcircuit_summary(inhibition_enabled=False),
        }
        fixed_rate = run_fixed_rate_reference()
        fixed_trace = fixed_rate["trace"]
        microcircuits["matched_size_fixed_rate_reference"] = {
            "neuron_count": fixed_rate["neuron_count"],
            "projection_count": fixed_rate["projection_count"],
            "final_rates": fixed_trace[-1]["rates"] if fixed_trace else {},
            "note": fixed_rate["note"],
        }
        suite_seconds["microcircuit_controls"] = round(perf_counter() - started, 6)

        started = perf_counter()
        developmental = asdict(run_developmental_comparison(training_trials=60))
        suite_seconds["developmental_learning"] = round(perf_counter() - started, 6)

        started = perf_counter()
        action_selection = asdict(run_action_selection_comparison(seed=23))
        suite_seconds["action_selection"] = round(perf_counter() - started, 6)

        started = perf_counter()
        adaptive_sensitivity = []
        for seed in seeds:
            for noise_std, delay in _SENSITIVITY:
                result = run_adaptive_control_comparison(
                    steps=600,
                    feedback_delay_steps=delay,
                    sensor_noise_std=noise_std,
                    seed=seed,
                )
                adaptive_sensitivity.append(asdict(result))
        suite_seconds["adaptive_control_sensitivity"] = round(perf_counter() - started, 6)

        started = perf_counter()
        standard_runs = tuple(
            _closed_loop_summary(
                seed=seed,
                sensor_noise_std=0.01,
                feedback_delay_steps=2,
            )
            for seed in seeds
        )
        noisy_delayed_runs = tuple(
            _closed_loop_summary(
                seed=seed,
                sensor_noise_std=0.03,
                feedback_delay_steps=5,
            )
            for seed in seeds
        )
        suite_seconds["embodied_closed_loop"] = round(perf_counter() - started, 6)

        started = perf_counter()
        organism_ablations = run_integration_ablation_comparison(ticks=integration_ticks)
        suite_seconds["organism_integration_ablations"] = round(perf_counter() - started, 6)

        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        report = {
            "schema_version": "1.0",
            "evaluation": "artificial-human-organism-evaluation-v1",
            "python_version": platform.python_version(),
            "fixed_seeds": list(seeds),
            "coverage": {
                "neural_propagation": "spiking recurrent, feed-forward, disconnected, inhibition-off and matched-size fixed-rate controls",
                "developmental_learning": "experience-dependent cue prediction vs frozen-plasticity and mature-from-start controls",
                "action_selection": "learned contextual policy vs seeded random and fixed-policy baselines",
                "adaptive_control": "learning-enabled vs frozen forward model across fixed seeds, three sensor-noise levels and feedback delays",
                "sensory_body_loop": "same seeded noisy sensory tasks with delayed feedback; includes prediction error after a fixed disturbance shift",
                "organism_integration": "full loop vs attention, working-memory, episodic-memory, LIF-substrate and learned-action ablations",
            },
            "neural_microcircuit": microcircuits,
            "neural_performance": neural_performance,
            "developmental_learning": developmental,
            "action_selection": action_selection,
            "adaptive_control_sensitivity": adaptive_sensitivity,
            "embodied_closed_loop": {
                "standard": _summarize_runs(standard_runs),
                "noisier_delayed_feedback": _summarize_runs(noisy_delayed_runs),
                "interpretation": (
                    "Each condition reports per-seed outcomes and prediction errors. "
                    "Failures remain visible; no success-rate threshold is inferred."
                ),
            },
            "organism_integration_ablations": list(organism_ablations),
            "resources": {
                "suite_wall_seconds": suite_seconds,
                "total_wall_seconds": round(perf_counter() - overall_started, 6),
                "python_traced_current_bytes": current_bytes,
                "python_traced_peak_bytes": peak_bytes,
                "memory_scope": (
                    "Python-traced allocations for the evaluation process; excludes "
                    "interpreter baseline, native allocations and process RSS."
                ),
                "timing_scope": (
                    "Host-specific elapsed time. Compare on the same runner class; "
                    "not used as a correctness gate."
                ),
            },
            "limitations": [
                "This is a deterministic software evaluation on synthetic tasks, not a human behavioural study.",
                "The older body/environment task and the continuous OrganismLoop are distinct experimental paths; their scores are not pooled.",
                "The microcircuit, developmental substrate and cognitive abstractions are reduced computational hypotheses, not anatomical replicas.",
                "No score establishes intelligence, consciousness, sentience, biological equivalence or human-level capability.",
                "Regression thresholds are intentionally not set here; repeated-run variance and task validity must be reviewed before gating on performance.",
            ],
        }
        return report
    finally:
        if not was_tracing:
            tracemalloc.stop()
