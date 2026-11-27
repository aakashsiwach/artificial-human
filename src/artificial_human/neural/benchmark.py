"""Reproducible performance and Python-heap baseline for the LIF microcircuit."""
from __future__ import annotations

import tracemalloc
from time import perf_counter
from typing import Any

from artificial_human.neural.circuits import build_recurrent_microcircuit


def benchmark_neural_simulation(*, steps: int = 20_000) -> dict[str, Any]:
    """Measure a fixed 4-neuron/5-synapse network without external dependencies.

    Timing and traced-memory values are descriptive, machine-specific observations,
    not deterministic correctness thresholds. The network topology, inputs and
    simulated duration are fixed so runs can be compared on the same machine.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1_000_000:
        raise ValueError("steps must be between 1 and 1000000")

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        simulation = build_recurrent_microcircuit(recurrent=True)
        started = perf_counter()
        for index in range(steps):
            simulation.step({"sensory": 180.0} if index < 4 else None)
        elapsed = max(perf_counter() - started, 1e-12)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        snapshot = simulation.snapshot()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    return {
        "benchmark": "small-recurrent-lif-microcircuit-v1",
        "network": {
            "neurons": snapshot.neuron_count,
            "synapses": snapshot.synapse_count,
            "topology": "sensory -> excitatory_a -> excitatory_b; excitatory_a -> inhibitory -> excitatory_b; excitatory_b -> excitatory_a",
        },
        "simulation": {
            "dt_ms": simulation.dt_ms,
            "steps": steps,
            "simulated_duration_ms": round(steps * simulation.dt_ms, 8),
            "stimulus": {"neuron": "sensory", "current": 180.0, "steps": min(4, steps)},
        },
        "performance": {
            "wall_time_seconds": round(elapsed, 6),
            "steps_per_second": round(steps / elapsed, 2),
        },
        "python_traced_memory": {
            "current_bytes": current_bytes,
            "peak_bytes": peak_bytes,
            "note": "Python allocations only; excludes interpreter baseline, native allocator overhead and process RSS",
        },
        "events": {
            "scheduled": snapshot.events_scheduled,
            "delivered": snapshot.events_delivered,
            "dropped": snapshot.events_dropped,
            "pending": snapshot.pending_events,
        },
        "limitations": [
            "Timing is host- and interpreter-dependent and is not a CI pass/fail threshold.",
            "This measures a four-neuron engineering microcircuit, not biological brain performance.",
            "The network has fixed weights; no synaptic plasticity is included in this baseline.",
        ],
    }
