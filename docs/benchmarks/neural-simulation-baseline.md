# Neural simulation performance and memory baseline

## Baseline network

The benchmark uses the first deterministic recurrent LIF microcircuit:

- Four neurons: sensory, two excitatory units, and one inhibitory unit.
- Five directed synapses, including delayed recurrent feedback and inhibitory feedback.
- One-millisecond simulation step.
- Four initial sensory-stimulation steps, then autonomous network evolution.
- With inhibition enabled, the deterministic demo decays back to rest after its evoked transient; disabling inhibition provides a recurrent-firing control.
- Fixed network topology and parameters; no model downloads, network calls, database, or LLM.
- The circuit tests additionally compare disconnected, feed-forward, recurrent, inhibition-off and fixed-rate controls.

## Reproduce

From the repository root:

```bash
python scripts/benchmark_neural_simulation.py --steps 20000
```

The command reports:
- wall-clock time and simulation steps per second;
- current and peak Python-traced memory allocations;
- simulated duration and event-queue counters;
- exact network size and benchmark identifier.

To save the measurement:

```bash
python scripts/benchmark_neural_simulation.py --steps 20000 \
  --output benchmark-output/neural-simulation-baseline.json
```

CI runs this benchmark on Python 3.13 and uploads the JSON with the offline benchmark artefacts.

**Recorded reference run (GitHub-hosted Ubuntu runner, Python 3.13.18, 27 November 2026):** 20,000 steps completed in 1.035225 seconds (19,319.47 steps/second), with 150,864 bytes peak Python-traced memory. The run scheduled and delivered 21 events with zero dropped or pending events. This is one runner-specific observation, not a portable performance guarantee.

Values are host/interpreter dependent: compare runs on the same runner class and do not use wall-clock speed as a correctness threshold. Python-traced allocations exclude interpreter baseline, native allocations, and total process RSS.

## Scientific and engineering limits

This is an engineering baseline for a small, deterministic LIF network. It is not a biological performance benchmark or a claim of anatomical fidelity. Weights are fixed in this milestone; plasticity, homeostasis, sensory encoding, and a closed-loop body/environment are separate roadmap tasks.
