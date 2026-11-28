# Local synaptic plasticity experiment (issue #24)

## Mechanisms

The event-driven LIF simulator supports mutable synaptic magnitudes while preserving each synapse's polarity and configured maximum. Initial topology weights are retained in memory: reset restores those values, while an in-memory simulation snapshot captures and restores current learned weights.

PairBasedSTDP is an explicit all-to-all pair-based timing hypothesis. Each simulation-step observation decays pre/post traces. A presynaptic spike pairs with the decayed postsynaptic trace to depress; a postsynaptic spike pairs with the decayed presynaptic trace to potentiate. Thus pre-before-post potentiates and post-before-pre depresses, using the canonical pair-based timing convention. Simultaneous spikes do not form a zero-lag pair. A [0, 1] gate scales updates. Trace and weight bounds are checked, and traces can be reset or snapshotted.

RateBasedHebbian is a simple exponential rate-trace correlation comparator. A fixed-weight rule supplies the no-learning control. A zero-modulatory-gate STDP run is an explicit gate ablation and should match the fixed-weight control. The paired/unpaired task is driven by actual spikes emitted by the deterministic LIF simulator, not manually supplied spike labels. These are deliberately small engineering baselines, not exhaustive biological models.

## Reproduce

    pytest -q tests/test_neural_plasticity.py tests/test_neural_simulation.py
    python -c "from artificial_human.neural.plasticity import run_learning_rule_comparison; print(run_learning_rule_comparison())"

The training phase uses a teacher-forced postsynaptic spike after presynaptic activity in the paired pathway; the unpaired pathway receives presynaptic activity only. The initial synaptic weights are calibrated below the LIF firing threshold, with STDP potentiation sufficient to cross it. A separate fresh-network recall probe then supplies only presynaptic cues and observes a 12-step response window. The paired learned synapse must evoke its target neuron, while the unpaired and fixed-weight controls must not. The report includes both learned weights and two-pathway recall accuracy. The test asserts improvement over fixed weights for this selected task, not universal superiority over the rate-Hebbian comparator.

## Scope and limitations

This is a falsifiable mechanism-level experiment, not a biological validation. The task is deliberately tiny, deterministic and selected to test the stated timing hypothesis. It does not establish that STDP is the unique biological learning mechanism, nor does it cover BCM, eligibility traces, neuromodulation, heterosynaptic plasticity, large-scale learning, or generalization. Those should be introduced only with a defined task and separately reported ablations. Learned weights are in RAM and reset to source-defined initial values; persistence is intentionally not added.
