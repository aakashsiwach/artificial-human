# Neural homeostasis and internal regulation (issue #25)

## Status and scope

This document records the evidence and falsifiable engineering hypothesis before implementation. The subsystem is a reduced, deterministic LIF-network control mechanism. It is not a molecular, anatomical, emotional, or neurotransmitter model.

The implementation remains in-process and bounded. It must not persist learned controller state to a database, call an LLM, or alter synaptic learning weights as a side effect of homeostatic regulation.

## Research basis

Established evidence:

- Neural activity is subject to homeostatic regulation at multiple spatial and temporal scales. Synaptic scaling is one mechanism: activity-dependent changes in synaptic strength can counter destabilising changes while preserving relative synaptic differences (Turrigiano, 2012, *Homeostatic synaptic plasticity: local and global mechanisms for stabilizing neuronal function*, https://pmc.ncbi.nlm.nih.gov/articles/PMC3249629/).
- Intrinsic excitability can change with activity history. Cultured cortical pyramidal neurons deprived of activity for two days increased their sensitivity to injected current, consistent with an activity-dependent change in intrinsic excitability (Desai, Rutherford & Turrigiano, 1999, *Plasticity in the intrinsic excitability of cortical pyramidal neurons*, https://pubmed.ncbi.nlm.nih.gov/10448215/).
- Inhibitory synapse plasticity has been proposed and modelled as a contributor to excitation/inhibition balance in sensory pathways and memory networks (Vogels et al., 2011, *Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks*, https://pubmed.ncbi.nlm.nih.gov/22075724/).

These findings motivate stabilising feedback, but they do not establish that one scalar rate controller reproduces biological homeostasis. Real systems have multiple cell types, local and global mechanisms, and distinct timescales.

## Computational hypothesis

**Hypothesis:** In this reduced LIF network, a slow per-neuron feedback controller that estimates spike rate over fixed simulation-step windows and adjusts a bounded input-current gain can reduce sustained runaway firing relative to a fixed-gain control, while allowing gain to recover after prolonged low activity.

The chosen control variable is an input-current gain, not a synaptic weight. This separation is deliberate: it preserves the learned synaptic magnitudes from issue #24 and makes the homeostatic action independently inspectable. The gain is a software control parameter, not a claim about a specific ion channel or receptor.

Proposed deterministic controller:

- Count spikes per neuron over a fixed 20-step observation window.
- Maintain an exponential moving average of observed spikes per simulation step.
- Compare the estimate with a documented target rate and deadband.
- Adjust each neuron's input gain multiplicatively in the corrective direction.
- Clamp gain to a closed interval (initially 0.01–1.0); never modify synapse weights.
- Expose the observed rate, smoothed rate, target error, current gain, quiescent windows, and low/high saturation counts.
- Snapshot and restore controller state in RAM; reset returns the gain to its source-defined initial value.

The first experiment uses a two-neuron recurrent excitatory loop under sustained external drive. It compares the same network and stimulus with regulation disabled versus enabled. The report records total/per-neuron spikes, rates, gain trajectory, saturation and dropped events. Separate tests cover silence, saturation, and recovery. This is a narrow computational stress test, not a claim of a general firing-rate law.

## Distinct internal signals for plasticity modulation

The modulation interface preserves separate typed channels:

- `salience` in [0, 1]: event relevance/selection strength.
- `reward_prediction_error` in [-1, 1]: signed teaching error; its sign remains inspectable.
- `stress_load` in [0, 1]: protective load that suppresses learning under stress.

The deterministic STDP magnitude gate is derived as:

`salience × abs(reward_prediction_error) × (1 − stress_load)`.

This gate is an engineering composition of distinct inputs. It is not named dopamine, does not claim to implement dopamine, and does not erase the signed prediction-error value. Pair-based STDP timing continues to determine potentiation versus depression; the gate only scales update magnitude. Other channels must not be collapsed into an unnamed generic scalar.

## Separate resource and biological constraints

Neural regulation is not compute-budget enforcement. The simulator's neuron/synapse/event capacities and current clamps remain hard software safety bounds. The `ResourceGovernor` tracks declared operations, tokens, and elapsed wall time; it is not an operating-system resource monitor. Neither category is a biological energy or metabolic model.

## Failure cases and acceptance evidence

Required tests and reports:

1. **Runaway excitation:** under the same sustained input, the recurrent network with homeostasis has a lower firing rate than its fixed-gain control; event queues remain within capacity.
2. **Quiescence/silence:** low observed activity increases gain only within its bound; lack of input alone must not fabricate spikes.
3. **Saturation:** sustained high activity reaches, but never crosses, the minimum gain; sustained silence reaches, but never crosses, the maximum gain. Saturation is observable.
4. **Recovery:** after the stress input is removed, the controller's rate estimate decays and gain moves back toward its source-defined baseline; a later moderate cue can again evoke activity.
5. **Determinism:** identical inputs produce identical spikes, gains, signals, and reports. In-memory snapshots restore the exact next-step trajectory; reset clears controller history and restores baseline gain.
6. **Modulation:** channel validation and gate derivation are deterministic; salience, signed prediction error, and stress remain separately inspectable.
7. **Baseline:** regulated and fixed-gain runs use identical topology, step count, and stimuli. Report both, not only the regulated score.

Limitations: the controller is a discrete engineering feedback law; its window size and parameters are chosen for an executable test, not mapped to biological seconds. It does not implement synaptic scaling, inhibitory plasticity, calcium sensing, ion-channel dynamics, dopamine, subjective emotion, or a literal hypothalamus/brainstem.

## Reproduction

The focused tests will be:

`pytest -q tests/test_neural_homeostasis.py tests/test_neural_simulation.py tests/test_neural_plasticity.py`

The experiment will also have a deterministic CLI/library entry point or test report that can be rerun without model downloads or external services.
