# Simulation-only action selection and reward learning

## Research question

Can a small, inspectable opponent-channel policy learn a context-dependent
choice among a fixed set of simulated actions from bounded reward-prediction
errors, and does it outperform random and fixed-policy controls on held-out
episodes?

The biological motivation is cortico-basal-ganglia-thalamo-cortical loops and
evidence that dopamine-related prediction errors modulate corticostriatal
plasticity. Direct/indirect pathway accounts are useful computational
abstractions but are not a complete or universally settled explanation of
basal-ganglia function. The current implementation is deliberately simpler
than those circuits.

## Computational hypothesis

- Each state/action pair has a non-negative direct-channel activation and
  indirect-channel suppression.
- The policy's estimated action value is direct activation minus indirect
  suppression.
- A positive signed reward-prediction error increments the direct channel;
  a negative error increments the indirect channel.
- A fixed, seeded exploration process and an optional uncertainty bonus support
  sampling. Uncertainty falls as visits accumulate.
- Channel weights, reward inputs, state/action table size, and action count are
  bounded.
- A separate immutable allowlist is the normative safety boundary. Learned
  rewards and pathway weights cannot create or enable actions.

This is a tabular model-free baseline, not a full actor-critic model, a detailed
D1/D2 medium-spiny-neuron model, a dopamine concentration model, or an
anatomical simulation. It does not yet communicate through the LIF neural
substrate. The simulated body/environment integration is a later work package
(#34).

## Behavioural task and controls

The fixed-seed synthetic task maps four discrete contexts to four simulated
actions. The learned policy is evaluated on episodes not used for updating.
The report compares accuracy and mean reward with:
- a seeded random policy;
- a fixed policy that always chooses the first action.

The task is intentionally small and diagnostic. It tests contextual action
selection, not general intelligence, biological fidelity, or real-world
decision quality. Synthetic task success is necessary engineering evidence,
not independent neuroscience validation.

## Safety and runtime boundary

- Only identifiers in the construction-time simulation allowlist can be chosen
  or updated.
- There is no actuator, tool, device, network, or real-world action executor.
- Rewards must be finite and normalized to [-1, 1].
- Direct/indirect channel values and state/action capacity are bounded.
- State is held in process memory and reset clears learned values.
- The module has no database, LLM, network, or filesystem dependency.

## Failure cases and next experiments

Tests cover unsafe-action injection, out-of-range/non-finite rewards, state
capacity, deterministic selection, positive/negative prediction-error
direction, contextual held-out performance, and reward-context reversal.
Further work should compare alternative action-value and policy formulations,
test partial observability and delayed outcomes, quantify sensitivity to
exploration and reward scale, and connect the circuit to #34's simulated body
without allowing learned policy to override normative constraints.

## Selected research motivation

- Schultz, Dayan & Montague (1997), *A neural substrate of prediction and
  reward*, Science 275:1593–1599. DOI: 10.1126/science.275.5306.1593.
- Frank (2005), *Dynamic dopamine modulation in the basal ganglia: a
  neurocomputational account of cognitive deficits in medicated and
  nonmedicated Parkinsonism*, Journal of Cognitive Neuroscience 17:51–72.
  DOI: 10.1162/0898929052880093.
- Maia & Frank (2011), *From reinforcement learning models to psychiatric and
  neurological disorders*, Nature Neuroscience 14:154–162.
  DOI: 10.1038/nn.2723.
