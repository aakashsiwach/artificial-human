# Distributed population coding (issue #26)

## Research grounding and scope

A population code represents a variable through the coordinated activity of multiple units rather than assigning the whole variable to one unit. In the standard tuning-curve view, units have different preferred values and their activity forms an overlapping population pattern; downstream decoders infer a value from that pattern. Basis-function views explain how a population can support downstream nonlinear computations. This is a useful computational starting point, not a unique or complete account of neural coding (Pouget, Dayan & Zemel, 2000: https://www.nature.com/articles/35039062).

Population patterns also have geometry: similarity and correlation across unit activity can affect what is decodable and how robust the representation is. A simple population-vector decoder is not universally reliable; assumptions about tuning distributions and response selectivity matter (for a concrete comparison of population-vector and maximum-likelihood decoding, see https://pmc.ncbi.nlm.nih.gov/articles/PMC4040453/; Panzeri et al., 2022: https://www.nature.com/articles/s41583-022-00606-4).

Pattern separation and pattern completion are distinct computational hypotheses. Separation transforms similar inputs into more distinct representations; completion retrieves a common stored pattern from partial or degraded cues. They are often discussed in relation to dentate-gyrus / hippocampal circuitry, but neither phenomenon should be inferred merely from a representation looking sparse or from a successful memory lookup. The input similarity and the recovered output similarity must both be measured (Leal & Yassa, 2018: https://www.nature.com/articles/s41593-017-0065-1; review of dentate-gyrus circuits: https://pmc.ncbi.nlm.nih.gov/articles/PMC7115869/).

## Candidate coding schemes

| Scheme | Expected property | Useful control / limitation |
|---|---|---|
| Hand-coded normalized feature vector | Direct, compact, linearly inspectable | No distributed tuning; useful as a feature baseline |
| One-hot nearest-centre code | Sparse and easy to inspect | Quantization collisions discard differences within one bin |
| Overlapping Gaussian tuning curves | Smooth nearby-input generalization and a decodable distributed activity pattern | Dense overlap can increase interference between similar patterns |
| Top-k winner-take-all transform | Reduces overlap between selected patterns | Loses some smoothness; sparsity is an engineering transform, not a claim about dentate-gyrus biology |

## Selected mechanism

For each numeric feature, use a bank of evenly spaced preferred values over [-1, 1]. Map a finite input feature to that interval with `tanh` and activate each unit with a Gaussian tuning function:

`activity = exp(-0.5 * ((x - preferred_value) / width) ** 2) * confidence`

Each feature has its own bank; activations are concatenated into one distributed code. The preferred values and width are explicit, deterministic parameters. The code carries only its activation tuple, preferred values, observed-feature mask, timestamp, feature count, coding scheme and uncertainty. It does not retain the source feature tuple or a raw sensory event.

The decoder estimates each observed feature by the activity-weighted mean of its preferred values. It returns normalized feature estimates (in [-1, 1]); it does not invert `tanh` or claim to reconstruct the original unbounded sensor units. The code preserves event time and confidence-derived uncertainty as separate metadata; similarity matching does not silently turn a timestamp into a feature.

A deterministic top-k-per-feature transform is available as an explicit separation experiment. It is compared to the dense code on a specified pair of moderately separated inputs. It is not the default encoder and does not imply that sparse coding is always better.

## Falsifiable engineering hypotheses

1. **Smooth generalization:** nearby values should have higher cosine similarity between dense population patterns than distant values. The one-hot baseline should exhibit a discontinuity at quantization boundaries.
2. **Interference control:** two inputs that quantize to the same one-hot bin should collide in the one-hot baseline; a dense code should preserve a measurable distinction and retrieve the correct stored pattern for a cue that exposes only that feature.
3. **Pattern separation:** for a documented moderately separated pair, top-k coding should reduce cosine overlap relative to the dense code. This is a task-specific trade-off, not a universal claim.
4. **Pattern completion:** when one feature bank is masked, a bounded associative store should retrieve the original complete code from the remaining feature banks. A zero-fill baseline cannot reconstruct the missing feature and should report it as unavailable.
5. **Neural drive:** code activations should map deterministically to a corresponding set of LIF external currents and produce structured spike activity. The neuron engine receives currents and population-unit identifiers, not original sensory values or event objects.

## Measurement and failure cases

- Report cosine similarity, decoded normalized-feature error, one-hot collision, top-k overlap reduction, correct completion identifier, and LIF spikes per unit.
- Exercise values at the preferred-value endpoints, outside the nominal range, zero confidence, partial masks, equal-score ties, malformed vectors, invalid timestamps, invalid units/width, and associative-store capacity exhaustion.
- The completion store is in-memory, bounded and stores immutable population codes plus opaque pattern identifiers. It must not store original numeric input features.
- Determinism is tested by repeating the same input and comparing codes, retrieval and simulated spikes.
- Baselines use the same feature dimension and preferred-value grid where applicable; metrics are reported separately rather than presenting one aggregate synthetic accuracy score.

## Explicit limitations

This is a small deterministic computational representation, not a fit to biological recordings and not a claim that cortical neurons use identical Gaussian curves. It has no measured trial noise, neuron-to-neuron noise correlations, probabilistic posterior decoder, learned receptive fields, or anatomically specified cortical/hippocampal pathways. Confidence-derived uncertainty is metadata; the activity pattern does not yet encode a calibrated probability distribution. The sparse transform is a controlled coding ablation, not a biological model of dentate gyrus. Numeric feature encoding is independent of text keywords and LLM calls.

## Reproduction

`pytest -q tests/test_population_coding.py tests/test_organism_loop.py`

Run the deterministic report without a model download, database or network service:

`python -c "from artificial_human.neural.population import run_population_coding_comparison; print(run_population_coding_comparison())"`
