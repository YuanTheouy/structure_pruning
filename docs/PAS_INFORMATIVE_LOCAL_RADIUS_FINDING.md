# PAS Informative Local Radius Finding

Date: 2026-05-23

This note records the current core PAS interpretation after the strict nested
local-delta probes on OPT-2.7B and OPT-1.3B.

## Core Finding

PAS does not appear to be a monotone "smaller radius is always better" story.
Instead, the evidence points to a finite, nonzero informative local radius:

```text
30.00 -> 30.25: local but comparatively weak / under-informative
30.00 -> 30.50: strongest local downstream@30 signal
30.00 -> 31.00: weaker or unstable, likely starting to leave the local regime
30.00 -> 35.00: larger-radius compression stress signal, not local capability
```

This matters because the endpoint itself, `L30`, is not enough. The useful
signal is not simply "good at the nominal point"; it is how the candidate
responds to a small but nonzero additional structured-compression perturbation,
after controlling for `L30`.

In paper language:

```text
There exists an informative local compression radius. A perturbation that is
too small is under-informative, while a perturbation that is too large leaves
the local capability basin and becomes a stress test. In our experiments,
30.50% is the best local probe around the 30% checkpoint, whereas 35% is a
larger-radius stress probe for stricter 40% compression.
```

## Method Implication

Separate PAS into two radii:

```text
Local PAS:
  S30.5(A) = L(A, 30.5%) - L(A, 30.0%)
  Claim: local compression-path fragility; downstream@30 signal beyond L30.

Stress PAS:
  S35(A) = L(A, 35.0%) - L(A, 30.0%)
  Claim: larger-radius compression stress; predicts 40% degradation/regret.
```

Do not describe `S35` as local flatness. It is better framed as a
cross-budget stress diagnostic. The SAM/flatness analogy belongs, if at all,
to the strict nested local probe, especially `S30.5`.

## Evidence Snapshot

### OPT-1.3B, seed2025

Artifacts:

```text
/workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_nested/local_delta_analysis.md
/workspace/ckpts/pas_local_delta_probe/opt13b_seed2025_nested_analysis/local_signal_correlation.md
/workspace/ckpts/pas_stress_recovery_opt13b/downstream_candidate_summary_opt13b_seed2025.csv
/workspace/ckpts/pas_stress_recovery_opt13b/candidate_stress_table_opt13b_seed2025.csv
```

All-candidate downstream@30 relation after controlling endpoint `L30`:

| Predictor | Target | Partial Corr | Reading |
| --- | --- | ---: | --- |
| `S3025` | downstream@30 avg score | `-0.237` | Correct sign, weaker. |
| `S3050` | downstream@30 avg score | `-0.562` | Strongest local signal. |
| `S31` | downstream@30 avg score | `-0.018` | Essentially gone. |
| `S35` | downstream@30 avg score | about `-0.05` to `-0.08` | Weak downstream signal. |

Stress relation:

| Predictor | Target | Partial Corr | Reading |
| --- | --- | ---: | --- |
| `S35` | `L40` / `Regret40` | about `0.51` to `0.57` | Strong stress signal. |
| `S35` | `Delta40` | about `0.57` to `0.58` | Strong stress-degradation signal. |

Selection note: for OPT-1.3B, the endpoint winner is already the `L40` oracle,
so this model is not a clean case for proving selection improvement over the
endpoint. It is still valuable because it sharply separates the two signals:
`S30.5` explains downstream@30, while `S35` explains stricter-budget stress.

### OPT-2.7B, seed3025

Artifacts:

```text
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nested/local_delta_analysis.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nested_analysis/local_signal_correlation.md
/workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv
/workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv
```

All-candidate downstream@30 relation after controlling endpoint `L30`:

| Predictor | Target | Partial Corr | Reading |
| --- | --- | ---: | --- |
| `S3025` | downstream@30 avg score | about `-0.23` | Correct sign, weaker. |
| `S3050` | downstream@30 avg score | about `-0.33` | Best local signal among the small deltas. |
| `S31` | downstream@30 avg score | about `-0.16` | Weaker. |
| `S35` | downstream@30 avg score | about `-0.19` to `-0.20` | Not the best downstream-local signal. |

Stress relation:

| Predictor | Target | Partial Corr | Reading |
| --- | --- | ---: | --- |
| `S35` | `L40` / `Regret40` | about `0.78` to `0.81` | Very strong stress signal. |
| `S35` | `Delta40` | about `0.81` | Very strong stress-degradation signal. |

This matches the OPT-1.3B separation, although the downstream-local signal is
weaker than in OPT-1.3B.

## Why This Is Not Just Endpoint Selection

The endpoint claim would be:

```text
L30 alone explains the useful candidate.
```

The local-radius claim is stronger:

```text
After controlling L30, S30.5 still explains downstream@30.
```

That is the key reason a nonzero perturbation is needed. A zero-radius probe is
just the endpoint. A small but informative nested compression step probes
local fragility of the retained structure.

## Paper Claim Boundary

Strong claim supported:

```text
PAS reveals two distinct compression responses: a local capability-relevant
response at a small nonzero radius, and a larger-radius stress response for
stricter compression budgets.
```

Claim to avoid:

```text
S35 is local flatness or directly predicts downstream@30.
```

Better terminology:

- `S30.5`: local compression-path fragility.
- `S35`: compression stress slope.
- `L30`: endpoint quality.

## Next Replication Target

The next model/seed should test whether the radius curve remains:

```text
S30.25: weak
S30.50: strongest local downstream signal
S31: weaker/unstable
S35: stress, not downstream-local
```

If this pattern repeats, the method section can present `30.50%` as the
default local PAS radius and `35.00%` as the stress PAS radius.
