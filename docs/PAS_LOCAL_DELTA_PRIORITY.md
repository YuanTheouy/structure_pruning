# PAS Local-Delta Priority Plan

Date: 2026-05-22

Purpose: test whether the current PAS probe radius is too large for a
SAM/flatness-style interpretation, without exploding the experiment budget.

## Current Concern

The existing PAS-Slope uses:

```text
S35 = L(35%) - L(30%)
```

This is useful as a stress signal for `40%`, but it may be too large to behave
like local flatness around the `30%` checkpoint. Relative to the remaining
structure at `30%`, a `30 -> 35` move removes:

```text
0.05 / 0.70 = 7.14% of the remaining structure
```

Even `30 -> 31` removes:

```text
0.01 / 0.70 = 1.43% of the remaining structure
```

For structured pruning, especially attention heads, this can still be a
discrete nonlocal jump.

## Priority Order

### P0: Projection-Only Mask-Change / Nestedness Sweep

Do this first. It is the cheapest and tells us which deltas are meaningful.

No PPL evaluation is needed in this stage.

Use the existing seed3025 candidate pool and compute projections at:

```text
30.00, 30.25, 30.50, 31.00, 32.00, 35.00, 40.00
```

For each adjacent pair, report:

- whether the projection is nested;
- how many candidates have any mask/dimension change;
- average and max changed modules;
- average and max changed channels/heads/neurons;
- whether the delta is too small to generate signal.

Decision rule:

- If `30.25` changes almost nothing, do not run PPL for it.
- If `30.50` changes enough but remains small, use it as the primary local
  flatness probe.
- If `30.50` is still too sparse/noisy, use `31.00`.
- Keep `35.00` as the stress probe, not as local flatness.

### P1: PPL-Only Local-Delta Probe

Run PPL only for the smallest 1-2 deltas that P0 says are meaningful.

Recommended starting choice:

```text
30.50 and 31.00
```

Fallback if `30.50` has too few mask changes:

```text
31.00 and 32.00
```

Do not run all deltas unless the first result is ambiguous.

For each candidate, compute:

```text
S30.5 = L(30.5%) - L(30%)
S31   = L(31.0%) - L(30%)
S35   = L(35.0%) - L(30%)   # existing stress signal
```

Analyze correlations / partial correlations with:

- downstream@30 average score from the existing P2 run;
- `L40`, `Delta40`, and `Regret40` from the existing P0 table.

Primary question:

```text
Does a smaller local-delta slope explain downstream@30 better than S35?
```

Secondary question:

```text
Does S35 remain better for predicting L40 / Regret40?
```

### P2: Downstream Re-Evaluation

Do not run new downstream tasks yet.

We already have downstream@30 for the 20 seed3025 candidates. Use that first.

Only run additional downstream if P1 gives a clear signal that a small local
delta predicts downstream@30 after controlling L30.

If P1 is negative, stop the flatness-generalization story and keep PAS as a
compression-stress diagnostic.

## Minimal Server Ask

Experiment execution chat should first implement:

```text
scripts/pas_check_projection_nestedness.py
```

from `docs/PAS_NESTED_PROJECTION_CHECK.md`, but make sure it also reports
mask-change magnitude for the delta grid:

```text
0.3000,0.3025,0.3050,0.3100,0.3200,0.3500,0.4000
```

Then, only after P0, implement/run a PPL delta probe for the selected 1-2 local
deltas.

## Recommended Commands

P0 projection-only:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

python scripts/pas_check_projection_nestedness.py \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --sparsities 0.3000,0.3025,0.3050,0.3100,0.3200,0.3500,0.4000 \
  --output-dir /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025_delta_sweep
```

Expected P0 outputs:

```text
nestedness_summary.csv
nestedness_violations.csv
nestedness_by_candidate.csv
mask_change_summary.csv
mask_change_by_candidate.csv
nestedness_manifest.json
```

Implementation note: `scripts/pas_check_projection_nestedness.py` now writes
both nestedness and mask-change magnitude artifacts. For the local-delta
decision, read `mask_change_summary.csv` first; use `nestedness_summary.csv` as
the protocol guardrail.

P1 PPL probe should wait until P0 identifies the smallest meaningful delta.

Strict nested follow-up: if current-projector local deltas are ambiguous, rerun
the PPL probe with the production nested projector path:

```bash
bash scripts/pas_run_local_probe_multigpu.sh \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --probe-sparsity 0.3025 \
  --delta 0.0025 \
  --base-sparsity 0.3000 \
  --projection-mode nested_from_base \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested \
  --gpu-ids "0 1 2 3 4 5 6 7" \
  --num-samples 64 \
  --batch-size 8 \
  --candidate-top-k 20
```

This evaluates `L30/L30.25/L30.50` using one cumulative path from the `30%`
projection. Later budgets may only preserve dimensions that survived the
previous budget, so the probe removes the current projector's re-addition
confound.

## Interpretation

If small deltas predict downstream@30:

```text
Local compression-path flatness may explain nominal capability beyond endpoint
PPL, while S35 is a separate larger-radius stress signal.
```

If small deltas do not predict downstream@30:

```text
Flatness/SAM is only motivation. PAS should not claim generalization or
downstream capability prediction.
```

If S35 remains the only strong signal:

```text
PAS is a stress-test selection method for cross-budget compression robustness,
not a local-flatness generalization method.
```
