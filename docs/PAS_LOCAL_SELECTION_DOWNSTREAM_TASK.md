# PAS-Local Selection And Downstream@30 Test

Date: 2026-05-22

Purpose: test a new local-probe PAS variant. This is different from the current
PAS-S35 stress-test method.

## Background

Current PAS-Slope uses:

```text
S35(A) = L(35%) - L(30%)
```

This is a medium-radius stress signal and currently predicts held-out `40%`
compression behavior. It does **not** explain downstream@30 well.

The new question is:

```text
If we replace the 35% probe with a much smaller local probe, does the selected
candidate have better downstream@30?
```

Define local PAS variants:

```text
PAS-S30.25: choose by S3025(A) = L(30.25%) - L(30%)
PAS-S30.50: choose by S3050(A) = L(30.50%) - L(30%)
PAS-S31.00: choose by S31(A)   = L(31.00%) - L(30%)
```

All selection must be done inside a fixed endpoint-compatible set.

## Important Distinction

Do not mix these up:

- `PAS-S35`: old/current stress-test PAS.
- `PAS-S30.25/S30.50/S31.00`: new local-probe PAS variants.

This experiment asks whether the **new local PAS variants** explain or improve
downstream@30. It is not a re-interpretation of old PAS-S35.

## Available Inputs

Local delta probe already produced:

```text
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025/local_delta_scores.csv
```

It contains:

```text
candidate_id,L30,L3025,L3050,S3025,S3050,PPL30,PPL3025,PPL3050
```

Existing downstream@30 outputs:

```text
/workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv
/workspace/ckpts/pas_stress_recovery/downstream_analysis_opt27b_seed3025.csv
```

Existing stress table:

```text
/workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv
```

This includes current `S35`, `L40`, `Delta40`, and `Regret40`.

## P0: Analyze Existing S30.25/S30.50 Without New Evaluation

First implement:

```text
scripts/pas_analyze_local_selection_downstream.py
```

Inputs:

```text
--local-delta-table /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025/local_delta_scores.csv
--downstream-summary /workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv
--stress-table /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv
--output-dir /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis
```

The script should join by `candidate_id`.

### Selection Rules To Compare

For each fixed endpoint-compatible set:

```text
top_m by L30: m = 2, 5, 8, 13
epsilon by L30: eps = 0.02, 0.05, 0.10
all candidates
```

Compute selected candidates for:

```text
FF-Endpoint: min L30
PAS-S30.25: min S3025
PAS-S30.50: min S3050
PAS-S35: min S35
Oracle-Downstream30: max avg_pruned_score  # analysis only, not a real method
Oracle-L40: min L40                         # analysis only
```

For every selected rule, report:

```text
rule,selection_scope,scope_size,candidate_id,
L30,S3025,S3050,S35,L40,Regret40,Delta40,
avg_pruned_score,task_count,
piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq
```

If task columns have different exact names, include all task score columns that
exist in the downstream summary.

### Correlations To Compute

For each scope above, compute Pearson, Spearman, and partial correlation
controlling `L30` for:

```text
S3025 -> avg_pruned_score
S3050 -> avg_pruned_score
S35   -> avg_pruned_score
S3025 -> L40
S3050 -> L40
S35   -> L40
S3025 -> Regret40
S3050 -> Regret40
S35   -> Regret40
```

For downstream, higher is better. For PPL/log-loss/regret, lower is better.
Make the direction explicit in the output.

### Required P0 Outputs

```text
local_selection_downstream_table.csv
local_selection_downstream_table.md
local_signal_correlation.csv
local_signal_correlation.md
local_selection_manifest.json
```

### Observed P0 Result

Recorded on 2026-05-22 for `OPT-2.7B` / `WikiText-2` / seed `3025`.

Artifacts:

```text
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_selection_downstream_table.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_selection_downstream_table.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_signal_correlation.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_signal_correlation.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_selection_manifest.json
```

Selection-level observations:

| Scope | FF-Endpoint downstream | PAS-S30.25 downstream | PAS-S30.50 downstream | PAS-S35 downstream | Reading |
| --- | --- | --- | --- | --- | --- |
| `top_m=2` | `0.343333` | `0.343333` | `0.343333` | `0.446667` | Tight top-2 scope: local probes collapse to endpoint; S35 selects the downstream/L40 oracle candidate. |
| `top_m=5` | `0.343333` | `0.426667` | `0.426667` | `0.468333` | Local probes improve over endpoint, but S35 matches the downstream oracle. |
| `top_m=8` | `0.343333` | `0.395000` | `0.426667` | `0.401667` | Local probes improve over endpoint, with S30.50 better than S30.25/S35 here. |
| `top_m=13` | `0.343333` | `0.395000` | `0.426667` | `0.401667` | Same pattern as top-8. |
| `epsilon=0.02` | `0.343333` | `0.343333` | `0.343333` | `0.343333` | Scope size is 1, so no selection difference. |
| `epsilon=0.05` | `0.343333` | `0.343333` | `0.343333` | `0.343333` | Scope size is 1, so no selection difference. |
| `epsilon=0.10` | `0.343333` | `0.343333` | `0.343333` | `0.446667` | Local probes still collapse to endpoint; S35 selects the stress/downstream oracle candidate. |
| `all_candidates` | `0.343333` | `0.385000` | `0.385000` | `0.385000` | Without an endpoint-compatible scope, all slope rules select the same negative-slope outlier. |

Correlation observations:

| Scope | Metric | Value | Reading |
| --- | --- | --- | --- |
| all candidates | `partial_corr(S3025,avg_pruned_score|L30)` | `0.217310` | Wrong sign for a local-fragility story if higher slope means worse downstream. |
| all candidates | `partial_corr(S3050,avg_pruned_score|L30)` | `-0.007190` | Essentially no controlled downstream relation. |
| all candidates | `partial_corr(S35,avg_pruned_score|L30)` | `-0.200453` | S35 also does not explain downstream@30 globally. |
| all candidates | `partial_corr(S35,L40|L30)` | `0.782805` | S35 remains a strong stricter-budget stress signal. |
| all candidates | `partial_corr(S35,Regret40|L30)` | `0.782805` | Same stress-regret signal. |

Gate result: this is **mixed/ambiguous**, not positive for a
local-PAS/downstream claim. The local rules do select better-than-endpoint
downstream candidates in several `top_m` endpoint-compatible scopes, but the
corresponding controlled local-slope/downstream relation is weak, unstable, or
opposite the intended local-flatness direction. Do not claim local PAS predicts
downstream@30 from `S3025/S3050`. At most, this motivates an optional `S31`
probe if one more cheap PPL-only check is desired.

## P1: Optional L31 Probe

Only run this if P0 suggests S3025/S3050 are promising or ambiguous.

If needed, extend the local delta probe to include:

```text
L31 = L(31.00%)
S31 = L31 - L30
```

Then rerun the analysis including:

```text
PAS-S31: min S31
S31 -> avg_pruned_score
S31 -> L40 / Regret40
```

Do not run L31 before P0 analysis unless implementation cost is trivial.

P1 is now allowed because P0 is mixed/ambiguous. Run a single PPL probe centered
at `30.50` with radius `0.50`:

```text
probe_sparsity = 0.3050
ew_delta       = 0.0050
```

This produces `L30`, `L30.50`, and `L31.00` in one pass. Merge the result with
the existing `30.25/30.50` local table before rerunning the local-selection
analysis.

Also audit the negative local-slope cases against projection nestedness:

```text
scripts/pas_audit_local_delta_nestedness.py
```

Use the existing P0 projection sweep artifacts:

```text
/workspace/ckpts/pas_nested_projection_check/opt27b_seed3025_delta_sweep/nestedness_by_candidate.csv
/workspace/ckpts/pas_nested_projection_check/opt27b_seed3025_delta_sweep/nestedness_violations.csv
/workspace/ckpts/pas_nested_projection_check/opt27b_seed3025_delta_sweep/mask_change_by_candidate.csv
```

This directly tests whether negative `S3025/S3050/S31` examples are driven by
non-nested re-additions or other large mask jumps.

### Observed P1 Result

Recorded on 2026-05-23 for the optional `S31` PPL probe and nestedness audit.

Artifacts:

```text
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050/local_delta_scores_with_s31.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_analysis/local_selection_downstream_table.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_analysis/local_selection_downstream_table.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_analysis/local_signal_correlation.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_analysis/local_signal_correlation.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nestedness_audit/local_delta_negative_cases.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nestedness_audit/local_delta_nestedness_summary.csv
```

Selection-level update:

| Scope | FF-Endpoint | PAS-S30.50 | PAS-S31 | PAS-S35 | Reading |
| --- | --- | --- | --- | --- | --- |
| `top_m=2` | `0.343333` | `0.343333` | `0.343333` | `0.446667` | Local probes collapse to endpoint. |
| `top_m=5` | `0.343333` | `0.426667` | `0.426667` | `0.468333` | `S31` selects the same candidate as `S30.50`. |
| `top_m=8` | `0.343333` | `0.426667` | `0.426667` | `0.401667` | `S31` again matches `S30.50`, above endpoint and S35 but below downstream oracle. |
| `top_m=13` | `0.343333` | `0.426667` | `0.426667` | `0.401667` | Same as top-8. |
| `epsilon=0.02/0.05` | `0.343333` | `0.343333` | `0.343333` | `0.343333` | Scope size is 1. |
| `epsilon=0.10` | `0.343333` | `0.343333` | `0.343333` | `0.446667` | Local probes still collapse to endpoint. |
| `all_candidates` | `0.343333` | `0.385000` | `0.385000` | `0.385000` | All slope rules select the same negative-slope outlier. |

Signal update:

| Scope | Metric | Value | Reading |
| --- | --- | --- | --- |
| all candidates | `partial_corr(S31,avg_pruned_score|L30)` | `0.039776` | Essentially no controlled downstream relation. |
| all candidates | `partial_corr(S31,L40|L30)` | `0.043607` | No useful stricter-budget stress relation. |
| all candidates | `partial_corr(S31,Regret40|L30)` | `0.043607` | Same. |
| all candidates | `partial_corr(S35,L40|L30)` | `0.782805` | The original `S35` stress signal remains much stronger. |
| all candidates | `partial_corr(S35,Regret40|L30)` | `0.782805` | Same. |

Nestedness audit:

| Slope | Group | n | Fraction with violation | Avg added dims | Avg removed dims | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `S3025` | negative | `6` | `0.333333` | `24.1667` | `773.167` | Some negative slopes are contaminated by re-additions, but most are pure removals. |
| `S3050` | negative | `6` | `0.500000` | `160.667` | `1634.33` | Non-nested contamination is more visible at this radius. |
| `S31` | negative | `6` | `0.500000` | `136.833` | `3719.83` | Half of negative `S31` examples have violations; half do not. |
| `S31` | nonnegative | `14` | `0.428571` | `120.143` | `3721.14` | Violation rate and added dims are close to the negative group. |

P1 gate result: `S31` does **not** resolve the ambiguity. It preserves the
selection-level phenomenon (`PAS-S31` matches `PAS-S30.50` in useful `top_m`
scopes), but it does not produce a stable local-flatness signal for
downstream@30 or stress. Current-projector non-nested re-additions explain some
negative-slope cases, especially for `S3050/S31`, but not all of them. Pure
removal can also lower PPL, so the negative slopes cannot be dismissed as only
a nestedness bug.

If the team wants to rescue a strict local-flatness interpretation, the next
experiment should evaluate a strictly nested projector path. Do not run more
downstream tasks before that; the current-projector local probes have reached
their evidential limit.

## P2: Strictly Nested Projector PPL Probe

This is a projector/evaluation change only. Do not retrain FastForward
candidates.

The historical projector evaluates every sparsity independently and may expand
modules back toward the global budget. For a local-flatness test, use:

```text
--projection-mode nested_from_base
--projection-base-sparsity 0.3000
```

This first projects the candidate at `30%`, then walks stricter budgets in
ascending sparsity and caps each module by the previous projection's preserved
dimension. It can redistribute within the surviving `30%` structure to hit the
next budget, but it cannot re-add heads/channels/neurons removed at an earlier
point.

Run the smallest useful nested probe first:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

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

Materialize the local-slope table:

```bash
OUT=/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested

python - <<'PY'
import csv
from pathlib import Path

out = Path("/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested")
rows = []
with (out / "probe_results.csv").open("r", encoding="utf-8") as handle:
    for r in csv.DictReader(handle):
        L30 = float(r["ell_minus"])
        L3025 = float(r["ell_0"])
        L3050 = float(r["ell_plus"])
        rows.append({
            "candidate_id": r["candidate_id"],
            "L30": L30,
            "L3025": L3025,
            "L3050": L3050,
            "S3025": L3025 - L30,
            "S3050": L3050 - L30,
            "PPL30": r.get("ppl_minus", ""),
            "PPL3025": r.get("ppl_0", ""),
            "PPL3050": r.get("ppl_plus", ""),
            "projection_mode": r.get("projection_mode", ""),
            "projection_base_sparsity": r.get("projection_base_sparsity", ""),
        })

with (out / "local_delta_scores.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out / 'local_delta_scores.csv'}")
PY
```

Then rerun the existing analysis stack against the nested table:

```bash
python scripts/pas_analyze_local_delta_probe.py \
  --local-delta-csv /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested/local_delta_scores.csv \
  --stress-table /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv \
  --downstream-summary /workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested

python scripts/pas_analyze_local_selection_downstream.py \
  --local-delta-table /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested/local_delta_scores.csv \
  --downstream-summary /workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv \
  --stress-table /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested_analysis
```

If `S3025/S3050` remain negative or uncorrelated under this strict path, stop
the local-flatness rescue. If the nested path changes the conclusion, run the
same command with `--probe-sparsity 0.3050 --delta 0.0050` to get nested `S31`.

### Observed Strict-Nested Result

Recorded on 2026-05-23 for the strict nested projector follow-up.

Artifacts:

```text
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested/local_delta_scores.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_nested/local_delta_analysis.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nested/local_delta_scores_with_s31.csv
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nested_analysis/local_signal_correlation.md
/workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0050_nested_analysis/local_selection_downstream_table.md
```

Strict nested projection sharply reduces current-projector artifacts: negative
small-radius slopes drop from `S3025=6/S3050=6` to `S3025=2/S3050=2`.

All-candidate downstream@30 controlled signal:

| Predictor | Partial corr with `avg_pruned_score` controlling `L30` | Reading |
| --- | --- | --- |
| `S3025` | `-0.225739` | Correct direction, weak. |
| `S3050` | `-0.331522` | Correct direction, strongest small-delta signal. |
| `S31` | `-0.159325` | Correct direction but weaker/noisier than `S3050`. |
| `S35` | `-0.200453` | Not a clean downstream/local-flatness signal. |

Stress-budget controlled signal remains separated:

| Predictor | Target | Partial corr controlling `L30` | Reading |
| --- | --- | --- | --- |
| `S3050` | `L40`/`Regret40` | about `0.12` | Weak cross-budget stress signal. |
| `S31` | `L40`/`Regret40` | about `0.08` | Weak cross-budget stress signal. |
| `S35` | `L40`/`Regret40` | `0.782805` | Strong stress-budget signal. |

Selection-level result:

| Scope | FF-Endpoint | PAS-S30.25/S30.50/S31 | PAS-S35 | Downstream oracle | Reading |
| --- | --- | --- | --- | --- | --- |
| `top_m=2` | `0.343333` | `0.343333` | `0.446667` | `0.446667` | Local probes collapse to endpoint in the tightest scope. |
| `top_m=5` | `0.343333` | `0.426667` | `0.468333` | `0.468333` | Local probes improve over endpoint but trail S35/oracle. |
| `top_m=8/13` | `0.343333` | `0.426667` | `0.401667` | `0.468333` | Local probes improve over endpoint and S35, but not oracle. |

Interpretation gate: strict nested small-delta probes provide evidence for a
local fragility signal relevant to downstream@30, strongest at `S3050`.
`S31` is a weaker confirmation, not a stronger one. Keep `S35` as a separate
cross-budget compression-stress signal rather than a local-flatness proxy.

## Server Commands

P0 analysis:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

python scripts/pas_analyze_local_selection_downstream.py \
  --local-delta-table /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025/local_delta_scores.csv \
  --downstream-summary /workspace/ckpts/pas_stress_recovery/downstream_candidate_summary_opt27b_seed3025.csv \
  --stress-table /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv \
  --output-dir /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis

cat /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_selection_downstream_table.md
cat /workspace/ckpts/pas_local_delta_probe/opt27b_seed3025_delta0025_analysis/local_signal_correlation.md
```

## Interpretation Gate

Positive for local-PAS/downstream story only if:

```text
PAS-S30.25 or PAS-S30.50 selects a candidate with better downstream@30 than
FF-Endpoint under a fixed endpoint-compatible scope, and the corresponding
local slope has a meaningful controlled relation with downstream@30.
```

If this does not happen:

```text
Do not claim PAS-local predicts downstream@30. Keep SAM/flatness as motivation
only, not as a main contribution.
```

If S35 remains strong only for L40/Regret40:

```text
PAS-S35 remains a stress-test selection rule for stricter-budget compression,
not a local-flatness generalization rule.
```

Please commit and push the analysis script and this doc. Do not commit large
server artifacts.
