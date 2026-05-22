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
