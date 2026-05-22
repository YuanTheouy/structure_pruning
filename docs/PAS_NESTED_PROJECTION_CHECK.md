# PAS Nested Projection Check

Date: 2026-05-22

Purpose: verify whether PAS's same-vector budget path is a true nested
compression path, or only a counterfactual same-vector projection path.

## Why This Matters

The current PAS story compares the same candidate priority vector at nearby
sparsity budgets:

```text
A -> F(A, 0.30), F(A, 0.35), F(A, 0.40)
```

For a SAM/flatness-style interpretation, the stricter budget should ideally be
a local nested perturbation of the nominal checkpoint:

```text
F(A, 0.35) subseteq F(A, 0.30)
```

If this does not hold, then `30% -> 35%` should be described as a same-vector
counterfactual projection, not as "continue pruning the 30% model".

## Task For Experiment Execution Chat

Please implement and run a server-side diagnostic script:

```text
scripts/pas_check_projection_nestedness.py
```

The script should not change model checkpoints. It only loads candidate
priority vectors, reconstructs the projected policies at multiple sparsities,
and reports whether the resulting per-module preserved dimensions are nested.

## Required Inputs

Use the existing seed3025 pool first:

```text
--model /workspace/Models/opt-2.7b
--model-name opt-2.7b
--dataset wikitext2
--seed 3025
--candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates
--sparsities 0.30,0.31,0.32,0.33,0.35,0.40
--output-dir /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025
```

Then optionally repeat on:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
```

## What To Check

For every candidate and every adjacent sparsity pair `sigma_a < sigma_b`:

1. Compute projected policy using the same function used by PAS:
   `project_candidate_score_with_metadata(env, score_vector, sigma)`.
2. Convert each module preserve ratio to integer preserved dimension:
   `d_i(sigma) = round(policy_i(sigma) * dim_i)`.
3. Check per-module monotonicity:

```text
d_i(sigma_b) <= d_i(sigma_a)
```

4. Since the actual pruning code keeps top-`d_i` entries by the same
   `A_metric` ranking, per-module monotonicity is enough to imply nested
   retained sets for that module. If any module has `d_i(sigma_b) > d_i(sigma_a)`,
   record it as a nestedness violation.

## Required Outputs

Write these artifacts:

```text
nestedness_summary.csv
nestedness_violations.csv
nestedness_by_candidate.csv
mask_change_summary.csv
mask_change_by_candidate.csv
nestedness_manifest.json
```

### `nestedness_summary.csv`

Columns:

```text
model,dataset,seed,candidate_pool,num_candidates,sigma_a,sigma_b,
num_candidate_pairs,num_pairs_with_violation,total_violation_modules,
max_dimension_increase,max_relative_dimension_increase,
all_pairs_nested
```

### `nestedness_violations.csv`

One row per violating module:

```text
candidate_id,sigma_a,sigma_b,module_index,module_type,layer_index,
dim,d_a,d_b,dimension_increase,relative_dimension_increase,
policy_a,policy_b,actual_sparsity_a,actual_sparsity_b,
budget_error_a,budget_error_b
```

### `nestedness_by_candidate.csv`

One row per candidate and sparsity pair:

```text
candidate_id,sigma_a,sigma_b,has_violation,num_violation_modules,
max_dimension_increase,total_dimension_increase,
actual_sparsity_a,actual_sparsity_b
```

### `mask_change_summary.csv`

One row per adjacent sparsity pair:

```text
model,dataset,seed,candidate_pool,num_candidates,sigma_a,sigma_b,delta_sigma,
num_candidate_pairs,num_candidates_with_any_change,
fraction_candidates_with_any_change,
avg_changed_modules,max_changed_modules,
avg_abs_dimension_change,max_abs_dimension_change,
avg_removed_dimensions,max_removed_dimensions,
avg_added_dimensions,max_added_dimensions,
avg_changed_head_modules,max_changed_head_modules,
avg_changed_ffn_modules,max_changed_ffn_modules,
avg_changed_head_dimensions,max_changed_head_dimensions,
avg_changed_ffn_dimensions,max_changed_ffn_dimensions,
too_small_to_generate_signal,all_pairs_nested,projection_mode,base_sigma
```

### `mask_change_by_candidate.csv`

One row per candidate and adjacent sparsity pair:

```text
candidate_id,sigma_a,sigma_b,has_any_change,
num_changed_modules,num_changed_head_modules,num_changed_ffn_modules,
total_abs_dimension_change,total_removed_dimensions,total_added_dimensions,
total_changed_head_dimensions,total_changed_ffn_dimensions,
max_abs_dimension_change,actual_sparsity_a,actual_sparsity_b,projection_mode
```

## Suggested Implementation Notes

Prefer reusing existing PAS helpers instead of duplicating projection logic:

- `amc_searchPPO.py::project_candidate_score_with_metadata`
- `amc_searchPPO.py::load_candidate_score`
- `amc_searchPPO.py::build_env_module_costs`
- `lib.ew_projector.project_score_to_policy`

The script may instantiate the same environment as probe/compile jobs to get
the real `dim_list`, `param_list`, `norm_para`, `channel_round`, and bounds.
This is acceptable on the server. Do not approximate these locally.

## Interpretation Rules

If no or near-zero violations:

```text
We can describe PAS local probes as approximately nested same-vector
compression paths.
```

If violations exist but are very small:

```text
We should report nestedness as empirical/near-nested, and avoid hard set
inclusion language.
```

If violations are common or large:

```text
We must describe PAS as same-vector counterfactual budget projection, not as
sequentially continuing to prune the 30% checkpoint.
```

## Optional Follow-Up: Nested Projector Variant

If current projection is not nested, implement a second script mode:

```text
--projection-mode nested_from_base
--base-sigma 0.30
```

This mode should:

1. First compute `F(A, base_sigma)`.
2. For stricter budgets, only reduce per-module `d_i`; never allow any
   `d_i(sigma_b) > d_i(base_sigma)`.
3. Keep the same within-module top-ranking rule, so retained channel/head sets
   are true subsets.
4. Evaluate whether `S31/S32/S35` under this nested projector better predicts
   downstream@30 or held-out `L40`.

Do not replace the main PAS result with this variant unless it improves both
scientific clarity and evidence.

## Server Commands

Implemented script:

```text
scripts/pas_check_projection_nestedness.py
```

It supports two modes:

```text
--projection-mode current
--projection-mode nested_from_base --base-sigma 0.30
```

Use `current` first to audit the PAS projector that produced the current
artifacts. Use `nested_from_base` only if `current` reports violations, to
materialize the strict repair path. In `nested_from_base`, the base `0.30`
projection initializes the retained dimensions, and every later stricter budget
is capped by the immediately previous projection. This cascading cap is required
for adjacent nestedness:

```text
d_i(0.3025) <= d_i(0.3000)
d_i(0.3050) <= d_i(0.3025)
d_i(0.3100) <= d_i(0.3050)
...
```

After implementation, run:

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main

python scripts/pas_check_projection_nestedness.py \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --sparsities 0.30,0.31,0.32,0.33,0.35,0.40 \
  --output-dir /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025

cat /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025/nestedness_summary.csv
cat /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025/mask_change_summary.csv
cat /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025/nestedness_violations.csv | head -n 30
```

If violations appear, immediately run the strict nested repair audit:

```bash
python scripts/pas_check_projection_nestedness.py \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --sparsities 0.30,0.31,0.32,0.33,0.35,0.40 \
  --projection-mode nested_from_base \
  --base-sigma 0.30 \
  --output-dir /workspace/ckpts/pas_nested_projection_check/opt27b_seed3025_nested_from_base
```

Please commit and push only the script and this doc update. Do not commit large
server artifacts.
