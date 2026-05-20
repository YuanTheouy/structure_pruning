# PAS Next-Round Tasks

Date: 2026-05-20

This document supersedes older "early-warning reranking" task prompts for the current FastForward paper direction.

Current paper draft:

```text
/Users/bytedance/Documents/AutoScience/FastForward/Compression-Resilient FastForward Pruning via Early-Warning Signals for Static LLM Compression/pas_revised_draft.tex
```

Current method name:

```text
Path-Aware Selection (PAS)
```

## Shared Core

The paper is not about:

- proving RL search is optimal;
- proposing a new compensation/calibration method;
- discovering LLM compression phase transitions;
- introducing runtime dynamic sparsity;
- claiming a new full pruning framework.

The current core is:

> FastForward-style search produces budget-transferable priority-vector candidates. The same priority vector can be projected to neighboring exact budgets, inducing a same-vector local budget path. PAS first forms an endpoint-competitive shortlist at the target sparsity, then uses a local path warning score to choose a less fragile policy before final compensation. The deployed checkpoint remains a normal static exact-budget pruned model.

## Chat B: Experiment Execution Chat

Role: code/script/runbook executor. Do not make broad paper claims.

Important constraint:

All GPUs, model weights, and datasets are on the server. The experiment execution chat should modify code, prepare scripts, make runs reproducible, commit/push, and report artifacts. It should not treat missing local GPU/model/data as failure.

### Immediate Goal

Make the PAS evidence pipeline server-ready and aligned with `pas_revised_draft.tex`.

### Required Evidence

Use the same FastForward candidate pool for all controlled rules.

For target sparsity:

```text
sigma = 0.30
delta = 0.05
probe budgets = 0.25 / 0.30 / 0.35
held-out future budget = 0.40
```

The held-out `0.40` point must not be used for:

- candidate selection;
- tuning `K_s` or `epsilon`;
- choosing the warning score;
- filtering candidates;
- threshold choice;
- early stopping.

It is only for analysis and final reporting.

### Controlled Selection Rules

Implement/report, all on the same candidate pool:

- `FF-Endpoint`: lowest `ell_0`.
- `Random-shortlist`: random candidate from endpoint-competitive shortlist; report mean/std over repeated draws.
- `PAS-Plus`: shortlist, then choose by `ell_plus`.
- `PAS-Slope`: shortlist, then choose by `(ell_plus - ell_0) / delta`.
- `PAS-Curv`: shortlist, then choose by `(ell_plus - 2*ell_0 + ell_minus) / delta^2`.
- `Oracle-heldout`: best `ell_h` at 0.40, analysis only.

### Required Scripts/Artifacts

Prepare or update scripts so the server can produce:

- `probe_results.csv`: per-candidate `ell_minus`, `ell_0`, `ell_plus`, `ppl_minus`, `ppl_0`, `ppl_plus`.
- `heldout_results.csv`: per-candidate `ell_h` and `ppl_h` at 0.40.
- `warning_correlation.csv`: correlations for `ell_plus`, slope, curvature vs held-out degradation.
- `selection_regret.csv`: chosen candidate and regret for each selection rule.
- `selected_candidates.json`: selected candidates for FF-Endpoint, PAS-Plus, PAS-Slope, PAS-Curv, Oracle-heldout.
- figures:
  - `path_divergence.pdf`
  - `endpoint_ambiguity_scatter.pdf`
  - `warning_correlation.pdf`
- an artifact manifest recording model, dataset, seed, sigma, delta, top_k, shortlist rule, probe samples, held-out samples, timestamp, and command.

### Suggested Server Run Shape

Use or adapt existing scripts:

```bash
export MODEL=/workspace/Models/opt-2.7b
export MODEL_NAME=opt-2.7b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts

export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=5000
unset EPISODES_PER_WORKER
export N_SAMPLES=32
export TOP_K=20
export NUM_COLLECT=15
export LEARNING_EPOCH=10
export SEED=2025

bash scripts/run_p0_candidate_search_multigpu.sh
bash scripts/run_ew_p0_multigpu.sh
```

Then run or create a PAS analysis command that evaluates held-out `0.40`, computes correlations, selection regret, and plots.

### Return Requirements

After changes:

- run `python -m py_compile` on modified Python files;
- commit and push;
- report commit hash;
- list exact server commands;
- list exact expected artifact paths;
- report blockers if a required script or resource is missing.

Do not edit the LaTeX manuscript and do not invent results.

## Chat A: Novelty / Draft Critic Chat

Role: literature, novelty, and reviewer-risk critic. Do not run code.

### Immediate Goal

Review `pas_revised_draft.tex` as the current main draft. Judge whether PAS is a defensible new story as of 2026-05-20.

### Must Read

```text
/Users/bytedance/Documents/AutoScience/FastForward/Compression-Resilient FastForward Pruning via Early-Warning Signals for Static LLM Compression/pas_revised_draft.tex
/Users/bytedance/Documents/AutoScience/FastForward/structure_pruning/docs/CHATGPT_PRO_BRIEF.md
/Users/bytedance/Documents/AutoScience/FastForward/structure_pruning/docs/NOVELTY_REVIEW.md
```

### Questions To Answer

1. Is "same-vector local budget path for budget-transferable priority-vector candidates" meaningfully distinct from:
   - GISP iterative pruning / nested subnetworks;
   - GRASPrune global budgeted gates;
   - Týr-the-Pruner coarse-to-fine or multi-sparsity search;
   - phase-transition/compression-collapse analysis;
   - generic multi-budget evaluation;
   - LoRA/calibration-based recovery methods?

2. Is the current title good?

   Current:

   ```text
   Path-Aware Policy Selection for Budget-Transferable Structured LLM Pruning
   ```

   Please suggest safer alternatives if this overclaims.

3. Does the compensation framing help or distract?

   Current stance:

   ```text
   PAS changes only the selected structure before compensation. Compensation is aligned across FF-Endpoint and PAS variants and is not the contribution.
   ```

4. Which claims are safe, and which should be weakened?

5. What are the top five likely reviewer objections?

6. What exact experiments/figures would make the PAS story convincing?

### Required Output

Append or create a review note with:

- novelty matrix;
- top reviewer objections;
- safe contribution wording;
- claims to delete/weaken;
- missing 2025-2026 citations;
- go/no-go recommendation for PAS framing.

Do not repeat old generic pruning summaries. Focus on PAS and current 2026 literature.

## Coordinator Gate

Do not fill manuscript tables until:

- Chat B produces raw CSV/JSON artifacts;
- held-out 0.40 is separated from PAS selection;
- same-pool endpoint/PAS-Plus/PAS-Slope/PAS-Curv/Oracle comparisons exist;
- compensation alignment is explicit;
- Chat A says the narrowed PAS claim is defensible or tells us exactly how to weaken it.

## Next Server Runs After Two-Pool P0

Status on 2026-05-20:

- Two OPT-2.7B/WikiText-2 PAS P0 pools exist.
- `PAS-Slope` is the current primary rule: it reaches held-out oracle regret `0.0` in both pools.
- `PAS-Plus` does not improve over FF-Endpoint in either pool.
- `PAS-Curv` is mixed and should stay an ablation.
- Do not headline curvature unless later evidence changes this.

### P1. High-Sample Recheck For Seed3025 Selected Candidates

Purpose:

- Confirm the second pool result is not a small held-out sample artifact.
- Do not change the selected candidates or selection rule.
- Re-evaluate only candidates already selected by FF-Endpoint, PAS-Plus, PAS-Slope, PAS-Curv, and Oracle-heldout.

Run on server:

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

python pas_selected_heldout_recheck.py \
  --selected_candidates_json /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selected_candidates.json \
  --model /workspace/Models/opt-2.7b \
  --model_name opt-2.7b \
  --future_sparsity 0.40 \
  --num_samples 64 \
  --batch_size 50 \
  --seed 3025 \
  --output_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64
```

Expected artifacts:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck.csv
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_regret.csv
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_manifest.json
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_commands.sh
```

Success condition:

- PAS-Slope selected candidate remains better than FF-Endpoint at held-out `0.40` under `num_samples=64`.
- If this fails, keep the P0 result as inconclusive and do not fill manuscript tables from the earlier 32-sample held-out result.

### P2. Cross-Model PAS Pilot

Purpose:

- Avoid a paper story supported only by OPT-2.7B.
- The minimum acceptable expansion is one additional model or model-size setting.
- Prefer a cheaper setting first if server time is tight.

Preferred order:

1. `OPT-1.3B` if available, because it is cheaper and tests size transfer inside OPT.
2. `LLaMA2-7B` or `LLaMA-7B` if available, because it tests a different model family.
3. `Mistral-7B` only if resources and scripts are already stable.

#### P2a. OPT-1.3B PAS Pilot

Adjust `MODEL` if the server path differs.

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

export MODEL=/workspace/Models/opt-1.3b
export MODEL_NAME=opt-1.3b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts

export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=400
export N_SAMPLES=16
export TOP_K=20
export SHORTLIST_SIZE=2
export RANDOM_REPEATS=500
export NUM_COLLECT=5
export LEARNING_EPOCH=3
export HELDOUT_N_SAMPLES=32
export SEED=2025

export RUN_ID_PREFIX=p0_candidates_opt13b_seed2025
export MERGED_CANDIDATE_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates
export CANDIDATE_DIR=$MERGED_CANDIDATE_DIR
export EW_OUTPUT_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_ew_seed2025
export PAS_OUTPUT_DIR=/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025

export RUN_CANDIDATE_SEARCH=true
export RUN_PROBE=true
export RUN_PAS=true

bash scripts/run_pas_p0_server.sh
```

Expected artifacts:

```text
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/artifact_manifest.json
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/selection_regret.csv
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/warning_correlation.csv
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/pas_joined_probe_heldout.csv
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/path_divergence.pdf
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/endpoint_ambiguity_scatter.pdf
/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/warning_correlation.pdf
```

#### P2b. LLaMA/LLaMA2-7B PAS Pilot

Run only after confirming the model path on the server. Replace `MODEL` and `MODEL_NAME` with the actual path/name.

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

export MODEL=/workspace/Models/Llama-2-7b-hf
export MODEL_NAME=llama2-7b
export WIKITEXT2_PATH=/workspace/datasets/wikitext/wikitext-2-raw-v1
export CKPT_ROOT=/workspace/ckpts

export GPU_IDS="0 1 2 3 4 5 6 7"
export TARGET_SPARSITY=0.30
export DELTA=0.05
export TRAIN_EPISODES=300
export N_SAMPLES=16
export TOP_K=20
export SHORTLIST_SIZE=2
export RANDOM_REPEATS=500
export NUM_COLLECT=5
export LEARNING_EPOCH=3
export HELDOUT_N_SAMPLES=32
export SEED=2025

export RUN_ID_PREFIX=p0_candidates_llama2_7b_seed2025
export MERGED_CANDIDATE_DIR=/workspace/ckpts/llama2-7b/sparsity_0.30/p0_candidates_seed2025/candidates
export CANDIDATE_DIR=$MERGED_CANDIDATE_DIR
export EW_OUTPUT_DIR=/workspace/ckpts/llama2-7b/sparsity_0.30/p0_ew_seed2025
export PAS_OUTPUT_DIR=/workspace/ckpts/llama2-7b/sparsity_0.30/p0_pas_seed2025

export RUN_CANDIDATE_SEARCH=true
export RUN_PROBE=true
export RUN_PAS=true

bash scripts/run_pas_p0_server.sh
```

If LLaMA2 model path is unavailable or incompatible, report the blocker and run P2a instead.

### P3. Compensation-Aligned Final Evaluation For Selected Policies

Purpose:

- Show that PAS changes only the selected structure before compensation.
- Keep recovery identical between FF-Endpoint and PAS-Slope.
- Do this only after P1 confirms the selected candidates at higher held-out samples.

Minimum controlled comparison:

```text
FF-Endpoint selected policy vs PAS-Slope selected policy
same target sparsity = 0.30
same model = OPT-2.7B
same recovery/calibration setting
same evaluation samples/tasks
```

If Ridge calibration is used, use it for both selected policies. If no calibration is used, use no calibration for both.

Chat B should either:

- prepare a script that compiles/evaluates both selected policies under identical recovery settings; or
- report a blocker explaining which compile/calibration path is missing.

Do not compare PAS-Slope with calibration against FF-Endpoint without calibration.

Prepared server command:

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

python pas_compile_selected_final.py \
  --selected_candidates_json /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selected_candidates.json \
  --model /workspace/Models/opt-2.7b \
  --model_name opt-2.7b \
  --target_sparsity 0.30 \
  --rules FF-Endpoint,PAS-Slope \
  --num_samples 64 \
  --batch_size 50 \
  --seed 3025 \
  --output_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon
```

Expected artifacts:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_eval.csv
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_manifest.json
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_commands.sh
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/final_eval/
```

### P4. Overhead Summary

Purpose:

- Support the claim: PAS adds selection-stage probe cost only and no inference-time overhead.

Required table fields:

```text
candidate search cost
PAS probe/held-out analysis cost
selected-candidate recheck cost
compensation cost, if any
final checkpoint type
inference-time overhead
```

If exact GPU-hours are not available, report wall-clock seconds from artifact manifests and mark GPU-hour accounting as missing.

Prepared server command:

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

python summarize_overhead.py \
  --pas_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025 \
  --recheck_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64 \
  --final_eval_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon \
  --output_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead
```

Expected artifacts:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.csv
/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.json
```

### What To Report Back

For each run, Chat B should update `docs/CLAIM_EVIDENCE.md` and `docs/PROJECT_PLAN.md` with:

- exact command;
- commit hash;
- model/dataset;
- seed;
- target/probe/held-out sparsities;
- candidate count and shortlist rule;
- artifact paths;
- key numeric rows from `selection_regret.csv`, `warning_correlation.csv`, and recheck regret CSV;
- explicit statement that `0.40` was analysis-only and not used for selection.

Do not edit manuscript tables until the coordinator reviews these summaries.
