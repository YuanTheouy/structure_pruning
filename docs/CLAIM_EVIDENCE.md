# FastForward PAS-Slope Claim-Evidence Checklist

| Claim | Evidence Needed | Current Asset | Gap | Next Action |
| --- | --- | --- | --- | --- |
| Endpoint-similar pruning candidates can have divergent future compression paths. | Candidate path CSV across sparsities, plus `path_divergence.pdf`. | Two P0 pools now have `path_divergence.pdf`: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/path_divergence.pdf` and `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/path_divergence.pdf`. | Need figure inspection before manuscript use. | Inspect figures and keep the claim scoped to same-pool P0 evidence. |
| Local budget sensitivity predicts future degradation. | Pearson/Spearman between PAS-Slope from `0.30/0.35` and separated future target `logPPL(0.40)-logPPL(0.30)`. | Two P0 pools show slope Spearman `0.8030075187969924` and `0.8285714285714285`. | Evidence is still OPT-2.7B/WikiText-2 P0 only. | Use PAS-Slope as the primary PAS score for the current paper table staging. |
| Curvature is a useful but not primary warning score. | Pearson/Spearman for curvature from `0.25/0.30/0.35` against separated future target. | Two P0 pools show curvature Spearman `0.5969924812030075` and `0.41052631578947363`; PAS-Curv succeeds in the first pool but fails in seed `3025`. | Curvature is weaker and mixed relative to slope. | Keep PAS-Curv as an ablation, not the headline selection rule. |
| PAS-Slope selection improves held-out stricter-budget stability over endpoint-only selection in tested pools. | Same-pool selection regret table: FF-Endpoint, PAS-Plus, PAS-Slope, PAS-Curv, Oracle-heldout. | Two P0 pools show PAS-Slope regret `0.0` in both; FF-Endpoint regret is `0.27734373462907325` and `0.5581475044600985`. | Need broader model/dataset evidence before broad claims. | Fill P0 table staging, but phrase final paper claims as P0 evidence unless expanded. |
| Gains come from local budget sensitivity, not merely extra probe evaluations. | Ablation table: endpoint, plus-probe, slope, curvature, oracle. | Two P0 pools have comparable `selection_regret.csv` artifacts under `p0_pas` and `p0_pas_seed3025`. | Need final table formatting and optional selected-candidate high-sample recheck for seed `3025`. | Use PAS-Plus as the probe-only control and PAS-Slope as the primary selection score. |
| PAS adds selection-stage cost but no inference-time overhead. | Runtime summary with search GPU-hours, probe cost, calibration cost, and final checkpoint type. | Smoke probe row records `eval_seconds_total=144.03351759910583` for best candidate; scripts record per-shard logs under `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/shards/`. | Need full run-level overhead summary and final checkpoint metadata. | Run `summarize_overhead.py` after larger candidate probe/evaluation. |
| The narrowed contribution is novel enough under 2026 related work. | Independent novelty matrix and reviewer-risk review with 2025-2026 pruning/phase-transition work. | `docs/NOVELTY_REVIEW.md` completed on 2026-05-19. | Need to ensure final manuscript follows the narrowed wording and cites GISP, GRASPrune, Olica, Týr-the-Pruner, and phase-transition work correctly. | Keep contribution wording as same-vector local-budget-sensitivity candidate selection; do not broaden to generic structured pruning. |

## Evidence Discipline

Do not fill manuscript high-sparsity or correlation claims from intuition. A claim becomes manuscript-ready only when the supporting CSV/JSON/plot path is recorded here.

## Current PAS P0 Pilot Evidence

Status: clean P0 pilot, not final paper evidence by itself.

Protocol recorded on 2026-05-20:

- Model/dataset: `OPT-2.7B` on `WikiText-2`.
- Candidate pool: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates`.
- Target sparsity: `0.30`.
- Probe budgets: `0.25 / 0.30 / 0.35`.
- Held-out future budget: `0.40`, recorded as analysis-only.
- Candidate count: `top_k=20`.
- Fixed shortlist rule: `top-2-by-ell_0`.
- Probe samples: `16`.
- Held-out samples: `32`.

Primary artifacts:

- Manifest: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/artifact_manifest.json`
- PAS regret table: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/selection_regret.csv`
- Warning correlation: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/warning_correlation.csv`
- Joined probe/held-out rows: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/pas_joined_probe_heldout.csv`
- High-sample selected-candidate recheck:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64/selected_heldout_recheck.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64/selected_heldout_recheck_regret.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64/selected_heldout_recheck_manifest.json`
- Figures:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/path_divergence.pdf`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/endpoint_ambiguity_scatter.pdf`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/warning_correlation.pdf`

Observed P0 pilot values:

- `FF-Endpoint` regret at held-out `0.40`: `0.27734373462907325`.
- `PAS-Slope` regret at held-out `0.40`: `0.0`.
- `PAS-Curv` regret at held-out `0.40`: `0.0`.
- `PAS-Plus` regret at held-out `0.40`: `0.27734373462907325`.
- `Oracle-heldout` candidate: `p0_candidates_parallel_gpu4_opt-2.7b_seed2029_step000047_ep000047`.
- Slope correlation with held-out degradation: Pearson `0.7119764114534842`, Spearman `0.8030075187969924`.
- Curvature correlation with held-out degradation: Pearson `0.5427058989743604`, Spearman `0.5969924812030075`.

Selected-candidate high-sample `0.40` recheck, recorded on 2026-05-20:

- Recheck samples: `64`.
- `FF-Endpoint` / `PAS-Plus` candidate: `p0_candidates_parallel_gpu0_opt-2.7b_seed2025_step000004_ep000004`.
- `FF-Endpoint` / `PAS-Plus` rechecked `ell_h`: `5.119642685746274`; rechecked PPL: `167.2755889892578`; regret vs. best rechecked: `0.27299068075105826`.
- `PAS-Slope` / `PAS-Curv` / `Oracle-heldout` candidate: `p0_candidates_parallel_gpu4_opt-2.7b_seed2029_step000047_ep000047`.
- `PAS-Slope` / `PAS-Curv` / `Oracle-heldout` rechecked `ell_h`: `4.8466520049952155`; rechecked PPL: `127.31343078613281`; regret vs. best rechecked: `0.0`.

Paper table staging:

| Table Slot | Current Fill | Artifact Source | Status |
| --- | --- | --- | --- |
| PAS same-pool selection regret, P0 OPT-2.7B/WikiText-2 | FF `0.27734373462907325`, PAS-Slope `0.0`, PAS-Curv `0.0`, Oracle `0.0` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/selection_regret.csv` | Pilot-only; high-sample selected-candidate recheck completed; needs second seed pool. |
| Selected-candidate high-sample `0.40` recheck, P0 OPT-2.7B/WikiText-2 | FF/PAS-Plus regret `0.27299068075105826`; PAS-Slope/PAS-Curv/Oracle regret `0.0` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_selected_recheck64/selected_heldout_recheck_regret.csv` | Supports the selected-candidate direction in this pool; still pilot-only until second seed pool. |
| Warning-to-heldout correlation, P0 OPT-2.7B/WikiText-2 | slope Spearman `0.8030075187969924`, curvature Spearman `0.5969924812030075` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/warning_correlation.csv` | Pilot-only; `0.40` held-out was not used for selection. |
| Path/ambiguity figures | `path_divergence.pdf`, `endpoint_ambiguity_scatter.pdf`, `warning_correlation.pdf` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/` | Ready as P0 pilot figures; not final paper figures yet. |

Interpretation guardrail:

- This supports the PAS direction for one controlled P0 pool.
- It does not establish universal PAS improvement.
- Before filling manuscript tables as final evidence, run at least one independent seed pool.

## Independent PAS P0 Seed Pool Evidence

Status: second independent P0 pool completed; still same model/dataset setting.

Protocol recorded on 2026-05-20:

- Model/dataset: `OPT-2.7B` on `WikiText-2`.
- Candidate pool: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates`.
- PAS output: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025`.
- Seed: `3025`.
- Target sparsity: `0.30`.
- Probe budgets: `0.25 / 0.30 / 0.35`.
- Held-out future budget: `0.40`, analysis-only.
- Candidate count: `top_k=20`.
- Fixed shortlist rule: `top-2-by-ell_0`.
- Probe samples: `16`.
- Held-out samples: `32`.

Primary artifacts:

- Manifest: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/artifact_manifest.json`
- PAS regret table: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selection_regret.csv`
- Warning correlation: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/warning_correlation.csv`
- Joined probe/held-out rows: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/pas_joined_probe_heldout.csv`
- Figures:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/path_divergence.pdf`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/endpoint_ambiguity_scatter.pdf`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/warning_correlation.pdf`

Observed seed `3025` values:

- `FF-Endpoint` regret at held-out `0.40`: `0.5581475044600985`.
- `PAS-Plus` regret at held-out `0.40`: `0.5581475044600985`.
- `PAS-Slope` regret at held-out `0.40`: `0.0`.
- `PAS-Curv` regret at held-out `0.40`: `0.5581475044600985`.
- `Oracle-heldout` candidate: `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048`.
- `PAS-Slope` selected the oracle candidate.
- `PAS-Curv` selected the endpoint candidate in this pool, so curvature should be treated as an ablation.
- Slope correlation with held-out degradation: Pearson `0.8337100870651009`, Spearman `0.8285714285714285`.
- Curvature correlation with held-out degradation: Pearson `0.391305442885115`, Spearman `0.41052631578947363`.

Selected-candidate high-sample `0.40` recheck for seed `3025`, recorded on 2026-05-20:

- Recheck artifacts:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_regret.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_manifest.json`
- Recheck samples: `64`.
- Total eval seconds: `221.16922211647034`.
- `PAS-Slope` / `Oracle-heldout` candidate: `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048`.
- `PAS-Slope` / `Oracle-heldout` rechecked `ell_h`: `4.954576021785408`; rechecked PPL: `141.8224639892578`; regret vs. best rechecked: `0.0`.
- `FF-Endpoint` / `PAS-Plus` / `PAS-Curv` candidate: `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037`.
- `FF-Endpoint` / `PAS-Plus` / `PAS-Curv` rechecked `ell_h`: `5.594643081425589`; rechecked PPL: `268.98162841796875`; regret vs. best rechecked: `0.6400670596401818`.
- Success condition satisfied: PAS-Slope selected candidate remains better than FF-Endpoint under `64` held-out samples.

## P0 Two-Pool Table Staging

| Pool | FF-Endpoint Regret | PAS-Plus Regret | PAS-Slope Regret | PAS-Curv Regret | Random Mean Regret | Slope Spearman | Curv Spearman | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `seed2025` | `0.27734373462907325` | `0.27734373462907325` | `0.0` | `0.0` | `0.14033592972231104` | `0.8030075187969924` | `0.5969924812030075` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/selection_regret.csv` |
| `seed3025` | `0.5581475044600985` | `0.5581475044600985` | `0.0` | `0.5581475044600985` | `0.27907375223004927` | `0.8285714285714285` | `0.41052631578947363` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selection_regret.csv` |

Interpretation guardrail:

- `PAS-Slope` is the current strongest P0 selection rule.
- `PAS-Plus` does not improve over endpoint in these two pools.
- `PAS-Curv` is mixed: it matches oracle in `seed2025` but fails in `seed3025`.
- This is enough to stage a P0 table, but not enough for a broad model-general claim.

## Next Artifact Slots

Priority order after two-pool P0:

1. Seed `3025` selected-candidate high-sample recheck: completed.
   - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck.csv`
   - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_regret.csv`
   - success condition satisfied: PAS-Slope selected candidate remains better than FF-Endpoint at held-out `0.40` with `64` samples.
2. Cross-model PAS pilot, preferred `OPT-1.3B`:
   - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/selection_regret.csv`
   - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/warning_correlation.csv`
   - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/artifact_manifest.json`
3. Compensation-aligned final evaluation:
   - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_eval.csv`
   - completed with identical no-reconstruction settings for `FF-Endpoint` and `PAS-Slope`.
4. Overhead summary:
   - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.csv`
   - completed; missing GPU-hour accounting is marked explicitly rather than guessed.

## PAS Overhead Summary Evidence

Status: seed `3025` overhead summary completed with explicit missing fields.

Protocol recorded on 2026-05-21:

- PAS directory: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025`.
- Selected-candidate recheck directory: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64`.
- Final-eval directory: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon`.

Primary artifacts:

- Overhead CSV: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.csv`
- Overhead JSON: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_overhead/pas_overhead_summary.json`

Observed overhead rows:

| Stage | Wall Seconds | Approx GPU-Hours | Status |
| --- | --- | --- | --- |
| candidate search | missing | missing | `missing` |
| PAS probe `0.25/0.30/0.35` | `2871.348848581314` | `0.7975969023836984` | `measured_from_probe_rows` |
| held-out analysis `0.40` | missing | missing | `missing_in_existing_artifacts` |
| selected-candidate recheck `0.40` | `221.16922211647034` | `0.061435895032352875` | `measured_from_recheck` |
| compensation-aligned final eval | `233.10833382606506` | `0.06475231495168474` | `measured_from_final_eval` |
| final checkpoint type | n/a | n/a | `not_gpu_cost`; static exact-budget pruned checkpoint |
| inference-time overhead | `0.0` | `0.0` | `by_design` |

Interpretation guardrail:

- PAS overhead is selection/evaluation time only; the deployed artifact remains a static checkpoint.
- Candidate-search and original held-out analysis timing are not recoverable from existing artifacts and should be reported as missing, not estimated.

## Policy-Selection Pivot Artifact Plan

Status: prepared on 2026-05-21 for the budget-transferable priority-vector framing.

P0 consolidated policy-selection table:

- Command:
  - `python pas_policy_selection_report.py --output_dir /workspace/ckpts/pas_policy_selection_20260521`
- Expected artifacts:
  - `/workspace/ckpts/pas_policy_selection_20260521/policy_selection_tradeoff.csv`
  - `/workspace/ckpts/pas_policy_selection_20260521/policy_selection_tradeoff.md`
  - `/workspace/ckpts/pas_policy_selection_20260521/price_of_budget_robustness_seed3025.csv`
  - `/workspace/ckpts/pas_policy_selection_20260521/policy_selection_manifest.json`
- Required interpretation:
  - metrics use polished-draft names: `PoBR_sigma`, `StressGain_h`, and `Regret_h`.
  - target-budget and stricter-budget values may come from different artifact protocols; `artifact_source_target`, `artifact_source_heldout`, and `notes` record this explicitly.
  - after the server reset, seed `3025` includes probe-side target values and selected-candidate `0.40` recheck values; use the source columns before citing a compensation-aligned target/stress comparison.
  - `uses_heldout_for_selection` must be `no` for FF/PAS rows; `Oracle-heldout` is analysis-only.

Completed server-reset run, recorded 2026-05-21:

- Commit used on server: `b407d39`.
- Processed pool: `opt27b_seed3025`; candidate count `20`.
- Missing after server reset: prior `opt27b_seed2025` and `opt13b_seed2025` artifacts.
- Candidate pool: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates`.
- Main table: `/workspace/ckpts/pas_policy_selection_20260521/price_of_budget_robustness_seed3025.csv`.
- Protocol status for the consolidated table: `protocol_mismatch_target_probe_vs_selected_recheck`; target columns come from `probe_ell_sigma`, stress columns come from `selected_recheck_64`.
- Key rows:

| Rule | Candidate | `PoBR_sigma` | `StressGain_h` | `Regret_h` | target ell/ppl | stress ell/ppl |
| --- | --- | --- | --- | --- | --- | --- |
| `FF-Endpoint` | `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037` | `0.0` | `0.0` | `0.6400670596401818` | `4.342522166971652` / `76.90125274658203` | `5.594643081425589` / `268.98162841796875` |
| `PAS-Slope` | `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048` | `0.04040185393219975` | `0.6400670596401818` | `0.0` | `4.382924020903852` / `80.07182312011719` | `4.954576021785408` / `141.8224639892578` |

P1 endpoint-compatibility sensitivity:

- Same command as P0, because the report script also writes:
  - `/workspace/ckpts/pas_policy_selection_20260521/shortlist_sensitivity.csv`
  - `/workspace/ckpts/pas_policy_selection_20260521/shortlist_sensitivity.md`
- Fixed predeclared shortlist rules:
  - `top_m = 2, 3, 5`
  - `epsilon_logloss = 0.02, 0.05, 0.10`
- Required interpretation:
  - `0.40` is analysis-only; shortlist size/type is predeclared, not tuned from held-out results.

P2 policy-selection figures:

- Same command as P0, because the report script also writes per-pool figures under:
  - `/workspace/ckpts/pas_policy_selection_20260521/figures/`
- Expected figure names per pool:
  - `robustness_frontier.pdf`
  - `path_divergence.pdf`
  - `sensitivity_correlation.pdf`
- Auxiliary/debug figure aliases are also kept:
  - `endpoint_ambiguity_scatter.pdf`
  - `policy_path_lines.pdf`
  - `target_future_tradeoff.pdf`
  - `warning_correlation.pdf`

P3 matched stricter-budget final evaluation:

- Existing selected-candidate recheck is protocol-equivalent to the requested matched `0.40` final eval: it uses `amc_searchPPO.py --job=compile`, `final_sparsity=0.40`, no reconstruction, same model/dataset, same sample count, and the same selected priority-vector candidates.
- Materialization command:
  - `python pas_export_future_eval_from_recheck.py --selected_candidates_json /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selected_candidates.json --recheck_csv /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck.csv --recheck_regret_csv /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_selected_recheck64/selected_heldout_recheck_regret.csv --model /workspace/Models/opt-2.7b --model_name opt-2.7b --future_sparsity 0.40 --rules FF-Endpoint,PAS-Slope --num_samples 64 --batch_size 8 --seed 3025 --output_dir /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck`
- Expected artifacts:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck/pas_compensation_aligned_eval_40.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck/pas_compensation_aligned_manifest_40.json`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck/pas_compensation_aligned_commands_40.sh`

Completed `0.40` matched eval materialization, recorded 2026-05-21:

- Output directory: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck`.
- Protocol match for stress budget: yes. The manifest states that the selected-candidate recheck used `amc_searchPPO.py --job=compile`, `final_sparsity=0.40`, no reconstruction, same model/dataset, same sample count, and the same selected priority-vector candidates.
- Batch size: `8`; samples: `64`; reconstruction: disabled.
- Key rows:

| Rule | Candidate | ell | ppl | actual sparsity | `Regret_h` | eval seconds |
| --- | --- | --- | --- | --- | --- | --- |
| `FF-Endpoint` | `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037` | `5.594643081425589` | `268.98162841796875` | `0.39979534733172495` | `0.6400670596401818` | `122.88303422927856` |
| `PAS-Slope` | `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048` | `4.954576021785408` | `141.8224639892578` | `0.3999388122211047` | `0.0` | `125.75517463684082` |

## PAS Stress-Recovery Evidence Gate

Status: scripts and runbook prepared on 2026-05-21. This gate supersedes
starting periodic PAS/RPVS.

| Claim | Status | Required artifact |
| --- | --- | --- |
| Claim 1: `S35` predicts cross-budget regret | P0 positive on seed `3025` after controlling `L30` | `/workspace/ckpts/pas_stress_recovery/stress_correlation_opt27b.csv` |
| Claim 2: `S35` predicts recovery quality | pending | `/workspace/ckpts/pas_stress_recovery/recovery_table_opt27b_seed3025.csv` |
| Claim 3: `S35` predicts downstream retention | pending | `/workspace/ckpts/pas_stress_recovery/downstream_retention_opt27b.csv` |

P0 command sequence:

```bash
cd /workspace/structure_pruning
git fetch origin
git pull --ff-only origin main

python scripts/pas_export_candidate_stress_table.py \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --probe-results /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/probe_results.csv \
  --heldout-results /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/heldout_results.csv \
  --output-dir /workspace/ckpts/pas_stress_recovery

python scripts/pas_analyze_stress_correlations.py \
  --model /workspace/Models/opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --stress-tables /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv \
  --output-dir /workspace/ckpts/pas_stress_recovery
```

P0 expected artifacts:

- `/workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv`
- `/workspace/ckpts/pas_stress_recovery/candidate_stress_manifest_opt27b.json`
- `/workspace/ckpts/pas_stress_recovery/stress_correlation_opt27b.csv`
- `/workspace/ckpts/pas_stress_recovery/stress_correlation_opt27b.md`
- `/workspace/ckpts/pas_stress_recovery/stress_correlation_manifest_opt27b.json`

P0 observed result, recorded 2026-05-21:

| Metric | Value | Interpretation |
| --- | --- | --- |
| `Pearson(S35,Regret40)` | `0.1625803636488299` | Raw held-out regret correlation is weak because endpoint loss confounds absolute `L40` quality. |
| `Spearman(S35,Regret40)` | `0.09473684210526315` | Same raw-regret caveat. |
| `Pearson(S35,Delta40)` | `0.8337100870651009` | Strong evidence that local stress predicts cross-budget degradation. |
| `Spearman(S35,Delta40)` | `0.8285714285714285` | Strong rank-level stress/degradation relation. |
| `partial_corr(S35,Regret40|L30)` | `0.8107895146421564` | Main P0 positive result: stress predicts held-out regret after controlling endpoint quality. |
| `partial_corr(S35,L40|L30)` | `0.8107895146421564` | Equivalent controlled stress/future-loss signal. |
| `linear_regression:L40~L30+S35` | `beta_L30=0.9626057100144196`, `beta_S35=1.2588788204496062`, `R2=0.8817489807506785` | `S35` remains a positive predictor alongside endpoint loss. |
| `linear_regression:Regret40~L30+S35` | `beta_L30=0.962605710014419`, `beta_S35=1.2588788204496064`, `R2=0.8817489807506785` | Same controlled-regret result. |

P0 reading: proceed to P1 same-protocol recovery. The claim should be phrased
as "local stress predicts cross-budget degradation and controlled held-out
regret", not as raw `S35` alone ranking absolute `Regret40`.

P1 command sequence:

```bash
python scripts/pas_build_recovery_subset.py \
  --model /workspace/Models/opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --stress-table /workspace/ckpts/pas_stress_recovery/candidate_stress_table_opt27b_seed3025.csv \
  --selected-candidates-json /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selected_candidates.json \
  --output-dir /workspace/ckpts/pas_stress_recovery

bash scripts/pas_run_recovery_multigpu.sh \
  --model /workspace/Models/opt-2.7b \
  --model-name opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --recovery-subset /workspace/ckpts/pas_stress_recovery/recovery_subset_opt27b_seed3025.csv \
  --output-dir /workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly \
  --gpu-ids "0 1 2 3 4 5 6 7" \
  --recovery-method ffn_only_ridge_reconstruction \
  --batch-size 8 \
  --num-samples 64 \
  --recon-sample 16

python scripts/pas_collect_recovery_results.py \
  --model /workspace/Models/opt-2.7b \
  --dataset wikitext2 \
  --seed 3025 \
  --candidate-pool /workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates \
  --target-sigma 0.30 \
  --probe-sigma 0.35 \
  --heldout-sigma 0.40 \
  --recovery-subset /workspace/ckpts/pas_stress_recovery/recovery_subset_opt27b_seed3025.csv \
  --recovery-dir /workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly \
  --recovery-method ffn_only_ridge_reconstruction \
  --output-dir /workspace/ckpts/pas_stress_recovery
```

P1 expected artifacts:

- `/workspace/ckpts/pas_stress_recovery/recovery_subset_opt27b_seed3025.csv`
- `/workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly/recovery_multigpu_manifest.json`
- `/workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly/recovery_launch.tsv`
- `/workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly/recovery_commands_gpu*.sh`
- `/workspace/ckpts/pas_stress_recovery/recovery_seed3025_ffnonly/logs/recovery_gpu*.log`
- `/workspace/ckpts/pas_stress_recovery/recovery_table_opt27b_seed3025.csv`
- `/workspace/ckpts/pas_stress_recovery/recovery_analysis_opt27b_seed3025.csv`
- `/workspace/ckpts/pas_stress_recovery/recovery_manifest_opt27b.json`

Protocol note: P1 uses `ffn_only_ridge_reconstruction`. Attention heads are
structurally pruned without ridge reconstruction because the old OPT head-recon
path can leave `out_proj` dimensions inconsistent; FFN modules receive the same
ridge recovery protocol for every candidate.

Interpretation guardrail: if P0 or P1 fails, do not claim PAS improves
recovery and do not start RPVS as a rescue experiment.

P4 one-more-setting command slot:

- Preferred next setting:
  - `OPT-2.7B`, `sigma=0.35`, probe `0.30/0.35/0.40`, held-out `0.45`.
- Expected artifacts:
  - `/workspace/ckpts/opt-2.7b/sparsity_0.35/p0_pas_seed2025/artifact_manifest.json`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.35/p0_pas_seed2025/selection_regret.csv`
  - `/workspace/ckpts/opt-2.7b/sparsity_0.35/p0_pas_seed2025/warning_correlation.csv`
  - `/workspace/ckpts/pas_policy_selection_sigma035/policy_selection_tradeoff_sigma035.csv`
  - `/workspace/ckpts/pas_policy_selection_sigma035/shortlist_sensitivity_sigma035.csv`
  - `/workspace/ckpts/pas_policy_selection_sigma035/figures/opt27b_sigma035_seed2025/robustness_frontier_sigma035.pdf`
  - `/workspace/ckpts/pas_policy_selection_sigma035/figures/opt27b_sigma035_seed2025/path_divergence_sigma035.pdf`
  - `/workspace/ckpts/pas_policy_selection_sigma035/figures/opt27b_sigma035_seed2025/sensitivity_correlation_sigma035.pdf`

## Compensation-Aligned Final Evaluation Evidence

Status: seed `3025` final target-sparsity evaluation completed without reconstruction/calibration.

Protocol recorded on 2026-05-20:

- Model/dataset: `OPT-2.7B` on `WikiText-2`.
- Source selection artifact: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/selected_candidates.json`.
- Output directory: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon`.
- Rules compared: `FF-Endpoint` vs. `PAS-Slope`.
- Target sparsity: `0.30`.
- Evaluation samples: `64`.
- Reconstruction/calibration: disabled for both rules.

Primary artifacts:

- Evaluation CSV: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_eval.csv`
- Manifest: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_manifest.json`
- Commands: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval_norecon/pas_compensation_aligned_commands.sh`

Observed final target-sparsity values:

- `FF-Endpoint` candidate: `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037`.
- `FF-Endpoint` final `ell`: `4.433705344243264`; final PPL: `84.24298858642578`; regret vs. best final rule: `0.0`.
- `PAS-Slope` candidate: `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048`.
- `PAS-Slope` final `ell`: `4.499888504299609`; final PPL: `90.00709533691406`; regret vs. best final rule: `0.06618316005634473`.
- Total eval seconds: `233.10833382606506`.

Interpretation guardrail:

- At the target sparsity `0.30`, FF-Endpoint is better in this no-reconstruction final evaluation.
- At held-out future sparsity `0.40`, PAS-Slope is better in the seed `3025` recheck.
- Therefore the current evidence supports a trade-off between endpoint quality and future robustness, not a blanket claim that PAS-Slope wins every metric.

## Cross-Model PAS Pilot Evidence

Status: OPT-1.3B/WikiText-2 pilot completed; no selection-regret gain because endpoint is already oracle in this pool.

Protocol recorded on 2026-05-20:

- Model/dataset: `OPT-1.3B` on `WikiText-2`.
- Candidate pool: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_candidates_seed2025/candidates`.
- PAS output: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025`.
- Seed: `2025`.
- Target sparsity: `0.30`.
- Probe budgets: `0.25 / 0.30 / 0.35`.
- Held-out future budget: `0.40`, analysis-only.
- Candidate count: `top_k=20`.
- Fixed shortlist rule: `top-2-by-ell_0`.
- Probe samples: `16`.
- Held-out samples: `32`.

Primary artifacts:

- Manifest: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/artifact_manifest.json`
- PAS regret table: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/selection_regret.csv`
- Warning correlation: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/warning_correlation.csv`
- Joined probe/held-out rows: `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/pas_joined_probe_heldout.csv`
- Figures:
  - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/path_divergence.pdf`
  - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/endpoint_ambiguity_scatter.pdf`
  - `/workspace/ckpts/opt-1.3b/sparsity_0.30/p0_pas_seed2025/warning_correlation.pdf`

Observed OPT-1.3B pilot values:

- `FF-Endpoint`, `PAS-Plus`, `PAS-Slope`, `PAS-Curv`, and `Oracle-heldout` all select `p0_candidates_opt13b_seed2025_gpu2_opt-1.3b_seed2027_step000010_ep000010`.
- Held-out `ell_h`: `5.113950659556641`; held-out PPL: `166.32615661621094`.
- Selection regret for all controlled deterministic rules: `0.0`.
- `Random-shortlist` regret mean/std: `0.5948401192835622` / `0.6020215478269607`.
- Slope correlation with held-out degradation: Pearson `0.6603016220996479`, Spearman `0.5774436090225563`.
- Curvature correlation with held-out degradation: Pearson `0.17611873583956558`, Spearman `0.17142857142857143`.

Interpretation guardrail:

- This supports cross-model feasibility and a positive slope/degradation relationship.
- It does not show PAS-Slope beating endpoint for OPT-1.3B, because endpoint already selects the held-out oracle candidate.
