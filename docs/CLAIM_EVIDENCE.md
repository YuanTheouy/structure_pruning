# FastForward Early-Warning Claim-Evidence Checklist

| Claim | Evidence Needed | Current Asset | Gap | Next Action |
| --- | --- | --- | --- | --- |
| Endpoint-similar pruning candidates can have divergent future compression paths. | Candidate path CSV across sparsities, plus `path_divergence.pdf`. | Two P0 pools now have `path_divergence.pdf`: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/path_divergence.pdf` and `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025/path_divergence.pdf`. | Need figure inspection before manuscript use. | Inspect figures and keep the claim scoped to same-pool P0 evidence. |
| Local path-warning slope predicts future degradation. | Pearson/Spearman between slope from `0.25/0.30/0.35` and separated future target `logPPL(0.40)-logPPL(0.30)`. | Two P0 pools show slope Spearman `0.8030075187969924` and `0.8285714285714285`. | Evidence is still OPT-2.7B/WikiText-2 P0 only. | Use slope as the primary PAS warning score for the current paper table staging. |
| Curvature is a useful but not primary warning score. | Pearson/Spearman for curvature from `0.25/0.30/0.35` against separated future target. | Two P0 pools show curvature Spearman `0.5969924812030075` and `0.41052631578947363`; PAS-Curv succeeds in the first pool but fails in seed `3025`. | Curvature is weaker and mixed relative to slope. | Keep PAS-Curv as an ablation, not the headline selection rule. |
| Path-warning selection improves high-sparsity stability over endpoint-only selection. | Same-pool selection regret table: FF-Endpoint, PAS-Plus, PAS-Slope, PAS-Curv, Oracle-heldout. | Two P0 pools show PAS-Slope regret `0.0` in both; FF-Endpoint regret is `0.27734373462907325` and `0.5581475044600985`. | Need broader model/dataset evidence before broad claims. | Fill P0 table staging, but phrase final paper claims as P0 evidence unless expanded. |
| Gains come from the early-warning criterion, not merely extra probe evaluations. | Ablation table: endpoint, plus-probe, slope, curvature, oracle. | Two P0 pools have comparable `selection_regret.csv` artifacts under `p0_pas` and `p0_pas_seed3025`. | Need final table formatting and optional selected-candidate high-sample recheck for seed `3025`. | Use PAS-Plus as the probe-only control and PAS-Slope as the warning selection. |
| Early-warning adds selection-stage cost but no inference-time overhead. | Runtime summary with search GPU-hours, probe cost, calibration cost, and final checkpoint type. | Smoke probe row records `eval_seconds_total=144.03351759910583` for best candidate; scripts record per-shard logs under `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/shards/`. | Need full run-level overhead summary and final checkpoint metadata. | Run `summarize_overhead.py` after larger candidate probe/evaluation. |
| The narrowed contribution is novel enough under 2026 related work. | Independent novelty matrix and reviewer-risk review with 2025-2026 pruning/phase-transition work. | `docs/NOVELTY_REVIEW.md` completed on 2026-05-19. | Need to ensure final manuscript follows the narrowed wording and cites GISP, GRASPrune, Olica, Týr-the-Pruner, and phase-transition work correctly. | Keep contribution wording as candidate-selection/early-warning diagnostic; do not broaden to generic structured pruning. |

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
   - missing GPU-hour accounting must be marked explicitly rather than guessed.

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
