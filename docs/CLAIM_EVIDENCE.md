# FastForward Early-Warning Claim-Evidence Checklist

| Claim | Evidence Needed | Current Asset | Gap | Next Action |
| --- | --- | --- | --- | --- |
| Endpoint-similar pruning candidates can have divergent future compression paths. | Candidate path CSV across sparsities, plus `path_divergence.pdf`. | Smoke candidate pool exists: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates/candidates/candidates.jsonl` with 8 selected candidates from 58 merged candidates. | Need path evaluation across future sparsities and plot artifact. | Run `evaluate_candidate_paths.py` or `evaluate_high_sparsity_curve.py`, then `plot_path_divergence.py`. |
| Local log-PPL curvature predicts future degradation. | Pearson/Spearman/AUROC between curvature at warning sparsity and separated future log-PPL increase: curvature from 0.25/0.30/0.35, future target `logPPL(0.40)-logPPL(0.30)`. | Smoke probe exists: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/probe_results.csv` with 8 probe rows; best smoke candidate has `curvature=-90.62480751317209`, local diagnostic `local_probe_degradation_0.35_minus_0.30=0.5305805215447261`. | Smoke run is too small (`N_SAMPLES=8`, 8 candidates), and its local diagnostic overlaps with the curvature probe. | Run larger P0 and compute separated future degradation at 0.40. Do not use 0.40 for reranking, lambda/tau tuning, candidate filtering, or early stopping. |
| Curvature reranking improves high-sparsity stability over endpoint-only selection. | Endpoint vs. slope vs. curvature selected candidates evaluated at 30/35/40% sparsity. | Smoke rerank exists: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/rerank_results.csv`, `best_candidate.json`, `selected_candidates.json`. | Need selected candidates evaluated at future sparsities before any improvement claim. | Run rerank modes, compile selected policies, evaluate high-sparsity curve. |
| Gains come from the early-warning criterion, not merely extra probe evaluations. | Ablation table: endpoint, probe-only, slope, curvature. | Smoke selected-candidate JSON exists at `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_ew/selected_candidates.json`. | Need comparable ablation outputs, not just one smoke rerank. | Generate ablation CSV/LaTeX after a larger P0 probe run. |
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

Paper table staging:

| Table Slot | Current Fill | Artifact Source | Status |
| --- | --- | --- | --- |
| PAS same-pool selection regret, P0 OPT-2.7B/WikiText-2 | FF `0.27734373462907325`, PAS-Slope `0.0`, PAS-Curv `0.0`, Oracle `0.0` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/selection_regret.csv` | Pilot-only; needs high-sample selected-candidate recheck and second seed pool. |
| Warning-to-heldout correlation, P0 OPT-2.7B/WikiText-2 | slope Spearman `0.8030075187969924`, curvature Spearman `0.5969924812030075` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/warning_correlation.csv` | Pilot-only; `0.40` held-out was not used for selection. |
| Path/ambiguity figures | `path_divergence.pdf`, `endpoint_ambiguity_scatter.pdf`, `warning_correlation.pdf` | `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas/` | Ready as P0 pilot figures; not final paper figures yet. |

Interpretation guardrail:

- This supports the PAS direction for one controlled P0 pool.
- It does not establish universal PAS improvement.
- Before filling manuscript tables as final evidence, run a selected-candidate high-sample `0.40` recheck and at least one independent seed pool.
