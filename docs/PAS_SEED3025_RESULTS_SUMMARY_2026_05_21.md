# PAS Seed3025 Result Summary

Date: 2026-05-21

Purpose: compact handoff for other chats. This records the server-reset
OPT-2.7B seed3025 artifacts and the current interpretation guardrails.

## Scope

- Model: `OPT-2.7B`
- Dataset: `WikiText-2`
- Candidate pool: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_candidates_seed3025/candidates`
- PAS output: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025`
- Policy-selection report: `/workspace/ckpts/pas_policy_selection_20260521`
- Stress eval output: `/workspace/ckpts/opt-2.7b/sparsity_0.30/p0_pas_seed3025_final_eval40_from_recheck`
- Candidate count: `20`
- Commit used for polished metric names: `b407d39`
- Commit recording these artifacts in docs: `7973f72`

## Main Artifacts

- `price_of_budget_robustness_seed3025.csv`
- `policy_selection_tradeoff.csv`
- `shortlist_sensitivity.csv`
- `policy_selection_manifest.json`
- `figures/opt27b_seed3025/robustness_frontier.pdf`
- `figures/opt27b_seed3025/path_divergence.pdf`
- `figures/opt27b_seed3025/sensitivity_correlation.pdf`
- `pas_compensation_aligned_eval_40.csv`
- `pas_compensation_aligned_manifest_40.json`
- `pas_compensation_aligned_commands_40.sh`

## Key Rows

| Rule | Candidate | `PoBR_sigma` | `StressGain_h` | `Regret_h` | target ell/ppl | stress ell/ppl |
| --- | --- | --- | --- | --- | --- | --- |
| `FF-Endpoint` | `p0_candidates_seed3025_gpu6_opt-2.7b_seed3031_step000037_ep000037` | `0.0` | `0.0` | `0.6400670596401818` | `4.342522166971652` / `76.90125274658203` | `5.594643081425589` / `268.98162841796875` |
| `PAS-Slope` | `p0_candidates_seed3025_gpu5_opt-2.7b_seed3030_step000048_ep000048` | `0.04040185393219975` | `0.6400670596401818` | `0.0` | `4.382924020903852` / `80.07182312011719` | `4.954576021785408` / `141.8224639892578` |

## Matched Stress Eval

The `0.40` stress side is matched no-recovery final evaluation for the selected
candidates:

- `FF-Endpoint`: ell `5.594643081425589`, PPL `268.98162841796875`, actual sparsity `0.39979534733172495`
- `PAS-Slope`: ell `4.954576021785408`, PPL `141.8224639892578`, actual sparsity `0.3999388122211047`

The manifest states the selected-candidate recheck used `amc_searchPPO.py
--job=compile`, `final_sparsity=0.40`, no reconstruction, same model/dataset,
same sample count, and the same selected priority-vector candidates.

## Guardrails

- Do not claim PAS improves target PPL from these artifacts.
- Current `price_of_budget_robustness_seed3025.csv` intentionally records
  `protocol_mismatch_target_probe_vs_selected_recheck`: target values are
  probe-side `L30`, while stress values are selected-candidate recheck.
- The safe current claim is: in this candidate pool, PAS-Slope pays a small
  probe-side endpoint price (`PoBR_sigma = 0.0404`) and eliminates held-out
  `0.40` selected-candidate regret (`Regret_h = 0.0`), while FF-Endpoint has
  `Regret_h = 0.6401`.
- The next priority is not RPVS. P0 has already shown that local stress `S35`
  predicts cross-budget degradation/controlled regret. P1 recovery is mixed, so
  the next useful check is raw no-compensation downstream retention.
- Follow-up P1 recovery on 2026-05-22 was mixed/weak under
  `ffn_only_ridge_reconstruction`: `Pearson(S35,L30_recovered)=-0.3160`,
  `Pearson(S35,RecoveryGain)=0.2264`, and
  `L30_recovered~L30_raw+S35` gave `beta_S35=-0.1146`.
  Do not claim PAS improves recovery from this evidence.

## Next Evidence Gate

Use `docs/PAS_STRESS_RECOVERY_EVIDENCE_2026_05_21.md` as the active plan:

1. P0: candidate-level `L30/L35/L40`, `S35`, `Regret40`, `Delta40`, plus
   correlation and partial correlation controlling `L30`.
2. P1: fixed recovery subset with identical reconstruction/recovery protocol
   for all candidates. Completed with mixed/weak recovery evidence; keep as a
   negative/guardrail result unless a stronger recovery protocol is later run.
3. P2: raw no-compensation downstream retention completed on 2026-05-22 with
   formal `lm-eval-harness` evaluation. The result is weak/mixed:
   all-candidate `partial_corr(S35,avg_pruned_score|L30_raw)=-0.1879`, while
   endpoint-close subsets are mildly positive (`top8=0.1432`, `top13=0.2512`).
   Do not use this as a headline downstream claim; it is a capability-retention
   sanity check that leaves the main evidence on P0 cross-budget stress.
