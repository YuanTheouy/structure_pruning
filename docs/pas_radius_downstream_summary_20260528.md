# PAS Radius/Downstream Evidence Summary 2026-05-28

Generated on server at `2026-05-28 11:42:07`.

Main interpretation:

- `S35` is the main compression-path stress signal for future-budget fragility.
- Small local radii (`S3025`, `S3050`, `S31`) can align with downstream@30 in some settings, but the signal is not stable enough to be the main claim.
- Downstream@30 and 40% path robustness can conflict, so these should be reported as separate objectives.

## Downstream Correlation

| model | seed | metric | pearson | spearman | partial_corr_given_L30 |
| --- | --- | --- | --- | --- | --- |
| OPT-1.3B | 4025 | S3025 | 18 | -0.441862 | -0.432558 |
| OPT-1.3B | 4025 | S3050 | 18 | -0.169838 | -0.399794 |
| OPT-1.3B | 4025 | S31 | 18 | 0.0526308 | -0.0640496 |
| OPT-1.3B | 4025 | S35 | 18 | 0.373123 | 0.355372 |
| OPT-1.3B | 5025 | S3025 | 0.160791 | 0.047163 | 0.165183 |
| OPT-1.3B | 5025 | S3050 | 0.0940579 | 0.116434 | 0.101779 |
| OPT-1.3B | 5025 | S31 | -0.0248149 | -0.0869567 | -0.0312283 |
| OPT-1.3B | 5025 | S35 | 0.0482882 | -0.0235815 | 0.0500476 |
| OPT-2.7B | 7025 | S3025 | 0.0866025 | -0.141018 | 0.103274 |
| OPT-2.7B | 7025 | S3050 | -0.00528905 | -0.131208 | -0.01613 |
| OPT-2.7B | 7025 | S31 | 0.0413671 | -0.0784795 | -0.0725508 |
| OPT-2.7B | 7025 | S35 | 0.54691 | 0.513795 | 0.513329 |
| OPT-2.7B | 8025 | S3025 | 0.163789 | 0.0919898 | 0.135636 |
| OPT-2.7B | 8025 | S3050 | 0.297231 | 0.240827 | 0.272393 |
| OPT-2.7B | 8025 | S31 | 0.257258 | 0.122998 | 0.243953 |
| OPT-2.7B | 8025 | S35 | 0.705486 | 0.546771 | 0.702734 |

## Selection Summary

| model | seed | rule | step | avg_pruned_score | L30 | S3025 | S3050 | S31 | S35 | L40 | Regret40 | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-1.3B | 4025 | FF-Endpoint | 789 | 3.52455 | 33.9386 | 0.013393 | 0.0290179 | 0.064732 | 0.439732 | 4.60714 | 100.197 | 0.359375 |
| OPT-1.3B | 4025 | PAS-S3025 | 708 | 3.63393 | 37.8613 | -0.0178571 | -0.00892861 | 0.00446413 | 0.620536 | 4.90625 | 135.132 | 0.658482 |
| OPT-1.3B | 4025 | PAS-S3050 | 714 | 3.81473 | 45.3646 | -0.013393 | -0.017857 | 0.189732 | 0.792411 | 5.68304 | 293.84 | 1.43527 |
| OPT-1.3B | 4025 | PAS-S31 | 683 | 3.90625 | 49.7122 | 0.033482 | 0.0111605 | -0.00223235 | 0.741071 | 5.84375 | 345.071 | 1.59598 |
| OPT-1.3B | 4025 | PAS-S35 | 494 | 3.81696 | 45.466 | 0.0066965 | 0.0200894 | 0.0647323 | 0.149554 | 4.3683 | 78.9097 | 0.120536 |
| OPT-1.3B | 4025 | Oracle40 | 648 | 3.68973 | 40.0341 | 0.0133929 | 0.0133929 | 0.049107 | 0.238839 | 4.24777 | 69.9491 | 0 |
| OPT-1.3B | 4025 | FF-Endpoint | 789 | 0.448333 | 3.52455 | 0.013393 | 0.0290179 | 0.064732 | 0.439732 | 4.60714 | 0.359375 | opt-1.3b_seed4025_step000789_ep000789 |
| OPT-1.3B | 4025 | PAS-S3025 | 708 | 0.483333 | 3.63393 | -0.0178571 | -0.00892861 | 0.00446413 | 0.620536 | 4.90625 | 0.658482 | opt-1.3b_seed4025_step000708_ep000708 |
| OPT-1.3B | 4025 | PAS-S3050 | 714 | 0.505 | 3.81473 | -0.013393 | -0.017857 | 0.189732 | 0.792411 | 5.68304 | 1.43527 | opt-1.3b_seed4025_step000714_ep000714 |
| OPT-1.3B | 4025 | PAS-S31 | 683 | 0.488333 | 3.90625 | 0.033482 | 0.0111605 | -0.00223235 | 0.741071 | 5.84375 | 1.59598 | opt-1.3b_seed4025_step000683_ep000683 |
| OPT-1.3B | 4025 | PAS-S35 | 494 | 0.47 | 3.81696 | 0.0066965 | 0.0200894 | 0.0647323 | 0.149554 | 4.3683 | 0.120536 | opt-1.3b_seed4025_step000494_ep000494 |
| OPT-1.3B | 5025 | FF-Endpoint | 654 | 0.481667 | 3.1808 | 0.0200891 | 0.0401785 | 0.102679 | 0.433036 | 4.09375 | 0.0267854 | opt-1.3b_seed5025_step000654_ep000654 |
| OPT-1.3B | 5025 | PAS-S3025 | 820 | 0.463333 | 3.35938 | -0.00892879 | -0.0111609 | 0.00892856 | 0.314732 | 4.6875 | 0.620535 | opt-1.3b_seed5025_step000820_ep000820 |
| OPT-1.3B | 5025 | PAS-S3050 | 800 | 0.488333 | 3.40848 | -0.00892861 | -0.0111609 | 0.015625 | 0.375 | 4.53571 | 0.46875 | opt-1.3b_seed5025_step000800_ep000800 |
| OPT-1.3B | 5025 | PAS-S31 | 915 | 0.48 | 3.37723 | 0.0111606 | 0.0111606 | 0.00223205 | 0.354911 | 4.57589 | 0.508928 | opt-1.3b_seed5025_step000915_ep000915 |
| OPT-1.3B | 5025 | PAS-S35 | 820 | 0.463333 | 3.35938 | -0.00892879 | -0.0111609 | 0.00892856 | 0.314732 | 4.6875 | 0.620535 | opt-1.3b_seed5025_step000820_ep000820 |
| OPT-1.3B | 5025 | Oracle40 | 575 | 0.476667 | 3.27232 | 0.0200894 | 0.0357144 | 0.0870537 | 0.470982 | 4.06696 | 0 | opt-1.3b_seed5025_step000575_ep000575 |
| OPT-2.7B | 7025 | FF-Endpoint | 950 | 0.483333 | 3.22991 | 0.015625 | 0.03125 | 0.0424107 | 0.314732 | 3.9442 | 0 | opt-2.7b_seed7025_step000950_ep000950 |
| OPT-2.7B | 7025 | PAS-S3025 | 987 | 0.506667 | 3.74777 | -0.0535712 | -0.0133927 | -0.0424106 | 1.00223 | 5.54018 | 1.59598 | opt-2.7b_seed7025_step000987_ep000987 |
| OPT-2.7B | 7025 | PAS-S3050 | 949 | 0.481667 | 3.78348 | -0.0491071 | -0.029018 | 0.622768 | 0.859375 | 6.29018 | 2.34598 | opt-2.7b_seed7025_step000949_ep000949 |
| OPT-2.7B | 7025 | PAS-S31 | 987 | 0.506667 | 3.74777 | -0.0535712 | -0.0133927 | -0.0424106 | 1.00223 | 5.54018 | 1.59598 | opt-2.7b_seed7025_step000987_ep000987 |
| OPT-2.7B | 7025 | PAS-S35 | 549 | 0.44 | 3.46429 | 0.0200894 | 0.049107 | 0.0691964 | 0.212054 | 4.02902 | 0.0848212 | opt-2.7b_seed7025_step000549_ep000549 |
| OPT-2.7B | 7025 | Oracle40 | 950 | 0.483333 | 3.22991 | 0.015625 | 0.03125 | 0.0424107 | 0.314732 | 3.9442 | 0 | opt-2.7b_seed7025_step000950_ep000950 |
| OPT-2.7B | 8025 | FF-Endpoint | 976 | 0.44 | 3.42188 | 0.0133927 | 0.0424106 | 0.080357 | 0.430804 | 4.63393 | 0.174107 | opt-2.7b_seed8025_step000976_ep000976 |
| OPT-2.7B | 8025 | PAS-S3025 | 987 | 0.481667 | 3.96205 | -0.0111608 | 0.0178571 | 0.125 | 0.787946 | 5.72321 | 1.26339 | opt-2.7b_seed8025_step000987_ep000987 |
| OPT-2.7B | 8025 | PAS-S3050 | 617 | 0.501667 | 3.84375 | 0.00223207 | -0.00446438 | 0.0178571 | 0.495536 | 5.12054 | 0.660714 | opt-2.7b_seed8025_step000617_ep000617 |
| OPT-2.7B | 8025 | PAS-S31 | 617 | 0.501667 | 3.84375 | 0.00223207 | -0.00446438 | 0.0178571 | 0.495536 | 5.12054 | 0.660714 | opt-2.7b_seed8025_step000617_ep000617 |
| OPT-2.7B | 8025 | PAS-S35 | 815 | 0.435 | 3.92857 | 0.00892856 | 0.0178571 | 0.03125 | 0.145089 | 4.54464 | 0.0848212 | opt-2.7b_seed8025_step000815_ep000815 |
| OPT-2.7B | 8025 | Oracle40 | 978 | 0.423333 | 3.73214 | 0.0245538 | 0.0401788 | 0.127232 | 0.334822 | 4.45982 | 0 | opt-2.7b_seed8025_step000978_ep000978 |

## Copied Source Files

| setting | label | repo copy | source artifact |
| --- | --- | --- | --- |
| opt13b_seed4025 | path30_35_40 | `docs/pas_radius_downstream_20260528/opt13b_seed4025/path30_35_40.md` | `/workspace/ckpts/pas_informative_radius/opt-1.3b_seed4025_ff1000_growth_path30_35_40_fixed5_no6/path30_35_40_fixed5_summary.md` |
| opt13b_seed4025 | local_radius_vs_40 | `docs/pas_radius_downstream_20260528/opt13b_seed4025/local_radius_vs_40.md` | `/workspace/ckpts/pas_informative_radius/opt-1.3b_seed4025_ff1000_growth_local_radius_fixed5_no6/local_radius_vs_40_fixed5.md` |
| opt13b_seed4025 | downstream30 | `docs/pas_radius_downstream_20260528/opt13b_seed4025/downstream30.md` | `/workspace/ckpts/pas_informative_radius/opt-1.3b_seed4025_ff1000_growth_downstream30_local_radius_fixed5_no6/local_radius_downstream30_table.md` |
| opt13b_seed5025 | final | `docs/pas_radius_downstream_20260528/opt13b_seed5025/final.md` | `/workspace/ckpts/pas_informative_radius/opt-1.3b_seed5025_ff1000_growth_rep1/FINAL_SUMMARY.md` |
| opt27b_seed7025 | final | `docs/pas_radius_downstream_20260528/opt27b_seed7025/final.md` | `/workspace/ckpts/pas_informative_radius/opt-2.7b_seed7025_ff1000_growth_rep1/FINAL_SUMMARY.md` |
| opt27b_seed8025 | final | `docs/pas_radius_downstream_20260528/opt27b_seed8025/final.md` | `/workspace/ckpts/pas_informative_radius/opt-2.7b_seed8025_ff1000_growth_rep1/FINAL_SUMMARY.md` |
| formal_tables | formal_summary | `docs/pas_radius_downstream_20260528/formal_tables_20260525/formal_summary.md` | `/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/summary.md` |
| formal_tables | endpoint_ambiguity | `docs/pas_radius_downstream_20260528/formal_tables_20260525/endpoint_ambiguity.md` | `/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/paper_endpoint_ambiguity_table.md` |
| formal_tables | stress_correlation | `docs/pas_radius_downstream_20260528/formal_tables_20260525/stress_correlation.md` | `/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/paper_stress_correlation_table.md` |
| formal_tables | selection_value | `docs/pas_radius_downstream_20260528/formal_tables_20260525/selection_value.md` | `/workspace/ckpts/pas_informative_radius/paper_formal_tables_20260525/paper_selection_value_table.md` |

## Candidate Prefix Check

All copied setting summaries passed the candidate-prefix check.
