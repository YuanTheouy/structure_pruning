# PAS Formal Tables Refresh

## Endpoint Ambiguity

| model | seed | target_to_heldout | endpoint_is_oracle | endpoint_candidate | oracle_candidate | endpoint_L30 | endpoint_L40 | endpoint_Regret40 | oracle_L40 |

## Stress Correlation

| model | seed | target_probe_heldout | n | pearson_S35_Delta40 | spearman_S35_Delta40 | partial_S35_L40_given_L30 | partial_S35_Regret40_given_L30 | partial_S35_Delta40_given_L30 |

## Selection Value

| model | seed | endpoint_is_oracle | FF_regret | PAS_Stress_regret | random_regret_mean | random_regret_std | oracle_regret | PAS_Stress_candidate |

## Auto Decision

- endpoint-ambiguous settings: `2`
- ambiguous settings where PAS-Stress reduces FF regret: `2`
- run new OPT-2.7B seed: `True`
