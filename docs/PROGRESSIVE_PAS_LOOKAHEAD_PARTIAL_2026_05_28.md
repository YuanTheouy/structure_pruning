# Progressive PAS Lookahead Partial Result 2026-05-28

This note records the in-progress server result for the Progressive PAS Lookahead / PAS promotion-gate experiment.

## Setting

- Model: OPT-2.7B
- Seed: 9025
- Search budget: 2000 episodes
- Search data: fixed 5% WikiText2 validation subset
- Pruning schedule: staircase, 5% -> 10% -> 15% -> 20% -> 25% -> 30%
- Search GPU: 4
- Replay/probe GPUs: 4,5,6,7
- PAS replay top-k: 5
- Promotion gate mode: simple
- Carry-forward mode: PAS candidate only
- Replay directory: `/workspace/ckpts/pas_progressive_lookahead/opt-2.7b_seed9025_ep2000_fixed0.05_staircase/replay`

## Partial Snapshot

- Completed gate checks: 12
- Decisions: PROMOTE=0, HOLD=12
- Raw PAS improvements: 0
- Best raw gain so far: 0
- Completed probe batches: 24 / 60
- Prefixes seen: 300, 500, 700, 1000
- Stages seen: 5%, 10%, 15%, 20%

## Compact Gate Table

| prefix | stage transition | decision | raw_gain | raw candidate | selected candidate | hold reason |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 5% -> 10% | HOLD | 0 | step18 | step18 | lookahead_gain <= 0 |
| 300 | 10% -> 15% | HOLD | 0 | step260 | step260 | lookahead_gain <= 0 |
| 500 | 5% -> 10% | HOLD | 0 | step18 | step18 | lookahead_gain <= 0 |
| 500 | 10% -> 15% | HOLD | 0 | step260 | step260 | lookahead_gain <= 0 |
| 500 | 15% -> 20% | HOLD | 0 | step416 | step416 | lookahead_gain <= 0 |
| 700 | 5% -> 10% | HOLD | 0 | step18 | step18 | lookahead_gain <= 0 |
| 700 | 10% -> 15% | HOLD | 0 | step260 | step260 | lookahead_gain <= 0 |
| 700 | 15% -> 20% | HOLD | 0 | step416 | step416 | lookahead_gain <= 0 |
| 700 | 20% -> 25% | HOLD | 0 | step631 | step631 | lookahead_gain <= 0 |
| 1000 | 5% -> 10% | HOLD | 0 | step18 | step18 | lookahead_gain <= 0 |
| 1000 | 10% -> 15% | HOLD | 0 | step260 | step260 | lookahead_gain <= 0 |
| 1000 | 15% -> 20% | HOLD | 0 | step416 | step416 | lookahead_gain <= 0 |

## Current Interpretation

So far, this run does **not** show PAS acceleration signal.

For every completed gate check, raw PAS selects the same candidate as the FF stage endpoint. This means the current HOLD decisions are not just caused by an overly strict gate; the raw lookahead selector itself has not found a better next-stage candidate in the checked prefixes/stages.

This is still a partial result. The most important remaining checks are later prefixes and higher transitions:

- prefix 1000 / 1500 / 2000
- 20% -> 25%
- 25% -> 30%

If those later checks also show PROMOTE=0 and raw_gain=0, the PAS promotion-gate acceleration direction should be treated as unsupported for this setting.

