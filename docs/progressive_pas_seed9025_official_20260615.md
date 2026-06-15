# Progressive PAS Seed9025 Official Replay 2026-06-15

## Current Local Status

- Local machine has code/scripts only; GPU/model/data artifacts are on server.
- Known prior partial result from `docs/PROGRESSIVE_PAS_LOOKAHEAD_PARTIAL_2026_05_28.md`:
  - model: `OPT-2.7B`
  - seed: `9025`
  - train episodes: `2000`
  - replay top-k: `5`
  - completed gate checks: `12`
  - PROMOTE: `0`
  - raw PAS improvements: `0`
  - completed probe batches: `24 / 60`
- Official gate after this patch:
  - PROMOTE iff `candidate_count >= promotion_min_candidates`
  - and `endpoint_price <= epsilon`
  - and `lookahead_gain >= margin`
  - otherwise HOLD and select the FF endpoint.

## Server Command: Finish Existing Replay

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main
git rev-parse --short HEAD

MODEL_NAME=opt-2.7b \
MODEL=/workspace/Models/opt-2.7b \
SEED=9025 \
TRAIN_EPISODES=2000 \
SEARCH_GPU=4 \
GPU_IDS="4,5,6,7" \
REPLAY_TOP_K=5 \
PREFIX_STEPS="300,500,700,1000,1500,2000" \
STAGES="0.05,0.10,0.15,0.20,0.25,0.30" \
EPSILON=0.05 \
MARGIN=0.00 \
PROMOTION_MODE=official \
CARRY_FORWARD_MODE=pas \
RUN_ID_OVERRIDE=progressive_pas_lookahead_seed9025_ep2000_gpu4_fixed0.05_staircase \
bash scripts/run_progressive_pas_lookahead_server.sh
```

Expected output directory:

```text
/workspace/ckpts/pas_progressive_lookahead/opt-2.7b_seed9025_ep2000_fixed0.05_staircase/replay
```

Required files:

```text
progressive_pas_selection.csv
progressive_pas_promotion_gate.csv
progressive_pas_efficiency.md
progressive_pas_partial_report.md
progressive_pas_manifest.json
```

## Server Command: Same-Pool Ablation

Run only after the existing seed9025 candidate pool is present. This command
does not launch a new PPO search.

```bash
cd /workspace/structure_pruning
git pull --ff-only origin main
git rev-parse --short HEAD

MODEL_NAME=opt-2.7b \
MODEL=/workspace/Models/opt-2.7b \
SEED=9025 \
TRAIN_EPISODES=2000 \
GPU_IDS="4,5,6,7" \
PREFIX_STEPS="300,500,700,1000,1500,2000" \
STAGES="0.05,0.10,0.15,0.20,0.25,0.30" \
CARRY_FORWARD_MODE=none \
bash scripts/run_progressive_pas_seed9025_ablation_server.sh
```

Default candidate pool:

```text
/workspace/ckpts/opt-2.7b/sparsity_0.30/progressive_pas_lookahead_seed9025_ep2000_gpu4_fixed0.05_staircase/candidates
```

If the existing server pool lives elsewhere, rerun with:

```bash
CANDIDATE_DIR_OVERRIDE=/actual/server/path/to/candidates \
bash scripts/run_progressive_pas_seed9025_ablation_server.sh
```

Expected ablation root:

```text
/workspace/ckpts/pas_progressive_lookahead/opt-2.7b_seed9025_ep2000_fixed0.05_staircase/ablation_official_20260615
```

Required ablation outputs:

```text
progressive_pas_ablation_summary.csv
progressive_pas_ablation_summary.md
progressive_pas_ablation_manifest.json
topk20_eps0p05_margin0p00/progressive_pas_selection.csv
topk20_eps0p10_margin0p00/progressive_pas_selection.csv
topk50_eps0p05_margin0p00/progressive_pas_selection.csv
topk50_eps0p10_margin0p00/progressive_pas_selection.csv
```

## Decision Rule

If `topk50_eps0p10_margin0p00` has `raw_lookahead_gain <= 0` for every gate
row, stop Progressive PAS as a main paper line and keep PAS diagnostic-only.
