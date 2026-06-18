# FastForward Journal Execution Lock 2026-06-18

Scope: finish FastForward journal submission experiments. Do not expand PAS.

## Immediate Stop

Progressive PAS is no longer a main-line experiment. Stop any currently running
PAS replay/probe jobs and preserve existing outputs as diagnostic artifacts.

Server stop commands:

```bash
ps -ef | grep -E '[p]as_progressive_lookahead_replay|[a]mc_searchPPO.py --job=probe'
pkill -u "$USER" -f 'pas_progressive_lookahead_replay.py' || true
pkill -u "$USER" -f 'amc_searchPPO.py --job=probe' || true
ps -ef | grep -E '[p]as_progressive_lookahead_replay|[a]mc_searchPPO.py --job=probe' || true
```

If the partial PAS ablation directory exists, keep it. Do not delete or rerun it.
Record it as `diagnostic_only` or `partial_negative` depending on completed rows.

## P0 Only

These are the only mandatory experiment workstreams.

1. Clean main-table verification
   - Models: OPT-1.3B, OPT-2.7B, LLaMA-7B.
   - Sparsities: 20% and 30%.
   - Recheck: FF raw, FF + calibration, dense, and main baselines.
   - Goal: ensure no debug, unconverged, or wrong-episode runs enter paper tables.

2. One-model core ablation
   - Preferred: LLaMA-7B at 30%.
   - Fallback if too slow: OPT-2.7B at 30%.
   - Variants: Full FastForward, Uniform + Calib, soft allocation/no exact
     projector, w/o sparsity curriculum, w/o fidelity curriculum, w/o transfer,
     w/o calibration.
   - Goal: isolate projector, progressive schedule, transfer, and calibration.

3. Search-cost table
   - From clean logs only.
   - Required fields: GPU count, total episodes, per-GPU episodes, search
     GPU-hours, calibration/eval GPU-hours, final PPL.
   - Compare at least: FF, EAS/zero-order search, FLAP, SliceGPT.

4. Mechanism plots
   - PPL vs episode.
   - Reward vs episode.
   - Search cost vs PPL.
   - Policy retention heatmap.
   - Calibration before/after PPL.

## P1 Only If P0 Is Clean

1. Multi-seed stability
   - Only OPT-2.7B at 30% or LLaMA-7B at 30%.
   - Three seeds.
   - Record Full FF PPL, cost, and selected policy stability.

2. Downstream sanity check
   - Only final Full FF + Calib and one or two baselines.
   - Tasks: PIQA, HellaSwag, WinoGrande, BoolQ, ARC-e, ARC-c.
   - Goal: sanity check, not a new downstream theory.

## Do Not Run

- Do not continue or expand PAS after current partial artifacts are frozen.
- Do not run LoRA recovery.
- Do not run 30.25/30.5/31 stress sweeps.
- Do not build a downstream-correlation story.
- Do not add new model families until all P0 rows are clean.

## Required Metadata For Every Artifact

Every run used in a table or figure must record:

- commit hash
- model and model path
- seed
- GPU IDs and GPU count
- total episodes
- per-GPU episodes
- eval samples
- dataset path/config
- checkpoint path
- candidate/artifact directory
- calibration recipe
- wall-clock time or start/end timestamps
- final PPL
- status: `paper_usable`, `diagnostic_only`, `partial_negative`, `debug_only`,
  or `missing_raw_artifact`

