# Manifest Generation Notes

- generated_at_utc: `2026-06-15T11:35:38Z`
- generator_commit: `5b2746c`
- generator_script: `scripts/build_fastforward_run_manifest.py`
- output_dir: `docs/fastforward_clean_manifest_20260615/`

## Status Rules

- `paper_usable`: complete metadata and raw artifacts are mapped. No inspected row met this bar yet.
- `diagnostic_only`: useful mechanism/PAS/static-stress evidence, but not final FastForward main-table evidence.
- `partial_negative`: incomplete, weak, mixed, or negative partial PAS evidence.
- `debug_only`: smoke tests, tiny samples/episodes, or incomplete high-PPL logs.
- `missing_raw_artifact`: PDF or documented result without a complete raw-artifact/log mapping in inspected sources.

## Important Guardrails

- PDF table values are recorded as PDF claims, not clean reproductions.
- `../opt13-log.txt` is not paper-usable because exact protocol, seed, final converged row, and model identity are incomplete.
- PAS static/path evidence is diagnostic unless regenerated and tied to final journal tables.
- Progressive PAS seed9025 remains partial/negative until the official replay and same-pool ablation finish on the server.

## Status Counts

- debug_only: `2`
- diagnostic_only: `23`
- missing_raw_artifact: `47`
- paper_usable: `0`
- partial_negative: `7`
