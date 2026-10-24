# R2 calibration and complete-workflow evidence

This directory adds detector-calibration sensitivity, supervised-comparator checks and four complete synthetic forecast–detector–replay workflows. It reuses the exact frozen R1 data, three-member PSNN/GRU checkpoints and forecast snapshots. It does not regenerate the historical synthetic benchmark, recover historical MOA selection, or claim physical BAS safety.

The principal numerical findings and relative artifact paths are in `FACTS.json` and `REPORT.md`. Main-paper tables are in `tables/`; source/lag/LR audits are in `calibration_audit/`; detector models, per-row scores and selection records are in `results/`; replay traces and decision records are in `replay/pipeline_comparison/`.

## Reproduce

Pass the extracted R1 **research** directory, not its parent folder:

```bash
python -B run_all.py --source /path/to/frozen_R1/research
```

For example, from the overall workspace root:

```bash
python -B r2_experiments/anomaly_replay/run_all.py --source sources/r1_package/JER_S105_Revision_20261012/research
```

The script audits raw features, fits and freezes development-only detector choices, computes the held-out sensitivities, runs four prospective replays, verifies the saved heads/scores, measures inference costs and regenerates tables/figures. The source package is read-only. The primary 60/40 causal-unpurged configuration remains fixed; none of the alternative cuts is selected from held-out results.

`--skip-cost` reuses existing cost summaries and skips timing repetitions; use it only when those summaries already exist. Timing results naturally vary by host and load. The reproducible R1 numerical data/model outputs and the same-host timing context are separately identified.

The portable replay manifest stores prediction/score paths relative to its own directory. Trained estimator paths in the selection manifest are relative to `results/`. Absolute source/output paths elsewhere are provenance records, rather than required hard-coded execution paths. The source and checkpoints are identified by SHA-256; required package versions are recorded in `results/selection_freeze.json` and the timing configuration.

## Verification

```bash
python -B verify_detector_outputs.py --results results
```

The verifier reconstructs all 126 saved detector/split combinations from serialized fitted estimators and retained input tensors, checks metrics/confusion counts/decisions, verifies the selection manifest freeze, and verifies common-feature alignment. `results/R1_anomaly_reproduction.json` confirms exact R1 primary scores and decisions. Replay verification is supplied in `replay/pipeline_comparison/pipeline_verification_report.json`; the canonical replay reproduces R1 exactly.

The six conditions comprise nominal 50/50, 60/40 and 70/30 cuts, each causal-unpurged or strictly raw-history-disjoint. The audited neural inference footprint spans meter observations t−48 through t. Strict separation removes 24 target rows on each side of the nominal boundary, leaving 49 hours between the last fitting and first threshold target. This removes direct raw inference-history reuse; common trained weights, serial correlation and shared weather remain. Purging also changes sample size and fault mix.

The supervised fairness sensitivity uses the same labelled fitting rows, eight development-only candidates per method and the same distinct-score F1 threshold rule. Candidate AP and threshold F1 use the same development selection block, which is disclosed rather than called independent reliability calibration. The original frozen head remains the principal pipeline, and the original baseline is retained.

Every replay uses prior-released anomaly scores and occupancy with the next-interval forecast and known tariff. All four workflows share 549 target building-hours, the same rule/rebound accounting and full-period score availability. Synthetic fault-label alignment is a post-hoc description, not a measure of harmful actions or protective benefit. The counterfactual replay does not feed simulated actions into subsequent observed-meter contexts.
