# Calibration source audit

This directory audits the retrieved R1 source and supplies detector-calibration masks for the R2 sensitivity experiments. It does not modify the retrieved source, fit a forecaster, or choose a calibration cut from held-out results.

## Reproduce the source and calibration audit

```bash
python calibration_audit/prepare_audit.py /path/to/research
```

Run from the parent anomaly-replay directory, or use the complete script path from the repository root. The script creates:

- `raw_feature_provenance.csv`: direct checks against within-building observed-meter shifts.
- `validation_row_provenance.csv`: the canonical validation-row order and conservative raw-meter dependency bounds.
- `calibration_assignments.csv`: all six predeclared cut/purge conditions, with each row assigned to fit, threshold selection, or purge.
- `calibration_boundaries.csv`: target-time and raw-support boundaries and retained counts.
- `calibration_fault_coverage.csv`: normal and fault-type counts in each phase, including zero counts.
- `provenance_and_LR_audit.json`: source hashes, raw support derivation, actual archived LR settings, fitting-only scaler checks, and threshold-search differences.
- `archived_logistic_metrics.csv`: the unchanged archived supervised-comparator results.

## Dependency and purge interpretation

The actual canonical code uses `X.append(a[i-L:i])` with `L=24`. The selected features include `lag24`. Each assessable lag in the supplied CSV exactly equals the corresponding within-building observed-meter shift. The rolling feature equals `meter.shift(1).rolling(24)` to floating-point precision, rather than a current-inclusive rolling mean.

For a neural prediction targeting hour `t`, the oldest input row is `t-24`; its `lag24` consumes the observed meter at `t-48`. The complete anomaly row also uses current observed meter `t`. Its conservative meter footprint is therefore `t-48` through `t`, or 49 timestamp positions. To make fit and threshold raw history disjoint, a first threshold target must be at least 49 hours after the last fit target. A 24-hour purge alone is insufficient. The balanced minimum purge removes 24 target rows at the end of fitting and 24 at the start of threshold selection.

The persistence/common-11 detector has a shorter intrinsic meter history, through its direct lag-24 feature. Using the neural maximum history for all four workflow rows preserves identical fitting-label availability and is a conservative common comparison.

| Nominal cut | Unpurged fit / threshold per building | Strict disjoint fit / threshold per building |
| --- | ---: | ---: |
| 50% | 91 / 91 | 67 / 67 |
| 60% | 109 / 73 | 85 / 49 |
| 70% | 127 / 55 | 103 / 31 |

The nominal cut is defined before purging. These are alternative development allocations, not candidate cuts from which to choose the best held-out result. Direct shared history is a source of dependence; causal observation reuse is not automatically future leakage. Even strict raw-history separation does not remove serial dependence, shared weather, or common trained-model uncertainty.

## Logistic-regression fairness audit

The archived LR comparator uses `StandardScaler` fitted on all 981 detector-fitting rows, including 79 fault rows; the 657 threshold rows are excluded from scaling and fitting. Its configuration is `C=1`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=2000`, and seed 11. The saved scaler moments exactly agree with the fitting-only feature tensor. The selected threshold is 0.9820489046507991. This is a classifier score rather than a reliability-calibrated physical fault probability.

Fitting-label availability is comparable with gradient boosting, but the archived threshold search is different: LR maximizes validation F1 over distinct score values, while gradient boosting uses 230 points between 0.03 and 0.60. Any new fair-tuning sensitivity should predeclare configurations and use the same development-only threshold rule for both. All archived results should remain available.

## Same-host cost benchmark

`benchmark_pipeline_cost.py` reads trained checkpoints and extracts only model class definitions by AST, avoiding source-module import side effects. A supplied JSON maps the four pipeline names to their fitted gradient-boosting heads. Timing uses a fixed normalized batch-one input and corresponding observed-meter/sensor context, one CPU thread, 100 warmups and 1,000 timed passes for each of three repeats.

The two reported scopes are (1) three-member forecaster ensemble plus load reconstruction and (2) forecaster plus anomaly-feature construction and detector scoring. Telemetry acquisition, preprocessing, transport, fitting and supervisory rule execution are excluded. The available server CPU is not an IoT gateway. Neural parameter/tensor and serialized bytes are separated from detector tree storage and total process memory.
