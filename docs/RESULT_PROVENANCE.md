# Manuscript Result Provenance

The `tables/` directory stores the final manuscript table cells as presentation summaries. The sources below are the numerical/raw evidence from which those tables are derived.

| Manuscript item | Raw source(s) | Generating script |
|---|---|---|
| Table 1 - related-work positioning | cited literature in manuscript | not an experimental table |
| Table 2 - executed benchmark | `data/synthetic_benchmark.csv`, `data/dataset_summary.csv` | benchmark generator described in manuscript; supplied data frozen |
| Table 3 - candidate variables | `data/synthetic_benchmark.csv`, `results/canonical_configuration.json` | `run_canonical_s105.py` |
| Table 4 - leakage-control partitions | `results/split_protocol.csv` | `run_canonical_s105.py` |
| Table 5 - anomaly ground truth | `results/anomaly_ground_truth.csv` | `run_canonical_s105.py` |
| Table 6 - implementation settings | `results/canonical_configuration.json`, `results/environment.json`, `results/baseline_tuned_hyperparameters.json` | `run_canonical_s105.py` |
| Table 7 - CPU latency | `results/latency_canonical.csv` | `run_canonical_s105.py` |
| Table 8 - one-hour forecasting | `results/forecasting_models.csv`, `results/classical_unseen_full.csv` | `run_canonical_s105.py` |
| Table 9 - six-hour forecasting | `results/forecast_horizon_6h.csv` | `run_6h_all.py` |
| Table 10 - Holm-corrected paired tests | `results/statistical_tests_holm.csv`; `results/daily_block_mae_raw.csv` is emitted on full rerun | `run_canonical_s105.py` |
| Table 11 - RF cross-building diagnostics | `results/rf_cross_building_diagnostics.csv` | `run_canonical_s105.py` |
| Table 12 - feature selection | `results/feature_selection_canonical.csv` | `run_canonical_s105.py` |
| Table 13 - optimizer roles | `results/optimizer_roles_canonical.csv` | `run_canonical_s105.py` |
| Table 14 - missing-input stress test | `results/missingness_cross_model.csv` | `run_canonical_s105.py` |
| 10-seed PSNN stability claim | `results/stability_10seeds_encoder_matched.csv`, `results/stability_variance_tests.csv` | `run_stability_10seeds.py` |
| Table 15 - anomaly metrics | `results/anomaly_results_final.csv`, `results/anomaly_row_predictions_*.csv` | `run_anomaly_control.py` |
| Table 16 - fault-type detection | `results/anomaly_per_type_unseen_full.csv` | `run_anomaly_control.py` |
| Table 17 - sensitivity | `results/sensitivity_canonical.csv`, `results/anomaly_threshold_sensitivity_final.csv` | `run_canonical_s105.py`, `run_anomaly_control.py` |
| Table 18 - closed-loop control | `results/control_results_final.csv`, `results/control_trace_primary.csv` | `run_anomaly_control.py` |
| Figure 1 | `results/anomaly_per_type_unseen_full.csv` | `make_figures.py` |
| Figure 2 | static architecture diagram | manually authored diagram |
| Figure 3 | `results/forecasting_models.csv`, `results/classical_unseen_full.csv` | `make_figures.py` |
| Figure 4 | `results/forecast_horizon_6h.csv` | `make_figures.py` |
| Figure 5 | `results/feature_selection_canonical.csv` | `make_figures.py` |
| Figure 6 | `results/optimizer_roles_canonical.csv` | `make_figures.py` |
| Figure 7 | `results/stability_10seeds_encoder_matched.csv` | `make_figures.py` |
| Figure 8 | `results/control_results_final.csv` | `make_figures.py` |

`python scripts/verify_raw_results.py` performs a fast audit that recomputes the principal numerical summaries from the raw files and compares them with the final manuscript table summaries.
