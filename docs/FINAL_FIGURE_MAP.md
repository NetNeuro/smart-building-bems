# Final figure provenance map

| Figure | Final visual | Primary source(s) |
|---|---|---|
| 1 | Overall forecast-anomaly-control framework | `scripts/run_canonical_s105.py`, `scripts/run_anomaly_control.py`, `results/canonical_configuration.json`, manuscript Secs. 3-6 |
| 2 | Leakage-aware partition/evaluation workflow | `results/split_protocol.csv`, `scripts/run_canonical_s105.py` |
| 3 | DOptDRPSNN executable architecture | model classes `RAB`, `HDRAB`, `DRA`, `PSNN`, `Proposed` in `scripts/run_canonical_s105.py`; `results/canonical_configuration.json` |
| 4 | 1 h and 6 h forecasting | `results/forecasting_models.csv`, `results/classical_unseen_full.csv`, `results/forecast_horizon_6h.csv` |
| 5 | MOA/DOA ablations | `results/feature_selection_canonical.csv`, `results/optimizer_roles_canonical.csv` |
| 6 | 10-seed initialization stability | `results/stability_10seeds_encoder_matched.csv`, `results/stability_variance_tests.csv` |
| 7 | Anomaly identification | `results/spike_rate_ablation.csv`, `results/anomaly_per_type_unseen_full.csv`, `results/anomaly_results_final.csv` |
| 8 | Supervisory control | `results/control_results_final.csv`, `results/control_trace_primary.csv` |
| S1 | Representative load profiles | `data/synthetic_benchmark.csv` |
| S2 | Environmental/operational profiles | `data/synthetic_benchmark.csv` |
| S3 | Building archetypes / evaluation roles | `data/synthetic_benchmark.csv`, `results/split_protocol.csv` |
| S4 | Input-masking stress test | `results/missingness_cross_model.csv` |

The legacy Round-3 graphics generated before the harmonization pass are preserved under `figures_source_round3/` for audit history only. The manuscript uses only files under `figures/main/`; the supplementary document uses only `figures/supplementary/`.
