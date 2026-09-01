# DOptDRPSNN-EMS — IoT-Enabled Smart Building Energy Management

Reproducibility package for the paper:

> **IoT-Enabled Smart Building Energy Management with Integrated Load Forecasting, Anomaly Detection, and Control**
> Ahmad H. Milyani, Arbab Mohammad Kamal, Atif Mohammad Kamal, Amjad Ali Pasha,
> Tauheed Khan Mohd. Submitted for publication.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

---

## What this repository contains

A complete, executable forecast–anomaly–control pipeline for building energy management,
together with **the exact raw per-seed outputs used to construct every mean and standard
deviation reported in the paper**. Nothing in the manuscript is reported that this package
cannot regenerate.

| Path | Contents |
|---|---|
| `data/` | Synthetic benchmark (12 buildings × 1,376 hourly rows), anomaly distribution, dataset summary |
| `scripts/` | Executable experiment code (see *Reproducing the results* below) |
| `results/` | Raw per-seed outputs, frozen configuration, confusion matrices, statistical tests |
| `tables/` | Frozen manuscript-table summaries (`table_01.csv` … `table_18.csv`) |
| `figures/` | Final main (`main/`) and supplementary (`supplementary/`) figures |
| `docs/` | Provenance map, key claims and limits, package manifest, verification report |
| `paper/` | Manuscript (typeset and submission DOCX), supplementary material |

## What is and is not claimed

This package supports a **stability-oriented, causally closed BEMS workflow**, not a
forecasting leaderboard win. Specifically:

- ✅ **Claimed:** significantly lower initialization variance than an exact encoder-matched
  GRU head (10 seeds; RMSE SD 0.189 vs 1.733 kWh; *F* = 83.91, one-sided *p* = 1.34 × 10⁻⁷;
  Brown–Forsythe *p* = 0.0028).
- ✅ **Claimed:** cross-building anomaly transfer (unseen-building F1 = 0.632, AUPRC = 0.749,
  AUROC = 0.904) with a measurable PSNN spike-rate contribution (F1 drops to 0.537 without it).
- ✅ **Claimed:** a genuinely forecast-driven, anomaly-gated controller that executes causally.
- ❌ **Not claimed:** best one-hour or six-hour forecasting accuracy. A tuned Transformer is
  more accurate at one hour (19.90 vs 26.95 kWh RMSE), and several baselines beat the proposed
  model at six hours. See `docs/KEY_CLAIMS_AND_LIMITS.md`.

## Data provenance

The benchmark is **synthetic**, generated from DOE commercial-reference-building archetypes as
structural priors. It is *not* ASHRAE GEPIII or BDG2 data, and no ASHRAE/BDG2 numerical result
is reported as if produced by this package. Synthetic generation is deliberate: it is what makes
anomaly ground truth, counterfactual clean load, and control evaluation possible at all.

Integrity check for the benchmark file:

```bash
sha256sum -c <<< "$(cat results/data_sha256.txt | awk '{print $1}')  data/synthetic_benchmark.csv"
```

Expected digest: `893509d55c764b4664275e7a1a64119d5a5c54bf4e56295702d74a44f963b8bc`

## Environment

Results in the paper were produced CPU-only on an AMD EPYC 9V74 with Python 3.13.5 and
PyTorch 2.10.0+cpu (full environment in `results/environment.json`).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml && conda activate doptdrpsnn
```

## Reproducing the results

Run in this order, or simply `make all`.

| Step | Command | Produces | Approx. CPU time |
|---|---|---|---|
| 1 | `python scripts/run_canonical_s105.py` | Forecasting, ablations, statistics, latency, sensitivity | ~40–90 min |
| 2 | `python scripts/run_6h_all.py` | Six-hour horizon results | ~15–30 min |
| 3 | `python scripts/run_stability_10seeds.py` | Ten-seed encoder-matched stability | ~30–60 min |
| 4 | `python scripts/run_anomaly_control.py` | Anomaly head, thresholds, supervisory control | ~10–20 min |
| 5 | `python scripts/make_figures.py` | All main and supplementary figures | ~1 min |
| 6 | `python scripts/verify_raw_results.py` | Checks raw outputs against manuscript summaries | seconds |

`scripts/run_all.py` chains steps 1–5. Timings are indicative for a CPU-only machine and scale
roughly linearly with core count.

**Determinism.** Seeds 11, 23 and 37 drive the primary tables; the stability experiment adds
41, 53, 67, 71, 83, 97 and 109. Torch and NumPy seeds are set per run. Small last-digit
differences across BLAS implementations and PyTorch patch versions are expected and do not
change any reported conclusion.

## Verifying without re-running

`results/` already contains every raw output. To confirm the manuscript numbers without
spending CPU time:

```bash
python scripts/verify_raw_results.py
```

This recomputes each reported mean and SD from the per-seed CSVs and reports any mismatch.

## Figure and table provenance

`docs/FINAL_FIGURE_MAP.md` maps every figure to its generating script and source CSVs.
`docs/RESULT_PROVENANCE.md` does the same for tables. `docs/PACKAGE_MANIFEST.csv` lists every
file with its role.

## Licence

- **Code** (`scripts/`): MIT — see [`LICENSE`](LICENSE).
- **Data, figures, tables and documentation**: CC BY 4.0 — see [`LICENSE-DATA`](LICENSE-DATA).

## Citation

See [`CITATION.cff`](CITATION.cff), or cite as:

```bibtex
@article{milyani_doptdrpsnn_ems,
  title  = {IoT-Enabled Smart Building Energy Management with Integrated Load
            Forecasting, Anomaly Detection, and Control},
  author = {Milyani, Ahmad H. and Kamal, Arbab Mohammad and Kamal, Atif Mohammad
            and Pasha, Amjad Ali and Khan Mohd, Tauheed},
  note   = {Submitted for publication}
}
```

Journal, year and article DOI will be added on acceptance.

Please also cite the archived software release: DOI [10.5281/zenodo.XXXXXXX].

## Contact

Tauheed Khan Mohd (corresponding author) — tkhanmoh@emich.edu. Issues and questions are also welcome via the
[issue tracker](../../issues).
