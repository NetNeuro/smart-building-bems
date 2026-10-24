# Integrated Load Forecasting and Anomaly-Gated Supervisory Replay for Building Energy Management

R2 scientific reproducibility release **v2.0.1** — analysis version
**R2-2026-10-22**, public distribution prepared 25 October 2026.

Authors: Ahmad H. Milyani; Md Mottahir Alam; Atif Mohammad Kamal; Amjad Ali
Pasha; Tauheed Khan Mohd.

## Get the complete R2 package

The R2 source, metadata and complete archive are now available in the
[v2.0.1 GitHub release](https://github.com/NetNeuro/smart-building-bems/releases/tag/v2.0.1).
Download the attached [complete R2 ZIP](https://github.com/NetNeuro/smart-building-bems/releases/download/v2.0.1/JER-D-26-02810_R2_Public_Reproducibility_v2.0.1.zip).
The complete archive is identified by the SHA-256 below and is published on [Zenodo: 10.5281/zenodo.23624875](https://doi.org/10.5281/zenodo.23624875),
version **v2.0.1**. This version refreshes R2 editorial metadata
without changing the scientific results. The earlier automatic GitHub source
snapshot is separately preserved at [10.5281/zenodo.23624304](https://doi.org/10.5281/zenodo.23624304);
that source snapshot alone omits the large numerical archive.

Published archive: `JER-D-26-02810_R2_Public_Reproducibility_v2.0.1.zip`

Size: **253,517,248 bytes**

SHA-256: `18a368a45fd7d94814dc320d0c1bc6394b09a3ffd5cede9db214eae121a19149`

It contains frozen inputs, pinned BDG2 data, code, learning curves, trained
checkpoints, predictions, detector/replay traces, uncertainty outputs and
file-level checksums. Large data and states are supplied in this release asset;
GitHub's automatically generated source archive alone is not the full data package.

From the extracted package, using Python 3.12.14:

```bash
python -m pip install -r requirements_r2.txt
python reproduce.py --verify
python reproduce.py --rerun --output-root rerun
```

Measured forecasting uses the same twelve selected buildings, with 2016-only
checkpoint selection and later-year assessment. PSNN mean RMSE/MAE were
62.404/8.569 kWh versus persistence at 62.839/9.695 kWh. Synthetic detection and
prospective supervisory replay are distinct evaluations. Component READMEs
provide the full protocols and interpretation. Journal correspondence and
document-only assembly material are outside this public numerical distribution.

## Citation, license and earlier release

CITATION.cff identifies the complete R2 archival deposit DOI. For numerical
reproduction, download its ZIP from Zenodo or the identical GitHub release asset,
identified by the SHA-256 above.
The verified original version remains at
[GitHub v1.0.0](https://github.com/NetNeuro/smart-building-bems/tree/v1.0.0),
commit `54552ef6693a25ced7150064e205e254d8ded715`, and
[Zenodo 10.5281/zenodo.22253768](https://doi.org/10.5281/zenodo.22253768).
[Concept DOI 10.5281/zenodo.22253767](https://doi.org/10.5281/zenodo.22253767)
identifies the version series. The original directories remain historical v1
material; the R2 release asset contains the current complete numerical package.

Software, author data and upstream BDG2 material retain their distinct licenses;
see THIRD_PARTY_LICENSES.md and the verbatim notices in the complete archive.
