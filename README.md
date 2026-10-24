# Integrated Load Forecasting and Anomaly-Gated Supervisory Replay for Building Energy Management

R2 scientific reproducibility release **v2.0.0** — analysis version
**R2-2026-10-22**, public distribution prepared 24 October 2026.

Authors: Ahmad H. Milyani; Md Mottahir Alam; Atif Mohammad Kamal; Amjad Ali
Pasha; Tauheed Khan Mohd.

## Get the complete R2 package

The R2 source and metadata are available in this repository. The complete verified
archive is prepared; uploading its GitHub release attachment and depositing the
new Zenodo version are pending. The original v1.0.0 DOI below identifies the
original release only.

Prepared archive: `JER-D-26-02810_R2_Public_Reproducibility_v2.0.0.zip`

Size: **253,518,518 bytes**

SHA-256: `f234843a999c9884a5903a666ce23a5c98489109496d382540f20e1272864888`

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

Use CITATION.cff and the exact new Zenodo version record for R2 when published.
The verified original version remains at
[GitHub v1.0.0](https://github.com/NetNeuro/smart-building-bems/tree/v1.0.0),
commit `54552ef6693a25ced7150064e205e254d8ded715`, and
[Zenodo 10.5281/zenodo.22253768](https://doi.org/10.5281/zenodo.22253768).
[Concept DOI 10.5281/zenodo.22253767](https://doi.org/10.5281/zenodo.22253767)
identifies the version series. The original directories remain historical v1
material; the R2 release asset contains the current complete numerical package.

Software, author data and upstream BDG2 material retain their distinct licenses;
see THIRD_PARTY_LICENSES.md and the verbatim notices in the complete archive.
Recorded preparation disclosures are retained in that archive.
