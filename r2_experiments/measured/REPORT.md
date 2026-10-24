# Measured forecasting R2 evidence

Protocol SHA-256: b81688abd338c44e02340e1c1ac8401fef11bd112730b8ce51804c872b62d8ac
Decision-lock SHA-256: fc1b8c09c4d2968f30822d42bb30082088a4e6a0e5009562c4ceb5576e3d9128

## Exact data/source preservation

The exact R1 architecture-class file, canonical source, external preprocessing
script and numerical configurations were copied and hashed before execution.
The source package was not modified. All four official BDG2 source hashes were
verified. Input scaling and increment normalization are bitwise identical to R1.
All 84,928 eligible training sequences, 17,296 validation sequences and 98,483
test targets have their original row bindings. Training is January–October
2016, validation November–December 2016, and the test grid is calendar 2017.
These are later-year predictions after retraining on the same twelve buildings
at six sites; this is not held-out-building/site or zero-shot transfer.

Selection uses only Jan–Oct 2016: >=95% finite nonnegative meter values, median
and SD both >1 kWh, and positive published floor area. Take the first six
lexicographic sites with at least two eligible buildings; prefer the first
lexicographic education and office meter where present, then lexicographic
fallback. No test-dependent meter selection or cleaning was added. Of 105,120
expected 2017 building-hours, 4,417 lack a target, 57 an anchor and 2,163 a complete
prior window, leaving 98,483 common scored targets. Counts are mutually
exclusive in that order. Raw zeros/extremes remain. Some Crow scored series end
in November; full-calendar describes the evaluation grid, not complete measured
coverage in every building. Detailed coverage and exclusions are supplied.

## Training-budget/checkpoint evidence

All four models and all three original seeds used every eligible training
sequence, batch 256, and the original AdamW settings without a new search or LR
schedule. Every epoch records evaluation-mode normalized MSE on the fixed 6,800
training diagnostic windows and all 17,296 validation windows, plus validation
meter RMSE/MAE, updates, exposures and process CPU/wall timing. The maximum
actually executed stage was 40epochs; there are 332
updates per full-pool epoch, compared with 135 total updates in the legacy run.
Selected epochs (same epoch for the three seeds of each model) were DRA-PSNN: 37, DRA-GRU: 25, DRA-LSTM: 27, Transformer: 38.
The rule minimizes mean validation RMSE across seeds, with earliest exact ties.
The predeclared 10→20→40 escalation rule was applied only to 2016 validation:
extend all runs when any model improves >=0.5% in the last 5 epochs or has its
best mean checkpoint in the last 2 epochs. Optimizer/RNG states were resumed.
The predeclared extension trigger was no longer met at the terminal cap; this is an empirical budget-plateau diagnostic, not a global-optimum certificate.

All budgets/checkpoints and validation-file hashes were locked before additional
2017 scoring. The2017 holdout had already been reported in R1; this follow-up is
therefore explicitly a reused-holdout robustness audit rather than pristine
external confirmation. No further training/selection follows2017 scoring.
Equal-update landmarks 135/540/1350 are validation diagnostics, not equal-compute
or pure sample-size-effect experiments.

## Final selected-checkpoint results

DRA-PSNN: RMSE 62.404042 +/- 0.020255 kWh;
MAE 8.569265 +/- 0.138119 kWh;
R² 0.886153; median-building CVRMSE
8.330483%.
Persistence: RMSE 62.838679kWh, MAE
9.695034kWh, median-building CVRMSE
8.410614%.
All comparators/seeds are in table_main_R2_selected.csv. The legacy fixed-budget
table is retained separately and is not silently overwritten.

## Paired building/time uncertainty

The 12-row-per-comparison building tables report paired DRA-PSNN minus
persistence/GRU/LSTM differences on identical rows; negative favors PSNN.
Across-seed SD measures training variability and is separate from the
exploratory building/time intervals. Pooled point estimates retain the mean of
seed-specific metrics rather than substituting ensemble-prediction error.

For 5,000 replicates, fourteen-calendar-day moving blocks are drawn within each
calendar quarter and synchronized across all buildings/models/seeds. Original
missing masks remain attached to dates; denominators are recomputed. Seven-
and 28-day sensitivities were fixed in advance. Fitted seed models are held
fixed. The site/time sensitivity resamples the six site clusters, keeping both
buildings in each chosen site; site draws and time draws are shared between
legacy and new panels. These are pointwise 95% percentile intervals conditional
on the selected data and fitted models. Six deterministically selected sites do
not support broad building-population/site-transfer inference. There are no
hour-iid significance tests, confirmatory rankings, equivalence or economic
claims.

For the selected model versus persistence, pooled delta RMSE is
-0.434637kWh, fixed-panel 14-day interval
[-1.092625,-0.276329]kWh. The exploratory site/time interval
is [-2.565914,-0.121740]kWh. Complete GRU/LSTM comparisons,
MAE and equal-building normalized-error results, and block sensitivities are in
paired_block_intervals.csv. The legacy delta RMSE is only
-0.003680kWh (-0.005857% of persistence RMSE),
which remains numerically negligible and does not establish operational value.

## Verification and portable outputs

Verification recomputes every new metric from predictions, checks validation
checkpoint choices and immutable hashes, and replays the first 1,024 test rows
of all 12 new and all 12 R1 checkpoints at1e-8 tolerance. Paired pooled estimands and
common-row denominators are independently checked. Evidence:
results/VERIFICATION.txt. FACTS.json provides the main table, all selected epochs,
actual budget, paired intervals, relative artifact paths and actual hardware.
Execution uses AMD EPYC 9V74 80-Core Processor, CPU-only, Python3.12.14,
PyTorch2.10.0+cpu, NumPy2.3.5, pandas2.2.3,
scikit-learn1.8.0, SciPy1.17.0. Training uses four worker
processes and two Torch threads per worker. Times exclude validation/I/O and do
not establish edge deployment or architectural compute equality.

## Figure captions

Training/validation curves: Mean normalized-increment MSE across seeds 11, 23, 37
on the fixed 6,800-window training diagnostic set and full17,296-window2016
validation set. Bands are sample SD across training seeds, not confidence bands
for buildings/time. Every optimization epoch uses all 84,928 eligible training
windows. Vertical lines mark the architecture-specific validation-locked epoch.

ValidationRMSE curves: Meter-scale2016validationRMSE mean and sample SD across
three seeds through the predeclared terminal training stage. No 2017 score enters
the epoch or budget decision.

Paired-building plot: DRA-PSNN minus persistence CVRMSE differences for the same
12 buildings and 2017 common rows. Negative values favorPSNN. Intervals are
exploratory pointwise 95% fixed-panel, synchronized 14-day block intervals; training
seeds are held fixed. These are later-year temporal comparisons after retraining,
not unseen-building/site or physical operational validation.
