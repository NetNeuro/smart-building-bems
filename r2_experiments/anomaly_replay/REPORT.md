# R2 anomaly calibration and workflow evidence

The exact submitted R1 detector and prospective replay were reproduced without changing the frozen source. New analyses do not establish a benefit from the complete PSNN workflow over simpler alternatives. The persistence pipeline has stronger anomaly F1/AP and slightly larger illustrative replay savings while using substantially less inference compute; all four workflows still issue just one primary intervention and leave peaks unchanged.

PASS: 126 saved detector/split reconstructions; R1 head exact scores/decisions; R1 replay exact aggregate/trace; all four common-key, accounting and zero-peak assertions; raw-lag provenance and strict-boundary checks.

## Four matched workflows

| Pipeline | Ensemble forecast RMSE kWh | Detector F1 | Detector AP | Normal FPR percent | Forecast plus detector ms | Neural members | Neural parameters total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRA-PSNN + GB12 | 26.7169 | 0.4712 | 0.3292 | 10.5305 | 5.8220 | 3.0000 | 27726.0000 |
| DRA-PSNN + GB11 | 26.7169 | 0.4811 | 0.3377 | 9.0836 | 5.7393 | 3.0000 | 27726.0000 |
| Persistence + GB11 | 29.1037 | 0.5504 | 0.3892 | 9.0300 | 0.1736 | 0.0000 | 0.0000 |
| Standalone GRU + GB11 | 25.8685 | 0.4921 | 0.3280 | 7.9046 | 1.9437 | 3.0000 | 18483.0000 |

Forecast RMSE is the actual ensemble metric on the same 4,056 held-out rows, not the mean individual-seed RMSE in R1 Table 8. The 12/11 suffix is detector input count; all multivariate anomaly components and their labels/calibration rule are common except the omitted firing-probability feature. The standalone GRU uses its frozen R1 settings.

| Pipeline | Building hours | Actions | Flags | Eligible blocks | Net saved kWh | Saved USD | Energy saved percent | Tariff saved percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRA-PSNN + GB12 | 549 | 1 | 60 | 2 | 2.6860 | 0.8548 | 0.0045 | 0.0101 |
| DRA-PSNN + GB11 | 549 | 1 | 56 | 2 | 2.6860 | 0.8548 | 0.0045 | 0.0101 |
| Persistence + GB11 | 549 | 1 | 57 | 0 | 3.1535 | 1.0036 | 0.0053 | 0.0119 |
| Standalone GRU + GB11 | 549 | 1 | 53 | 2 | 2.5046 | 0.7971 | 0.0042 | 0.0095 |

All primary replay rows use identical 549 target building-hours and the same prospective rule, known tariff, prior occupancy/score and rebound code. No tariff, threshold or rule was selected using held-out savings. Differences of fractions of a dollar across one intervention do not establish economic value. Physical feedback and comfort/harm outcomes remain untested.

## Calibration chronology sensitivity

| Cut | Gap excluded target hours | threshold | F1 | Precision | Normal FPR percent | Stuck recall percent |
| --- | --- | --- | --- | --- | --- | --- |
| 50/50 | 0 | 0.4979 | 0.4963 | 0.4163 | 7.4759 | 34.6939 |
| 50/50 | 48 | 0.3536 | 0.4904 | 0.4198 | 7.0740 | 10.2041 |
| 60/40 | 0 | 0.1345 | 0.4712 | 0.3599 | 10.5305 | 14.2857 |
| 60/40 | 48 | 0.1047 | 0.4570 | 0.3205 | 14.6570 | 77.5510 |
| 70/30 | 0 | 0.5975 | 0.4796 | 0.5649 | 2.7867 | 12.2449 |
| 70/30 | 48 | 0.1345 | 0.4924 | 0.3658 | 11.3344 | 63.2653 |

The nominal 50/50, 60/40, 70/30 cuts were declared in advance; 60/40 unpurged remains the primary submitted configuration. A 48 target-hour purge means 24 rows removed each side and 49 hours between the last fitting and first threshold target. The actual code uses feature rows [t−24,t−1], and their lag24 reaches raw meter[t−48]; scoring also uses meter[t]. Every assessable lag matches within-building meter.shift(k); roll24 matches meter.shift(1).rolling(24). A 24-hour purge would not separate the nested raw history. The persistence-only row intrinsically needs 24 hours of history, but the conservative common 48-hour footprint keeps all four pipelines' labels/masks matched.

Historical sharing is causal dependence, not automatically future leakage. Strict separation only removes shared raw inference observations; trained weights/scalers and serial/weather dependence remain shared. Purging changes fault allocation and sample count as well, so this is a sensitivity study rather than identification of an overlap causal effect. No cut or purge is picked from these held-out scores.

Across the six canonical conditions, F1 ranges 0.457–0.496, but normal FPR ranges 2.79%–14.66% and stuck recall 10.2%–77.6%. Similar F1 therefore does not imply a stable operating tradeoff. Every fitting condition lacks schedule faults and every threshold condition lacks HVAC degradation faults; other rare-class absences remain. Full counts and paired decision agreement are supplied.

## Fair supervised logistic comparison

The original LR was correctly scaled on 981 supervised fitting rows only, C=1, class_weight=balanced, max_iter=2000, but selected its threshold across distinct development scores while GB used the 230-point [.03,.60] grid. Its extreme threshold 0.982049 gave held-out F1 0.044568. This is not evidence that all linear classifiers are inadequate.

A separate equal-eight-candidate comparison used the same 12 features and fitting labels: LR C∈{.01,.1,1,10}×{unweighted,balanced} and GB depth∈{1,2,3,4}×learning rate∈{.025,.05}, with 150 trees fixed. Candidate AP and the final F1 threshold were both selected on the development threshold/selection block; this block is reused for both choices and is not an independent probability calibration sample. No held-out result chooses a candidate. At the primary 60/40 cut, development-selected GB achieves F1 0.516046, AP 0.365298 and FPR 0.068060, whereas LR achieves F1 0.449655, AP 0.247317 and FPR 0.063773. LR selects C=1, unweighted, threshold 0.234959; GB selects depth=2/lr=0.05, threshold 0.321595. The original frozen head remains primary; these are controlled comparator sensitivities.

## Gate decision accounting

At the primary PSNN12 operating point, 549 building-hours comprise 488 unflagged/ineligible, 58 flagged/ineligible, 1 unflagged/eligible and 2 flagged/eligible. Of 60 prior flags, 20 match prior synthetic fault labels and 40 normal labels. The two blocked targets are 26 February 08:00: Outpatient prior bias score 0.770512 and School prior spike score 0.991503. These alignments describe synthetic alarm context; they do not measure physical harmful actions or protective benefit. Ungated savings exceed gated savings by USD 0.352567. The matched gross comparison remains an ex-post accounting diagnostic.

## Cost and reproducibility

The new timings use the same host/thread/input policy for all four workflows and loaded R1 weights: three-member neural ensembles, one GB detector, batch one, 100 warmups and 1000 passes in each of three repeats. Medians ± repeat SD quantify implementation cost; normalized-input preprocessing, network/I/O and common replay-rule execution are excluded. Do not compare these ensemble times directly with old single-member R1 latency, infer edge performance, or infer a firing-feature speed benefit from the noisy PSNN12/11 difference.

Source hashes and protocol are in results/selection_freeze.json; the complete detector selection manifest was written before held-out metric calculation. Per-row forecasts/features/scores, fitted heads, all cut labels, candidate trials, raw latencies, replay traces and verifiers accompany the summaries. No synthetic data, fault labels or source weights were regenerated. Rerun commands are in README.md.
