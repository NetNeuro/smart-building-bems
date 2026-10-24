"""Read-only source/data audit and generation of frozen calibration masks.

Run with the retrieved research directory as the positional argument. No
forecasters or detectors are fitted, and no source-package files are changed.
"""
from pathlib import Path
import argparse,ast,json,hashlib,sys
import numpy as np,pandas as pd,joblib
from calibration_masks import write_calibration_grid

P=argparse.ArgumentParser();P.add_argument('source',type=Path);a=P.parse_args()
SRC=a.source.resolve();OUT=Path(__file__).resolve().parent
core=SRC/'revision/canonical/core.py';head=SRC/'revision/anomaly/run_anomaly_baselines.py'
data=SRC/'data/synthetic_benchmark.csv'
df=pd.read_csv(data,parse_dates=['timestamp']).sort_values(['building_id','timestamp']).reset_index(drop=True)
code=core.read_text();tree=ast.parse(code)
settings={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name) and target.id in ['L','F','SEL']:
    settings[target.id]=ast.literal_eval(node.value)
assert settings['L']==24
assert 'lag24' in settings['SEL']
assert "X.append(a[i-L:i])" in code
assert "last=float(g.meter.iloc[i-1])" in code
lagrows=[]
for lag in [1,2,6,12,24]:
 expected=df.groupby('building_id').meter.shift(lag);mask=expected.notna()
 actual=df[f'lag{lag}'];delta=(expected[mask]-actual[mask]).abs()
 matches=np.isclose(expected[mask],actual[mask],rtol=1e-10,atol=1e-8)
 lagrows.append({'feature':f'lag{lag}','candidate_definition':f'meter.shift({lag}) within building',
                 'assessable_rows':int(mask.sum()),'matching_rows':int(matches.sum()),'max_abs_difference':float(delta.max())})
 assert matches.all(),f'{lag} lag mismatch'
for shift in [0,1]:
 expected=df.groupby('building_id').meter.transform(lambda x:x.shift(shift).rolling(24,min_periods=24).mean())
 mask=expected.notna();delta=(expected[mask]-df.loc[mask,'roll24']).abs()
 matches=np.isclose(expected[mask],df.loc[mask,'roll24'],rtol=1e-10,atol=1e-8)
 lagrows.append({'feature':'roll24','candidate_definition':f'meter.shift({shift}).rolling(24) within building',
                 'assessable_rows':int(mask.sum()),'matching_rows':int(matches.sum()),'max_abs_difference':float(delta.max())})
 if shift==1:assert matches.all(),'Past-only roll24 mismatch'
pd.DataFrame(lagrows).to_csv(OUT/'raw_feature_provenance.csv',index=False)
pred=SRC/'revision/canonical/results/canonical_predictions_validation.csv'
v=pd.read_csv(pred,parse_dates=['timestamp'])
assert not v.duplicated(['building_id','timestamp']).any()
raw=df.set_index(['building_id','timestamp']).loc[v.set_index(['building_id','timestamp']).index].reset_index()
assert np.array_equal(raw.anomaly.to_numpy(),v.anomaly.to_numpy())
assert np.array_equal(raw.atype.to_numpy(),v.anomaly_type.to_numpy())
frame=raw[['building_id','timestamp','anomaly','atype']].copy()
# Selected lag24 at oldest input row t-24 consumes meter at t-48.
# The complete anomaly feature row also consumes current meter at t.
frame['raw_support_start']=frame.timestamp-pd.Timedelta(hours=48)
frame['raw_support_end']=frame.timestamp
frame.to_csv(OUT/'validation_row_provenance.csv',index=False)
write_calibration_grid(frame,OUT)
# Audit source LR choices and the actual retained fitting scaler, without fitting.
archive=SRC/'revision/anomaly/main_chronological'
cfg=json.loads((archive/'experiment_config.json').read_text())
z=np.load(archive/'features_validation.npz',allow_pickle=True)
assert np.array_equal(z['building_id'],frame.building_id.to_numpy())
assert np.array_equal(pd.to_datetime(z['timestamp']),frame.timestamp)
fit=np.zeros(len(frame),bool)
for _,g in frame.groupby('building_id',sort=True):fit[g.index[:int(.6*len(g))]]=True
saved=joblib.load(archive/'fitted_detectors.joblib')
lr=saved['logistic_regression'];sc=saved['supervised_scaler']
mean_expected=z['X'][fit].mean(axis=0);var_expected=z['X'][fit].var(axis=0)
assert np.allclose(sc.mean_,mean_expected,rtol=0,atol=1e-12)
assert np.allclose(sc.var_,var_expected,rtol=0,atol=1e-12)
params=lr.get_params()
threshold=cfg['thresholds']['Logistic regression (supervised)']
metrics=pd.read_csv(archive/'baseline_comparison.csv')
metrics=metrics[(metrics.model=='Logistic regression (supervised)')&(metrics.operating_point=='F1')]
metrics.to_csv(OUT/'archived_logistic_metrics.csv',index=False)
report={
 'source_directory':str(SRC),'source_data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
 'source_files_sha256':{str(x.relative_to(SRC)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [core,head,pred,archive/'experiment_config.json']},
 'sequence_length':settings['L'],'candidate_features':settings['F'],'selected_features':settings['SEL'],
 'neural_forecast_origin':'t-1','neural_input_feature_times':'t-24 through t-1 inclusive',
 'oldest_neural_raw_meter':'lag24 at feature time t-24 = meter[t-48]',
 'complete_anomaly_meter_support':'meter[t-48] through meter[t] inclusive; 49 timestamp positions',
 'direct_head_oldest_meter':'dev_lag24 at target t = meter[t-24]',
 'persistence_common11_intrinsic_meter_support':'meter[t-24] through meter[t] inclusive',
 'comparison_common_support':'Use the conservative neural support [t-48,t] for all four pipelines to keep fitting/threshold labels matched.',
 'strict_purge_rule':'last fitting target c; first threshold target s requires s-48h>c, i.e. s>=c+49h on hourly data. Balanced removal of24 target rows on each side yields48 discarded rows.',
 'overlap_interpretation':'Causal historical reuse is direct sample dependence, not automatically future leakage. Purging removes shared raw meter observations across phases, not all serial/weather/shared-model dependence.',
 'original_LR':{'params':params,'scaler':'StandardScaler fitted on ALL supervised detector-fitting rows, including faults; no threshold/test rows used for scaling.',
                'fitting_rows':int(fit.sum()),'threshold_rows':int((~fit).sum()),
                'fit_fault_rows':int(frame.loc[fit,'anomaly'].sum()),'threshold_fault_rows':int(frame.loc[~fit,'anomaly'].sum()),
                'scaler_n_samples_seen':int(sc.n_samples_seen_),
                'scaler_mean_max_abs_difference_from_fit_only':float(np.max(np.abs(sc.mean_-mean_expected))),
                'scaler_var_max_abs_difference_from_fit_only':float(np.max(np.abs(sc.var_-var_expected))),
                'threshold_selection':threshold,
                'threshold_search_comparability':'LR maximizes F1 over distinct threshold-subset scores. GB searches230 evenly spaced thresholds in[.03,.60]. Same fitting-label availability but not identical threshold search.'},
 'new_LR_fairness_recommendation':'Preserve archived models/threshold results. For new controlled sensitivity, scale LR on the corresponding fitting phase only, fix/freeze class weighting and solver, predeclare a modest C grid, select with development data only, and use the same threshold candidate rule for LR and GB. Never select cuts/settings by held-out metrics.',
 'runtime':{'python':sys.version}}
(OUT/'provenance_and_LR_audit.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
print(pd.read_csv(OUT/'calibration_boundaries.csv').groupby(['nominal_fraction','purged'])[['fit_rows','threshold_rows','purged_rows','target_boundary_gap_hours']].first().to_string())
print('Archived LR:',params)
print('Archived LR threshold:',threshold['threshold'])
