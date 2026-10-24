"""R2 fixed-forecaster pipeline and chronological calibration experiments.
All submitted R1 source remains read-only. Threshold/configuration choices use
only development validation labels; the complete selection manifest is written
before any held-out metric is computed. Existing 60/40 unpurged stays primary.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time,platform,datetime
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score,precision_score,recall_score,accuracy_score,
 average_precision_score,roc_auc_score,confusion_matrix,mean_squared_error,
 mean_absolute_error,r2_score,precision_recall_curve)
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'calibration_audit'))
from calibration_masks import make_calibration_split
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,required=True);P.add_argument('--out',type=Path,default=HERE/'results');args=P.parse_args()
SRC=args.source.resolve();OUT=args.out.resolve();OUT.mkdir(exist_ok=True,parents=True)
FORE=SRC/'revision/canonical/results';SPLITS=['validation','seen_test','unseen_full','unseen_temporal']
SEEDS=[11,23,37]
NAMES12=['abs_rel_residual','signed_rel_residual','jump_from_meter_anchor','dev_lag2','dev_lag24','abs_residual_sq','jump_sq','mean_firing_probability','occupancy','temperature','humidity','solar']
NAMES11=[v for v in NAMES12 if v!='mean_firing_probability']
PIPELINES={'PSNN_12':('PSNN',12),'PSNN_11':('PSNN',11),'Persistence_11':('Persistence',11),'GRU_11':('GRU',11)}
files=[SRC/'data/synthetic_benchmark.csv',SRC/'revision/canonical/core.py',SRC/'revision/anomaly/run_anomaly_baselines.py',SRC/'revision/control/run_prospective_control.py']
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_directory':str(SRC),
 'primary_pipeline_calibration':'within-building nominal60/40 causal unpurged; unchanged GB150/depth3/lr.05, seed11; original230-point[.03,.60]validationF1 threshold grid',
 'pipelines':PIPELINES,'forecasters':'Frozen R1 three-seed PSNN and standalone GRU snapshots; observed-meter persistence on identical target keys. No forecast retraining or test-based model selection.',
 'calibration_grid':{'fractions':[.5,.6,.7],'conditions':['causal_unpurged','strict_history_disjoint'],'strict_support':'Common conservative meter support[t-48,t], derived from audited selected lag24 over X[t-24:t]; balanced minimal purge required by support bounds. All4 pipelines use identical fit/threshold masks.'},
 'original_comparison_preserved':True,'primary_cut_never_selected_on_test':True,
 'LR_fairness_sensitivity':{'features':'PSNN12, same labelled fit rows as GB','LR_grid':[{'C':C,'class_weight':w} for C in [.01,.1,1.,10.] for w in [None,'balanced']],
  'GB_grid':[{'max_depth':d,'learning_rate':lr} for d in [1,2,3,4] for lr in [.025,.05]],
  'selection':'8 candidates per method; maximize AP on development threshold/selection block, deterministic grid-order tie; then maximum F1 over distinct scores on that same development block, lower threshold tie. This block jointly selects candidate and threshold; no independent probability calibration or IID claim.'},
 'no_test_selection':'No cut, purge, candidate or threshold chosen using held-out labels. Sensitivities reported in full, not used to replace the primary cut.'}
(OUT/'protocol_frozen.json').write_text(json.dumps(protocol,indent=2))
raw=pd.read_csv(SRC/'data/synthetic_benchmark.csv',parse_dates=['timestamp']).sort_values(['building_id','timestamp']).reset_index(drop=True)
raw['meter_anchor']=raw.groupby('building_id').meter.shift(1);KEY=['building_id','timestamp'];key=raw.set_index(KEY)
frames={};forecasts={p:{} for p in PIPELINES};Xs={p:{} for p in PIPELINES};predout={p:{} for p in PIPELINES}
for split in SPLITS:
 basepath=FORE/f'canonical_predictions_{split}.csv';files.append(basepath)
 base=pd.read_csv(basepath,parse_dates=['timestamp']);index=base.set_index(KEY).index
 frame=key.loc[index].reset_index();assert np.array_equal(frame.anomaly.to_numpy(),base.anomaly.to_numpy())
 frame['raw_support_start']=frame.timestamp-pd.Timedelta(hours=48);frame['raw_support_end']=frame.timestamp
 frames[split]=frame
 gru=[]
 for seed in SEEDS:
  fn=FORE/f'predictions_GRU_{seed}_{split}.csv';files.append(fn)
  a=pd.read_csv(fn,parse_dates=['timestamp']).set_index(KEY).loc[index].reset_index()
  assert np.allclose(a.actual_clean,frame.clean,rtol=0,atol=1e-10)
  gru.append(a.forecast.to_numpy())
 pp={'PSNN':base.forecast.to_numpy(),'Persistence':frame.meter_anchor.to_numpy(),'GRU':np.mean(gru,axis=0)}
 for name,(which,nf) in PIPELINES.items():
  fc=pp[which];scale=np.maximum(np.abs(fc),10.);signed=(frame.meter.to_numpy()-fc)/scale;absr=np.abs(signed)
  jump=(frame.meter.to_numpy()-frame.meter_anchor.to_numpy())/np.maximum(np.abs(frame.meter_anchor.to_numpy()),10.)
  dev2=(frame.meter.to_numpy()-frame.lag2.to_numpy())/np.maximum(np.abs(frame.lag2.to_numpy()),10.)
  dev24=(frame.meter.to_numpy()-frame.lag24.to_numpy())/np.maximum(np.abs(frame.lag24.to_numpy()),10.)
  z=np.c_[absr,signed,jump,dev2,dev24,absr**2,jump**2,frame.occupancy.to_numpy(),frame.temp.to_numpy(),frame.humidity.to_numpy(),frame.solar.to_numpy()]
  if nf==12:z=np.insert(z,7,base.spike_rate.to_numpy(),axis=1)
  Xs[name][split]=z;forecasts[name][split]=fc
  dest=OUT/'inputs'/name;dest.mkdir(exist_ok=True,parents=True)
  f=frame[KEY+['anomaly','atype']].copy();f['forecast']=fc;f.to_csv(dest/f'forecasts_{split}.csv',index=False)
  predout[name][split]=dest/f'forecasts_{split}.csv'
  np.savez_compressed(dest/f'features_{split}.npz',X=z,feature_names=np.array(NAMES12 if nf==12 else NAMES11),building_id=frame.building_id.to_numpy(),timestamp=frame.timestamp.astype(str).to_numpy(),y=frame.anomaly.to_numpy())

def best_grid(y,s):
 if np.unique(y).size<2:return None
 best=(-1.,.5)
 for t in np.linspace(.03,.60,230):
  v=f1_score(y,s>=t,zero_division=0)
  if v>best[0]:best=(float(v),float(t))
 return {'validation_F1':best[0],'threshold':best[1]}
def exact_threshold(y,s):
 if np.unique(y).size<2:return None
 order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];end=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
 tp=np.cumsum(yy)[end];fp=end+1-tp;fn=y.sum()-tp;fv=2*tp/np.maximum(2*tp+fp+fn,1)
 q=pd.DataFrame({'threshold':ss[end],'F1':fv}).sort_values(['F1','threshold'],ascending=[False,True]).iloc[0]
 return {'threshold':float(q.threshold),'validation_F1':float(q.F1)}

def fitted_gb(X,y,**kwargs):return GradientBoostingClassifier(n_estimators=150,max_depth=kwargs.get('max_depth',3),learning_rate=kwargs.get('learning_rate',.05),random_state=11).fit(X,y)
# STAGE A: select and serialize everything using validation labels only.
fits={};selection=[];candidate_rows=[];status=[]
yv=frames['validation'].anomaly.to_numpy()
for fraction in [.5,.6,.7]:
 for purge in [False,True]:
  cond=f"cut{round(fraction*100)}_"+('strict_history_disjoint' if purge else 'causal_unpurged')
  masks=make_calibration_split(frames['validation'],fraction,purge=purge)
  a,b=masks.fit_mask,masks.threshold_mask
  sub=OUT/'calibration'/cond;sub.mkdir(exist_ok=True,parents=True)
  masks.assignments.to_csv(sub/'assignments.csv',index=False);masks.boundaries.to_csv(sub/'boundaries.csv',index=False);masks.fault_coverage.to_csv(sub/'fault_coverage.csv',index=False)
  for pipe in PIPELINES:
   spec={'condition':cond,'pipeline':pipe,'fit_hours':int(a.sum()),'threshold_hours':int(b.sum()),'purged_hours':int(masks.purged_mask.sum()),'fit_fault_hours':int(yv[a].sum()),'threshold_fault_hours':int(yv[b].sum())}
   if np.unique(yv[a]).size<2 or np.unique(yv[b]).size<2:
    status.append({**spec,'status':'not_estimable_single_class_phase'});continue
   start=time.perf_counter();model=fitted_gb(Xs[pipe]['validation'][a],yv[a]);elapsed=time.perf_counter()-start
   score=model.predict_proba(Xs[pipe]['validation'][b])[:,1];chosen=best_grid(yv[b],score)
   fn=sub/f'{pipe}_GB.joblib';joblib.dump(model,fn,compress=3)
   row={**spec,'method':'frozen_GB','threshold_rule':'original_grid','model_file':str(fn.relative_to(OUT)),**chosen,'validation_AP':float(average_precision_score(yv[b],score)),'fit_wall_seconds':elapsed,'calibrator_file_bytes':fn.stat().st_size}
   fits[(cond,pipe,'frozen_GB')]=(model,None,row);selection.append(row);status.append({**spec,'status':'estimable'})
  # Fair supervised-comparator sensitivity: same12 features and labelled rows.
  pipe='PSNN_12';X=Xs[pipe]['validation'];scale=StandardScaler().fit(X[a])
  for method,grid in [('GB_tuned_development',protocol['LR_fairness_sensitivity']['GB_grid']),('LR_tuned_development',protocol['LR_fairness_sensitivity']['LR_grid'])]:
   trials=[]
   if np.unique(yv[a]).size<2 or np.unique(yv[b]).size<2:continue
   for trial,pars in enumerate(grid):
    start=time.perf_counter()
    if method.startswith('GB'):model=fitted_gb(X[a],yv[a],**pars);s=model.predict_proba(X[b])[:,1];scaler=None
    else:
     model=LogisticRegression(**pars,max_iter=2000,random_state=11).fit(scale.transform(X[a]),yv[a]);s=model.predict_proba(scale.transform(X[b]))[:,1];scaler=scale
    ap=float(average_precision_score(yv[b],s));th=exact_threshold(yv[b],s)
    tr={'condition':cond,'pipeline':pipe,'method':method,'candidate':trial,'parameters':json.dumps(pars,sort_keys=True),'validation_AP':ap,**th,'fit_wall_seconds':time.perf_counter()-start}
    candidate_rows.append(tr);trials.append((ap,-trial,model,scaler,tr))
   _,_,model,scaler,tr=max(trials,key=lambda z:(z[0],z[1]));fn=sub/f'{method}.joblib';joblib.dump({'model':model,'scaler':scaler},fn,compress=3)
   row={**tr,'threshold_rule':'exact_distinct','model_file':str(fn.relative_to(OUT)),'fit_hours':int(a.sum()),'threshold_hours':int(b.sum()),'fit_fault_hours':int(yv[a].sum()),'threshold_fault_hours':int(yv[b].sum()),'purged_hours':int(masks.purged_mask.sum()),'calibrator_file_bytes':fn.stat().st_size}
   fits[(cond,pipe,method)]=(model,scaler,row);selection.append(row)
  # Audit original LR configuration under original exact-score threshold policy.
  if np.unique(yv[a]).size==2 and np.unique(yv[b]).size==2:
   model=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=11).fit(scale.transform(X[a]),yv[a]);s=model.predict_proba(scale.transform(X[b]))[:,1];chosen=exact_threshold(yv[b],s)
   fn=sub/'LR_original_configuration.joblib';joblib.dump({'model':model,'scaler':scale},fn,compress=3)
   row={'condition':cond,'pipeline':pipe,'method':'LR_original_configuration','threshold_rule':'exact_distinct',**chosen,'validation_AP':float(average_precision_score(yv[b],s)),'model_file':str(fn.relative_to(OUT)),'fit_hours':int(a.sum()),'threshold_hours':int(b.sum()),'fit_fault_hours':int(yv[a].sum()),'threshold_fault_hours':int(yv[b].sum()),'purged_hours':int(masks.purged_mask.sum()),'calibrator_file_bytes':fn.stat().st_size}
   fits[(cond,pipe,'LR_original_configuration')]=(model,scale,row);selection.append(row)

pd.DataFrame(selection).to_csv(OUT/'selection_manifest.csv',index=False);pd.DataFrame(candidate_rows).to_csv(OUT/'development_candidate_trials.csv',index=False);pd.DataFrame(status).to_csv(OUT/'calibration_estimability.csv',index=False)
freeze={'stage':'All detector fits and thresholds frozen before held-out metric calculation','selection_manifest_sha256':hashlib.sha256((OUT/'selection_manifest.csv').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'protocol_frozen.json').read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(SRC)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()}}
(OUT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2))
# STAGE B: held-out evaluation. The objects above can no longer be selected here.
metrics=[];faults=[];replayjobs=[]
for (cond,pipe,method),(model,scaler,spec) in fits.items():
 for split in ['seen_test','unseen_full','unseen_temporal']:
  X=Xs[pipe][split];X=scaler.transform(X) if scaler is not None else X;s=model.predict_proba(X)[:,1];f=frames[split];y=f.anomaly.to_numpy();z=s>=spec['threshold'];tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
  rec={'condition':cond,'pipeline':pipe,'method':method,'split':split,'threshold':spec['threshold'],'n':len(f),'Precision':precision_score(y,z,zero_division=0),'Recall':recall_score(y,z,zero_division=0),'F1':f1_score(y,z,zero_division=0),'AP':average_precision_score(y,s),'AUROC':roc_auc_score(y,s),'Normal_FPR':fp/(fp+tn),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Forecast_RMSE_kWh':np.sqrt(mean_squared_error(f.clean,forecasts[pipe][split])),'Forecast_MAE_kWh':mean_absolute_error(f.clean,forecasts[pipe][split]),'Forecast_R2':r2_score(f.clean,forecasts[pipe][split]),'fit_hours':spec['fit_hours'],'threshold_hours':spec['threshold_hours'],'threshold_fault_hours':spec['threshold_fault_hours']};metrics.append(rec)
  for typ in sorted(f.atype.unique()):
   q=f.atype.eq(typ).to_numpy();faults.append({'condition':cond,'pipeline':pipe,'method':method,'split':split,'fault_type':typ,'hours':int(q.sum()),'alarm_hours':int(z[q].sum()),'missed_hours':int(q.sum()-z[q].sum()),'alarm_rate':float(z[q].mean())})
  dest=OUT/'predictions'/cond/pipe/method;dest.mkdir(exist_ok=True,parents=True)
  out=f[KEY+['anomaly','atype']].copy();out['forecast']=forecasts[pipe][split];out['score']=s;out['predicted']=z.astype(int);out.to_csv(dest/f'scores_{split}.csv',index=False)
  pp,rr,tt=precision_recall_curve(y,s);pd.DataFrame({'Precision':pp,'Recall':rr,'threshold':np.r_[tt,np.nan]}).to_csv(dest/f'PR_{split}.csv',index=False)
 if method=='frozen_GB' and cond=='cut60_causal_unpurged':
  dest=OUT/'predictions'/cond/pipe/method
  replayjobs.append({'pipeline':pipe,'forecasts':str(predout[pipe]['unseen_temporal'].relative_to(OUT)),'scores':str((dest/'scores_unseen_full.csv').relative_to(OUT)),'threshold':spec['threshold']})
pd.DataFrame(metrics).to_csv(OUT/'all_detector_metrics.csv',index=False);pd.DataFrame(faults).to_csv(OUT/'all_fault_metrics.csv',index=False)
(OUT/'primary_replay_jobs.json').write_text(json.dumps(replayjobs,indent=2))
m=pd.DataFrame(metrics);primary=m[(m.condition=='cut60_causal_unpurged')&(m.method=='frozen_GB')&(m.split=='unseen_full')].copy();primary.to_csv(OUT/'primary_pipeline_prediction_detection.csv',index=False)
sens=m[(m.method=='frozen_GB')&(m.split=='unseen_full')];sens.to_csv(OUT/'calibration_sensitivity_full_unseen.csv',index=False)
m[(m.pipeline=='PSNN_12')&(m.split=='unseen_full')].to_csv(OUT/'supervised_comparator_sensitivity.csv',index=False)
print(primary[['pipeline','Forecast_RMSE_kWh','Precision','Recall','F1','AP','AUROC','Normal_FPR','threshold']].to_string(index=False))
# Exact submitted R1 primary head reconstruction; numerical tolerances only for CSV storage.
a=pd.read_csv(SRC/'revision/anomaly/main_chronological/predictions_gradient_boosting_unseen_full.csv');b=pd.read_csv(OUT/'predictions/cut60_causal_unpurged/PSNN_12/frozen_GB/scores_unseen_full.csv')
reproduction={'same_key_order':bool(a[KEY].equals(b[KEY])),'max_abs_score_difference':float(np.max(np.abs(a.score-b.score))),'same_binary_decisions':bool(np.array_equal(a.predicted,b.predicted))}
(OUT/'R1_anomaly_reproduction.json').write_text(json.dumps(reproduction,indent=2));assert reproduction['same_key_order'] and reproduction['same_binary_decisions'] and reproduction['max_abs_score_difference']<1e-12,reproduction
print('R1 reproduction:',reproduction)
