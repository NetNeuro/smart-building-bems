import csv,gzip,hashlib,json,math,statistics
from pathlib import Path
R=Path('r2_experiments/measured/results')
def metricfile(p,predcol):
 errs=[];abserrs=[];truth=[];by={};h=hashlib.sha256()
 with gzip.open(p,'rt') as f:
  for row in csv.DictReader(f):
   a=float(row['actual_kWh']);q=float(row[predcol]);e=q-a
   errs.append(e*e);abserrs.append(abs(e));truth.append(a)
   h.update(('|'.join(row[x]for x in ['building_id','site_id','timestamp','actual_kWh'])+'\n').encode())
   b=by.setdefault(row['building_id'],{'sq':[],'abs':[],'actual':[]})
   b['sq'].append(e*e);b['abs'].append(abs(e));b['actual'].append(a)
 n=len(errs);mu=statistics.fmean(truth)
 return {'n':n,'row_truth_sha256':h.hexdigest(),'RMSE_kWh':math.sqrt(math.fsum(errs)/n),'MAE_kWh':math.fsum(abserrs)/n,'R2':1-math.fsum(errs)/math.fsum((x-mu)**2 for x in truth),'CVRMSE_pct':100*math.sqrt(math.fsum(errs)/n)/mu,'buildings':{k:{'n':len(v['sq']),'RMSE_kWh':math.sqrt(math.fsum(v['sq'])/len(v['sq'])),'MAE_kWh':math.fsum(v['abs'])/len(v['abs']),'CVRMSE_pct':100*math.sqrt(math.fsum(v['sq'])/len(v['sq']))/statistics.fmean(v['actual'])}for k,v in by.items()}}
with (R/'test_metrics_seed.csv').open()as f:seedrows=list(csv.DictReader(f))
with (R/'table_main_R2_selected.csv').open()as f:main=list(csv.DictReader(f))
out={'status':'PASS','scope':'Independent stdlib recomputation from every final R2 prediction CSV; no reuse of training or metric helpers.','models':{}}
common=metricfile(R/'common_test_rows.csv.gz','persistence_kWh');keyhash=common['row_truth_sha256'];assert common['n']==98483 and len(common['buildings'])==12
for model in ['DRA_PSNN','DRA_GRU','DRA_LSTM','Transformer']:
 ds=[]
 for seed in [11,23,37]:
  q=metricfile(R/f'predictions/{model}_seed{seed}_R2_fullpool_test.csv.gz','prediction_kWh');ds.append(q)
  assert q['row_truth_sha256']==keyhash
  ref=next(x for x in seedrows if x['panel']=='R2_fullpool_validation_selected'and x['model']==model and int(x['seed'])==seed)
  for m in ['RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']:assert abs(q[m]-float(ref[m]))<1e-9,(model,seed,m,q[m],ref[m])
 ref=next(x for x in main if x['model']==model)
 summary={m+'_mean':statistics.fmean(q[m]for q in ds) for m in ['RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']}
 summary['median_building_CVRMSE_pct']=statistics.fmean(statistics.median(v['CVRMSE_pct']for v in q['buildings'].values())for q in ds)
 for m,v in summary.items():assert abs(v-float(ref[m]))<1e-9,(model,m,v,ref[m])
 summary['RMSE_kWh_SD']=statistics.stdev(q['RMSE_kWh']for q in ds)
 summary['MAE_kWh_SD']=statistics.stdev(q['MAE_kWh']for q in ds)
 summary['building_metrics']={b:{m:statistics.fmean(q['buildings'][b][m]for q in ds)for m in ['RMSE_kWh','MAE_kWh','CVRMSE_pct']}for b in common['buildings']}
 out['models'][model]=summary
out['persistence']=common
for model in ['DRA_PSNN']:
 p=out['models'][model]
 out['new_delta_vs_persistence']={m:p[m+'_mean']-common[m]for m in ['RMSE_kWh','MAE_kWh']}
 out['relative_reduction_pct']={m:100*(common[m]-p[m+'_mean'])/common[m]for m in ['RMSE_kWh','MAE_kWh']}
 out['buildings_better_RMSE_than_persistence']=sum(p['building_metrics'][b]['RMSE_kWh']<common['buildings'][b]['RMSE_kWh']for b in common['buildings'])
 out['buildings_better_MAE_than_persistence']=sum(p['building_metrics'][b]['MAE_kWh']<common['buildings'][b]['MAE_kWh']for b in common['buildings'])
Path('r2_experiments/editorial/independent_measured_numeric_audit.json').write_text(json.dumps(out,indent=2))
print('PASS: all12 R2 predictionfiles, all98483commonrow truthkeys, allpooledseedmetrics and main-tablemeans/medianbuildingdefinition.')
print('PSNN versus persistence',out['new_delta_vs_persistence'],out['relative_reduction_pct'])
print('Buildings with lower RMSE/MAE',out['buildings_better_RMSE_than_persistence'],out['buildings_better_MAE_than_persistence'])
