"""Read-only same-host CPU cost comparison for four frozen forecast pipelines.

No training or test-dependent selection. Class definitions are extracted by AST,
so importing the archived experiment module cannot write into the source package.
Optional --heads-json maps pipeline names to trusted local fitted joblib files:
  {"canonical_psnn12": "/.../head.joblib", ...}
Each joblib may contain a fitted estimator directly, or a dictionary with key
'gradient_boosting', 'classifier', 'estimator', 'model', or 'head'.
"""
from pathlib import Path
import argparse,ast,json,hashlib,time,os,platform,math,io
import numpy as np,pandas as pd,torch,torch.nn as nn,joblib

P=argparse.ArgumentParser();P.add_argument('source',type=Path);P.add_argument('--heads-json',type=Path)
P.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'pipeline_cost')
P.add_argument('--warmups',type=int,default=100);P.add_argument('--passes',type=int,default=1000);P.add_argument('--repeats',type=int,default=3)
a=P.parse_args();SRC=a.source.resolve();OUT=a.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
try:torch.set_num_interop_threads(1)
except RuntimeError:pass
available=sorted(os.sched_getaffinity(0))
if available:os.sched_setaffinity(0,{available[0]})
code=(SRC/'revision/canonical/core.py').read_text();tree=ast.parse(code)
names={'RAB','HDRAB','DRA','PSNN','Proposed','GRU'}
namespace={'torch':torch,'nn':nn}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[]),'<archived_class_definitions>','exec'),namespace)
ckdir=SRC/'revision/canonical/results/checkpoints';models={};states={};ckfiles={}
for family,tag in [('PSNN','DOptDRPSNN'),('GRU','GRU')]:
 models[family]=[];states[family]=[];ckfiles[family]=[]
 for seed in [11,23,37]:
  file=ckdir/f'{tag}_{seed}_1h.pt';saved=torch.load(file,map_location='cpu',weights_only=True);h=saved['hyperparameters']
  if family=='PSNN':m=namespace['Proposed'](len(saved['features']),h['channels'],h['dropout'],h['beta'],h['theta'],h['tau'],h['kappa'])
  else:m=namespace['GRU'](len(saved['features']),h['h'])
  m.load_state_dict(saved['state_dict']);m.eval();models[family].append(m);states[family].append(saved);ckfiles[family].append(file)
# One shared actual normalized input and sensor context, fixed before benchmarking.
df=pd.read_csv(SRC/'data/synthetic_benchmark.csv',parse_dates=['timestamp'])
pred=pd.read_csv(SRC/'revision/canonical/results/canonical_predictions_unseen_full.csv',parse_dates=['timestamp'])
p0=pred.iloc[0];g=df[df.building_id==p0.building_id].sort_values('timestamp').reset_index(drop=True)
ti=int(np.flatnonzero(g.timestamp.to_numpy()==np.datetime64(p0.timestamp))[0]);assert ti>=24
for frame in [g]:
 frame['log_area']=np.log1p(frame.square_feet);frame['hour_sin']=np.sin(2*np.pi*frame.hour/24);frame['hour_cos']=np.cos(2*np.pi*frame.hour/24)
 frame['dow_sin']=np.sin(2*np.pi*frame.dow/7);frame['dow_cos']=np.cos(2*np.pi*frame.dow/7)
F=['temp','humidity','solar','occupancy','log_area','hour_sin','hour_cos','dow_sin','dow_cos','lag1','lag2','lag6','lag12','lag24','roll24']
s=states['PSNN'][0];means=np.array(s['scaler_mean']);scales=np.array(s['scaler_scale']);ids=[F.index(f) for f in s['features']]
values=((g[F].to_numpy()-means)/scales).astype('float32');x=torch.tensor(values[ti-24:ti,ids][None])
r=g.iloc[ti];anchor=float(g.iloc[ti-1].meter);sqrtarea=math.sqrt(float(r.square_feet));MU=float(s['MU']);SD=float(s['SD'])
raw={k:float(r[k]) for k in ['meter','lag2','lag24','occupancy','temp','humidity','solar']}

@torch.no_grad()
def forecast_psnn(with_rate):
 ps=[];ss=[]
 for m in models['PSNN']:
  if with_rate:p,rate=m(x,return_rate=True);ss.append(float(rate[0]));ps.append(float(p[0]))
  else:ps.append(float(m(x)[0]))
 return anchor+(sum(ps)/len(ps)*SD+MU)*sqrtarea, sum(ss)/len(ss) if ss else None

@torch.no_grad()
def forecast_gru():
 ps=[float(m(x)[0]) for m in models['GRU']]
 return anchor+(sum(ps)/len(ps)*SD+MU)*sqrtarea,None

def forecast_persistence():return anchor,None

def features(forecast,rate,include_rate):
 signed=(raw['meter']-forecast)/max(abs(forecast),10.);ab=abs(signed)
 jump=(raw['meter']-anchor)/max(abs(anchor),10.)
 dev2=(raw['meter']-raw['lag2'])/max(abs(raw['lag2']),10.)
 dev24=(raw['meter']-raw['lag24'])/max(abs(raw['lag24']),10.)
 vals=[ab,signed,jump,dev2,dev24,ab*ab,jump*jump]
 if include_rate:vals.append(rate)
 vals.extend([raw['occupancy'],raw['temp'],raw['humidity'],raw['solar']])
 return np.array(vals,dtype=float)[None,:]

def get_head(path):
 q=joblib.load(path)
 if hasattr(q,'predict_proba'):return q
 if isinstance(q,dict):
  for k in ['gradient_boosting','classifier','estimator','model','head','gb']:
   if k in q and hasattr(q[k],'predict_proba'):return q[k]
 raise TypeError(f'No fitted predict_proba estimator in{path}')

headfiles=json.loads(a.heads_json.read_text()) if a.heads_json else {}
headfiles={k:str((a.heads_json.parent/Path(v)).resolve()) if not Path(v).is_absolute() else v for k,v in headfiles.items()}
configs=[('canonical_psnn12','PSNN',True,lambda:forecast_psnn(True)),
         ('psnn_common11','PSNN',False,lambda:forecast_psnn(False)),
         ('persistence11','Persistence',False,forecast_persistence),
         ('standalone_gru11','GRU',False,forecast_gru)]
latencies=[];summaries=[];verification=[];storage_rows=[]
for name,family,rate,forecaster in configs:
 head=get_head(headfiles[name]) if name in headfiles else None
 p,sr=forecaster();row=features(p,sr,rate)
 if family=='PSNN':
  verification.append({'pipeline':name,'forecast':p,'archived_forecast':float(p0.forecast),'absolute_difference_kWh':abs(p-float(p0.forecast)),
                       'firing_probability':sr,'archived_firing_probability':float(p0.spike_rate)})
 elif family=='GRU':
  originals=[]
  for seed in [11,23,37]:
   z=pd.read_csv(SRC/f'revision/canonical/results/predictions_GRU_{seed}_unseen_full.csv');originals.append(float(z.iloc[0].forecast))
  verification.append({'pipeline':name,'forecast':p,'archived_forecast':sum(originals)/3,'absolute_difference_kWh':abs(p-sum(originals)/3)})
 else:verification.append({'pipeline':name,'forecast':p,'archived_forecast':anchor,'absolute_difference_kWh':0.})
 # Forecaster-only and complete feature/classifier paths are reported separately.
 modes={'forecaster_only':forecaster}
 if head is not None:
  def complete(f=forecaster,h=head,with_rate=rate):
   q,sp=f();v=features(q,sp,with_rate);return float(h.predict_proba(v)[0,1])
  modes['forecast_features_classifier']=complete
 for mode,operation in modes.items():
  for repeat in range(a.repeats):
   for _ in range(a.warmups):operation()
   ns=[]
   for i in range(a.passes):
    start=time.perf_counter_ns();operation();ns.append(time.perf_counter_ns()-start)
   for i,value in enumerate(ns):latencies.append({'pipeline':name,'mode':mode,'repeat':repeat+1,'pass':i+1,'latency_ns':value})
   summaries.append({'pipeline':name,'mode':mode,'repeat':repeat+1,'median_ms':float(np.median(ns)/1e6),'p95_ms':float(np.quantile(ns,.95)/1e6),
                      'min_ms':float(np.min(ns)/1e6),'host_load_1min':os.getloadavg()[0]})
 storage={}
 if family!='Persistence':
  params=sum(p.numel() for m in models[family] for p in m.parameters());parameter_bytes=sum(p.numel()*p.element_size() for m in models[family] for p in m.parameters())
  buf=io.BytesIO();torch.save([m.state_dict() for m in models[family]],buf)
  (OUT/f'forecaster_{name}_ensemble_weights.pt').write_bytes(buf.getvalue())
  storage={'forecaster_members':3,'neural_parameters_total':params,'neural_parameter_bytes':parameter_bytes,'source_checkpoint_bytes':sum(p.stat().st_size for p in ckfiles[family]),'weights_only_ensemble_serialized_bytes':len(buf.getvalue())}
 else:storage={'forecaster_members':0,'neural_parameters_total':0,'neural_parameter_bytes':0,'source_checkpoint_bytes':0,'weights_only_ensemble_serialized_bytes':0}
 storage.update({'pipeline':name,'anomaly_features':12 if rate else 11,'detector_available':head is not None})
 if head is not None:
  trees=head.estimators_.ravel();storage['detector_tree_nodes_total']=sum(t.tree_.node_count for t in trees)
  storage['detector_tree_array_bytes']=sum(t.tree_.__getstate__()['nodes'].nbytes+t.tree_.__getstate__()['values'].nbytes for t in trees)
  storage['detector_serialized_file_bytes']=Path(headfiles[name]).stat().st_size
 storage_rows.append(storage)
 (OUT/f'storage_{name}.json').write_text(json.dumps(storage,indent=2))
 print(name,'complete' if head is not None else'forecaster only',flush=True)
pd.DataFrame(storage_rows).to_csv(OUT/'pipeline_storage_summary.csv',index=False)
pd.DataFrame(latencies).to_csv(OUT/'latency_passes.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'latency_repeats.csv',index=False)
summary=pd.DataFrame(summaries).groupby(['pipeline','mode']).agg(repeat_mean_median_ms=('median_ms','mean'),repeat_SD_median_ms=('median_ms','std'),repeat_min_median_ms=('median_ms','min'),repeat_max_median_ms=('median_ms','max'),mean_p95_ms=('p95_ms','mean')).reset_index()
summary.to_csv(OUT/'pipeline_latency_summary.csv',index=False)
pd.DataFrame(verification).to_csv(OUT/'inference_snapshot_check.csv',index=False)
assert max(v['absolute_difference_kWh'] for v in verification)<.005,'Loaded inference materially differs from archived forecast'
cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
meta={'cpu_model':cpu_model,'source':str(SRC),'python':platform.python_version(),'torch':torch.__version__,'processor':platform.processor(),'threads':torch.get_num_threads(),'CPU_affinity':sorted(os.sched_getaffinity(0)),
      'available_affinity_before_pin':available,'warmups_per_repeat':a.warmups,'passes_per_repeat':a.passes,'repeats':a.repeats,'batch_size':1,'input_window_shape':list(x.shape),'input_tensor_bytes':x.numel()*x.element_size(),
      'context_building_id':int(p0.building_id),'context_target_timestamp':str(p0.timestamp),
      'timing_scope':'Forecaster-only includes three sequential ensemble members and observed-meter reconstruction. Complete path additionally builds anomaly features and scores fitted gradient boosting. Fixed input is already normalized; telemetry acquisition, preprocessing, transport, detector fitting and supervisory action execution are excluded.',
      'interpretation':'Same-host eager CPU software timing, not an IoT gateway/hardware measurement. Three timing repeats measure local runtime variability, not three independent machines. Persistence timing includes Python/timer overhead at small microsecond scales.',
      'storage_scope':'Neural parameter counts/bytes are summed over three forecaster members. Weights-only serialization excludes training metadata. Detector node-array bytes exclude Python object overhead and are separate from serialized file size.',
      'source_sha256':{str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for paths in ckfiles.values() for p in paths},
      'head_files':headfiles,'head_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in headfiles.items()},
      'class_source_sha256':hashlib.sha256((SRC/'revision/canonical/core.py').read_bytes()).hexdigest(),
      'raw_dataset_sha256':hashlib.sha256((SRC/'data/synthetic_benchmark.csv').read_bytes()).hexdigest(),
      'normalized_input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
(OUT/'benchmark_config.json').write_text(json.dumps(meta,indent=2))
print(summary.to_string(index=False))
