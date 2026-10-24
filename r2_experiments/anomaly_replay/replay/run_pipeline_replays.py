"""Run identical prospective replay for predeclared pipeline forecasts/heads.

Manifest is a JSON list of dictionaries: pipeline, forecasts, scores, threshold.
Paths may be absolute or relative to the working directory. No model selection is
performed here. All pipeline outputs are retained irrespective of their results.
"""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib
import numpy as np,pandas as pd
from audit_replay import decision_audit

HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--raw-data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--detector-metrics',type=Path);a=p.parse_args()
 rows=json.loads(a.manifest.read_text())
 if isinstance(rows,dict):rows=rows['pipelines']
 for cfg in rows:
  for field in ['forecasts','scores']:
   value=Path(cfg[field]);cfg[field]=str(value if value.is_absolute() else (a.manifest.resolve().parent/value).resolve())
 a.out.mkdir(parents=True,exist_ok=True);raw=pd.read_csv(a.raw_data,parse_dates=['timestamp'])
 comparisons=[];decisions=[];blocks=[];truth=[];hashes={str(a.manifest):hashlib.sha256(a.manifest.read_bytes()).hexdigest(),str(a.raw_data):hashlib.sha256(a.raw_data.read_bytes()).hexdigest()};common_keys=None
 for cfg in rows:
  name=cfg['pipeline'];safe=''.join(c if c.isalnum() or c in '_-' else '_' for c in name);dest=a.out/safe
  cmd=[sys.executable,str(HERE/'prospective_replay_engine.py'),'--raw-data',str(a.raw_data),'--forecasts',str(cfg['forecasts']),'--scores',str(cfg['scores']),'--threshold',str(cfg['threshold']),'--out',str(dest)]
  if cfg.get('forecast_model'):cmd+=['--forecast-model',cfg['forecast_model']]
  run=subprocess.run(cmd,text=True,capture_output=True)
  dest.mkdir(parents=True,exist_ok=True);(dest/'execution_stdout.txt').write_text(run.stdout);(dest/'execution_stderr.txt').write_text(run.stderr)
  if run.returncode:raise RuntimeError(f'Replay failed for {name}: {run.stderr[-2000:]}')
  for path in [Path(cfg['forecasts']),Path(cfg['scores'])]:hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  t=pd.read_csv(dest/'prospective_control_trace_primary.csv',parse_dates=['timestamp'])
  keys=t[['building_id','timestamp']].copy()
  if common_keys is None:common_keys=keys
  elif not keys.equals(common_keys):raise ValueError(f'{name}: target keys differ between pipeline comparisons')
  t=t.rename(columns={'gate_flagged':'gate_flag','curtailed_kWh':'curtailment_kWh'})
  t.to_csv(dest/'normalized_trace_primary.csv',index=False)
  summary=decision_audit(t,raw,dest/'decision_audit');summary['pipeline']=name;decisions.append(summary)
  block=pd.read_csv(dest/'decision_audit/blocked_event_audit.csv');block.insert(0,'pipeline',name);blocks.append(block)
  cond=pd.read_csv(dest/'decision_audit/decision_truth_table.csv');cond.insert(0,'pipeline',name);truth.append(cond)
  z=pd.read_csv(dest/'prospective_control_aggregate.csv');z.insert(0,'pipeline',name);z['gate_threshold']=cfg['threshold'];comparisons.append(z)
 allc=pd.concat(comparisons,ignore_index=True)
 allc.to_csv(a.out/'pipeline_replay_all_variants.csv',index=False)
 primary=allc[allc.variant=='Prospective model + prior gate'].copy()
 primary.to_csv(a.out/'pipeline_replay_primary_comparison.csv',index=False)
 if a.detector_metrics:
  detection=pd.read_csv(a.detector_metrics)
  primary.merge(detection,on='pipeline',validate='one_to_one',suffixes=('','_detector')).to_csv(a.out/'complete_pipeline_comparison.csv',index=False)
  hashes[str(a.detector_metrics)]=hashlib.sha256(a.detector_metrics.read_bytes()).hexdigest()
 pd.DataFrame(decisions).to_csv(a.out/'pipeline_decision_counts.csv',index=False)
 nonempty_blocks=[b for b in blocks if len(b)]
 pd.concat(nonempty_blocks or blocks[:1],ignore_index=True).to_csv(a.out/'pipeline_blocked_events.csv',index=False)
 pd.concat(truth,ignore_index=True).to_csv(a.out/'pipeline_decision_truth_tables.csv',index=False)
 # Monetary difference is descriptive accounting, not a safety reward.
 effects=[]
 for name,z in allc.groupby('pipeline',sort=False):
  g=z.set_index('variant').loc['Prospective model + prior gate'];u=z.set_index('variant').loc['Prospective model no gate']
  effects.append({'pipeline':name,'gated_saved_USD':g.cost_saved_USD,'ungated_saved_USD':u.cost_saved_USD,
   'gated_minus_ungated_saved_USD':g.cost_saved_USD-u.cost_saved_USD,'gated_saved_kWh':g.net_saved_kWh,'ungated_saved_kWh':u.net_saved_kWh,
   'gated_minus_ungated_saved_kWh':g.net_saved_kWh-u.net_saved_kWh,'interpretation':'Accounting effect only; no physical harmful-action or protective-benefit outcome was measured.'})
 pd.DataFrame(effects).to_csv(a.out/'gating_accounting_effect.csv',index=False)
 (a.out/'pipeline_replay_manifest.json').write_text(json.dumps({'pipelines':rows,'source_sha256':hashes,'common_target_rows':len(common_keys),'numeric_control_rule':'Unchanged exact R1 prospective replay rule; same tariff, threshold derivation, prior occupancy, building-specific rebound and ceilings.'},indent=2))
 print(allc[allc.variant=='Prospective model + prior gate'].to_string(index=False))
if __name__=='__main__':main()
