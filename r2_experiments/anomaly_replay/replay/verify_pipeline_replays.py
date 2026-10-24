"""Verify retained replay traces, decision accounting and reference comparison."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--reference-aggregate',type=Path);a=p.parse_args();D=a.out
manifest=json.loads((D/'pipeline_replay_manifest.json').read_text());checks=[];keys=None
for cfg in manifest['pipelines']:
 name=cfg['pipeline'];safe=''.join(c if c.isalnum() or c in '_-' else '_' for c in name);f=D/safe
 t=pd.read_csv(f/'prospective_control_trace_primary.csv');agg=pd.read_csv(f/'prospective_control_aggregate.csv').set_index('variant');g=agg.loc['Prospective model + prior gate']
 kk=t[['building_id','timestamp']]
 if keys is None:keys=kk
 else:assert kk.equals(keys),'Pipeline target horizons differ'
 threshold=float(cfg['threshold']);flag=t.gate_available&(t.gate_probability_available>=threshold)
 assert np.array_equal(flag,t.gate_flagged),'Stored flag differs from frozen threshold'
 assert not ((t.curtailed_kWh>0)&flag).any(),'A gated action was applied'
 assert int(flag.sum())==int(g.gate_flagged_hours)
 assert int((t.curtailed_kWh>0).sum())==int(g.intervention_hours)
 assert int((t.otherwise_eligible&flag).sum())==int(g.otherwise_eligible_gate_blocks)
 base=t.clean.sum();controlled=t.controlled_load.sum();basecost=(t.clean*t.price_USD_per_kWh).sum();controlledcost=(t.controlled_load*t.price_USD_per_kWh).sum()
 assert np.allclose([base,controlled,basecost,controlledcost,base-controlled,basecost-controlledcost],g[['baseline_kWh','controlled_kWh','baseline_cost_USD','controlled_cost_USD','net_saved_kWh','cost_saved_USD']].to_numpy(float),rtol=0,atol=2e-10)
 truth=pd.read_csv(f/'decision_audit/decision_truth_table.csv');assert truth.building_hours.sum()==len(t)
 peak=t.groupby('building_id').agg(basepeak=('clean','max'),replaypeak=('controlled_load','max'))
 checks.append({'pipeline':name,'target_rows':len(t),'decision_counts_verified':True,'accounting_verified':True,'maximum_peak_change_kW':float((peak.replaypeak-peak.basepeak).max())})
 if name=='PSNN_12' and a.reference_aggregate:
  old=pd.read_csv(a.reference_aggregate).set_index('variant');cols=old.select_dtypes(include='number').columns
  diff=float(np.max(np.abs(old[cols]-agg[cols]).to_numpy()));assert diff<1e-10
  checks[-1]['canonical_max_abs_difference_vs_R1']=diff
for path,digest in manifest['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,f'Input hash changed: {path}'
(D/'pipeline_verification_report.json').write_text(json.dumps({'checks':checks,'source_hashes_verified':True},indent=2));print(json.dumps(checks,indent=2))
