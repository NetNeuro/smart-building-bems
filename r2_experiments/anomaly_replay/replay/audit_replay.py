"""Decision accounting for prospective replay traces.

The replay engine must emit normalized target-interval rows. Audit joins labels only
for post-hoc evaluation; no target or future fault label enters the replay policy.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

REQUIRED=['building_id','timestamp','gate_flag','otherwise_eligible','curtailment_kWh']

def decision_audit(trace, raw, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    t=trace.copy(); t['timestamp']=pd.to_datetime(t['timestamp'])
    missing=set(REQUIRED)-set(t.columns)
    if missing:raise ValueError(f'Missing normalized trace columns: {sorted(missing)}')
    if t.duplicated(['building_id','timestamp']).any():raise ValueError('Duplicate target building/time')
    for col in ['gate_flag','otherwise_eligible']:
        if not t[col].isin([0,1,False,True]).all():raise ValueError(f'Nonbinary {col}')
        t[col]=t[col].astype(bool)
    t['blocked_eligible']=t.gate_flag&t.otherwise_eligible
    t['intervention']=t.curtailment_kWh>1e-12
    if (t.intervention&t.gate_flag).any():raise ValueError('A gated interval has positive curtailment')
    if (t.intervention&~t.otherwise_eligible).any():raise ValueError('Ineligible interval has positive curtailment')
    r=raw.copy(); r['timestamp']=pd.to_datetime(r['timestamp'])
    if r.duplicated(['building_id','timestamp']).any():raise ValueError('Duplicate raw building/time')
    r=r.set_index(['building_id','timestamp'])
    t['prior_interval']=t.timestamp-pd.Timedelta(hours=1)
    for label in ['anomaly','atype']:
        if label not in r:continue
        for prefix,ts in [('prior',t.prior_interval),('target',t.timestamp)]:
            keys=pd.MultiIndex.from_arrays([t.building_id,ts],names=['building_id','timestamp'])
            t[f'{prefix}_{label}']=r[label].reindex(keys).to_numpy()
    t.to_csv(out/'decision_audit_rows.csv',index=False)
    conditional=[]
    for g in [False,True]:
        for e in [False,True]:
            q=t[(t.gate_flag==g)&(t.otherwise_eligible==e)]
            conditional.append({'prior_gate_flag':int(g),'otherwise_eligible':int(e),
                'building_hours':len(q),'interventions':int(q.intervention.sum()),
                'gross_curtailed_kWh':float(q.curtailment_kWh.sum())})
    pd.DataFrame(conditional).to_csv(out/'decision_truth_table.csv',index=False)
    t[t.otherwise_eligible].to_csv(out/'otherwise_eligible_events.csv',index=False)
    t[t.blocked_eligible].to_csv(out/'blocked_event_audit.csv',index=False)
    if 'prior_anomaly' in t:
        counts=t.groupby(['gate_flag','otherwise_eligible','prior_anomaly'],dropna=False).size().rename('building_hours').reset_index()
        counts.to_csv(out/'prior_fault_alignment.csv',index=False)
    if 'target_anomaly' in t:
        counts=t.groupby(['gate_flag','otherwise_eligible','target_anomaly'],dropna=False).size().rename('building_hours').reset_index()
        counts.to_csv(out/'target_fault_alignment_posthoc.csv',index=False)
    summary={'building_hours':len(t),'prior_flags':int(t.gate_flag.sum()),
        'otherwise_eligible_commands':int(t.otherwise_eligible.sum()),
        'blocked_eligible_commands':int(t.blocked_eligible.sum()),
        'interventions':int(t.intervention.sum()),
        'interpretation':'Fault-label alignment is post-hoc synthetic alarm context. No physical harmful-action or protective-benefit outcome is established.'}
    (out/'decision_audit_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True,type=Path);p.add_argument('--raw',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    result=decision_audit(pd.read_csv(a.trace),pd.read_csv(a.raw),a.out)
    result['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.trace,a.raw]}
    (a.out/'decision_audit_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
