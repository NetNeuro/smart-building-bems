from pathlib import Path
import hashlib, re, sys
import numpy as np, pandas as pd
from scipy.stats import f, levene

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; T=ROOT/'tables'; DATA=ROOT/'data'/'synthetic_benchmark.csv'
fail=[]
def close(a,b,tol=0.015,label=''):
    if not (np.isfinite(a) and np.isfinite(b) and abs(a-b)<=tol): fail.append(f'{label}: {a} != {b}')
def num(s):
    s=str(s).replace('$','').replace(',','').replace('%','').strip()
    return float(re.split(r'\s*\+/-\s*',s)[0])
def pm(s):
    p=re.split(r'\s*\+/-\s*',str(s)); return float(p[0].replace(',','')), float(p[1].replace(',','')) if len(p)>1 else 0.0

def table(n): return pd.read_csv(T/f'table_{n:02d}.csv')

# Hash identity
sha=hashlib.sha256(DATA.read_bytes()).hexdigest(); stored=(R/'data_sha256.txt').read_text().split()[0]
if sha!=stored: fail.append('synthetic_benchmark SHA-256 mismatch')

# Table 8: 1h per-seed raw means/SD + persistence
raw=pd.read_csv(R/'forecasting_models.csv'); q=raw[raw.split=='unseen_full'].groupby('model')[['RMSE','MAE','MAPE','R2']].agg(['mean','std']); tab=table(8).set_index('Model')
for m in ['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer']:
    for col,tc in [('RMSE','RMSE kWh'),('MAE','MAE kWh'),('MAPE','MAPE %'),('R2','R2')]:
        mean,sd=pm(tab.loc[m,tc]); close(q.loc[m,(col,'mean')],mean,0.015,f'T8 {m} {col} mean'); close(q.loc[m,(col,'std')],sd,0.015,f'T8 {m} {col} SD')
cl=pd.read_csv(R/'classical_unseen_full.csv').set_index('model');
for col,tc in [('RMSE','RMSE kWh'),('MAE','MAE kWh'),('MAPE','MAPE %'),('R2','R2')]: close(cl.loc['Persistence',col],num(tab.loc['Persistence',tc]),0.015,f'T8 Persistence {col}')

# Table 9: 6h all baselines
raw=pd.read_csv(R/'forecast_horizon_6h.csv'); q=raw.groupby('model')[['RMSE','MAE','MAPE','R2']].agg(['mean','std']); tab=table(9).set_index('Model')
for m in tab.index:
    for col,tc in [('RMSE','RMSE kWh'),('MAE','MAE kWh'),('MAPE','MAPE %'),('R2','R2')]:
        mean,sd=pm(tab.loc[m,tc]); close(q.loc[m,(col,'mean')],mean,0.02,f'T9 {m} {col} mean'); close(q.loc[m,(col,'std')],sd,0.02,f'T9 {m} {col} SD')

# Table 10 statistical summary
raw=pd.read_csv(R/'statistical_tests_holm.csv').set_index('baseline'); tab=table(10).set_index('Baseline')
for m in tab.index:
    close(raw.loc[m,'proposed_MAE'],num(tab.loc[m,'DOpt MAE']),0.002,f'T10 {m} prop MAE'); close(raw.loc[m,'baseline_MAE'],num(tab.loc[m,'Baseline MAE']),0.002,f'T10 {m} base MAE'); close(raw.loc[m,'p_holm'],num(tab.loc[m,'Holm p']),0.002,f'T10 {m} Holm')

# Tables 12/13 controlled ablations
for n,file,group,key in [(12,'feature_selection_canonical.csv','selector','Selector'),(13,'optimizer_roles_canonical.csv','strategy','Strategy')]:
    raw=pd.read_csv(R/file); q=raw.groupby(group)[['RMSE','MAE','R2']].agg(['mean','std']); tab=table(n).set_index(key)
    mapping={}
    if n==12: mapping={'All features':'All features','Random-k':'Random-k','Correlation-k':'Correlation-k','MOA':'MOA'}
    else: mapping={'No metaheuristic':'No metaheuristic','MOA only':'MOA only','DOA only':'DOA only - all features','MOA→DOA':'Sequential MOA→DOA'}
    for shown,rawname in mapping.items():
        mean,sd=pm(tab.loc[shown,'RMSE kWh']); close(q.loc[rawname,('RMSE','mean')],mean,0.02,f'T{n} {shown} RMSE'); close(q.loc[rawname,('RMSE','std')],sd,0.02,f'T{n} {shown} RMSE SD')

# Table 14 missingness means and degradation
raw=pd.read_csv(R/'missingness_cross_model.csv'); tab=table(14).set_index('Model')
for m in tab.index:
    g=raw[raw.model==m].groupby('missing_rate').RMSE.mean(); r0=g.loc[0.0]; r30=g.loc[0.3]; deg=100*(r30-r0)/r0
    close(r0,num(tab.loc[m,'0% RMSE']),0.02,f'T14 {m} 0%'); close(r30,num(tab.loc[m,'30% masked RMSE']),0.02,f'T14 {m} 30%'); close(deg,num(tab.loc[m,'Degradation']),0.03,f'T14 {m} degr')

# 10-seed stability claim
s=pd.read_csv(R/'stability_10seeds_encoder_matched.csv'); a=s[s.model=='DOptDRPSNN'].RMSE.to_numpy(); b=s[s.model=='DRA-GRU'].RMSE.to_numpy(); Fv=b.var(ddof=1)/a.var(ddof=1); pv=1-f.cdf(Fv,9,9); bf=levene(a,b,center='median')
close(a.std(ddof=1),0.1892322696,1e-6,'stability DOpt SD'); close(b.std(ddof=1),1.7334136161,1e-6,'stability DRA-GRU SD'); close(Fv,83.9100560,1e-5,'stability F'); close(pv,1.336570133e-7,1e-10,'stability p'); close(bf.pvalue,0.0027928092,1e-7,'Brown-Forsythe p')

# Tables 15-16 anomaly results/per-type
raw=pd.read_csv(R/'anomaly_results_final.csv').set_index('split'); tab=table(15).set_index('Split'); mapn={'Seen chronological':'seen_test','Unseen full':'unseen_full','Unseen temporal':'unseen_temporal'}
for shown,rn in mapn.items():
    for col in ['Accuracy','Precision','Recall','F1','AUPRC','AUROC']: close(raw.loc[rn,col],num(tab.loc[shown,col]),0.0015,f'T15 {shown} {col}')
per=pd.read_csv(R/'anomaly_per_type_unseen_full.csv').set_index('fault_type'); tab=table(16).set_index('Fault type')
for m in tab.index: close(100*per.loc[m,'detection_rate'],num(tab.loc[m,'Detection rate']),0.15,f'T16 {m} detection')

# Table 17 threshold values are sourced from final threshold sweep; other rows from sensitivity_canonical
sens=pd.read_csv(R/'sensitivity_canonical.csv'); th=pd.read_csv(R/'anomaly_threshold_sensitivity_final.csv'); tab=table(17)
for _,r in tab[tab['Parameter']=='Anomaly probability threshold'].iterrows():
    val=num(r['Value']); f1=num(r['Metric']); j=(th['threshold']-val).abs().idxmin(); close(th.loc[j,'F1'],f1,0.002,f'T17 threshold {val}')

# Table 18 primary controller + aggregate
raw=pd.read_csv(R/'control_results_final.csv'); raw=raw[raw.controller=='Model forecast + anomaly gate']; tab=table(18).set_index('Archetype')
for _,r in raw.iterrows():
    m=r.archetype; close(r.energy_reduction_pct,num(tab.loc[m,'Energy red.']),0.002,f'T18 {m} energy'); close(r.cost_reduction_pct,num(tab.loc[m,'Cost red.']),0.002,f'T18 {m} cost')
be=raw.baseline_kWh.sum(); ce=raw.controlled_kWh.sum(); bc=raw.baseline_cost_USD.sum(); cc=raw.controlled_cost_USD.sum(); er=100*(be-ce)/be; cr=100*(bc-cc)/bc
close(er,num(tab.loc['Aggregate','Energy red.']),0.002,'T18 aggregate energy'); close(cr,num(tab.loc['Aggregate','Cost red.']),0.002,'T18 aggregate cost')

if fail:
    print('VERIFY FAILED')
    for x in fail: print('-',x)
    sys.exit(1)
print('VERIFY PASSED: raw outputs reproduce the manuscript summaries checked for Tables 8-18 and the 10-seed stability claim.')
