"""Compact supplementary views; no selections are made here."""
from pathlib import Path
import pandas as pd
H=Path(__file__).resolve().parent;D=H/'results';T=H/'tables';T.mkdir(exist_ok=True)
f=pd.read_csv(D/'all_detector_metrics.csv');x=f[(f.pipeline=='PSNN_12')&(f.condition=='cut60_causal_unpurged')&(f.split=='unseen_full')].copy()
order=['LR_original_configuration','frozen_GB','GB_tuned_development','LR_tuned_development'];x=x.set_index('method').loc[order].reset_index()
x[['method','threshold','Precision','Recall','F1','AP','AUROC','Normal_FPR']].to_csv(T/'supplement_supervised_primary_four_rows.csv',index=False)
c=pd.read_csv(H/'calibration_audit/calibration_fault_coverage.csv');c=c[c.phase.isin(['fit','threshold'])].pivot(index=['condition','nominal_fraction','purged','phase'],columns='fault_type',values='hours').reset_index();types=['bias','drift','hvac_deg','schedule','spike','stuck'];c['fault_hours']=c[types].sum(axis=1);c['total_hours']=c.fault_hours+c.normal
c[['condition','phase','normal',*types,'fault_hours','total_hours']].to_csv(T/'supplement_cut_fault_coverage_12_rows.csv',index=False)
b=pd.read_csv(H/'replay/pipeline_comparison/PSNN_12/decision_audit/blocked_event_audit.csv');cols=['timestamp','archetype','prior_interval','prior_atype','gate_probability_available','target_atype','otherwise_eligible','blocked_eligible','forecast','load_threshold_kWh','occupancy_available','tariff_multiplier','ungated_candidate_fraction']
b[cols].to_csv(T/'supplement_two_blocked_events.csv',index=False)
