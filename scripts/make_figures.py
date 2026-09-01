from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)

# Figure 1: unseen-building per-fault detection
p=pd.read_csv(R/'anomaly_per_type_unseen_full.csv').sort_values('detection_rate')
fig,ax=plt.subplots(figsize=(7.0,4.5)); ax.barh(p['fault_type'],100*p['detection_rate']); ax.set_xlabel('Detection rate (%)'); ax.set_title('Anomaly detection by fault type - unseen buildings')
for i,v in enumerate(100*p['detection_rate']): ax.text(v+1,i,f'{v:.1f}%',va='center',fontsize=8)
fig.tight_layout(); fig.savefig(F/'figure_1_anomaly_unseen.png',dpi=180); plt.close(fig)

# Figure 3: one-hour RMSE
f=pd.read_csv(R/'forecasting_models.csv'); q=f[f['split']=='unseen_full'].groupby('model').RMSE.agg(['mean','std']).reindex(['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer'])
cl=pd.read_csv(R/'classical_unseen_full.csv').set_index('model'); names=list(q.index)+['Persistence']; vals=list(q['mean'])+[cl.loc['Persistence','RMSE']]; errs=list(q['std'])+[0]
fig,ax=plt.subplots(figsize=(7.1,4.4)); ax.bar(range(len(names)),vals,yerr=errs,capsize=3); ax.set_xticks(range(len(names)),names,rotation=25,ha='right'); ax.set_ylabel('RMSE (kWh)'); ax.set_title('One-hour unseen-building forecasting'); fig.tight_layout(); fig.savefig(F/'figure_3_forecast_rmse.png',dpi=180); plt.close(fig)

# Figure 4: six-hour all-baseline RMSE
h=pd.read_csv(R/'forecast_horizon_6h.csv'); hs=h.groupby('model').RMSE.agg(['mean','std']).reindex(['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer','Persistence'])
fig,ax=plt.subplots(figsize=(7.1,4.4)); ax.bar(range(len(hs)),hs['mean'],yerr=hs['std'].fillna(0),capsize=3); ax.set_xticks(range(len(hs)),hs.index,rotation=25,ha='right'); ax.set_ylabel('RMSE (kWh)'); ax.set_title('Six-hour unseen-building forecasting'); fig.tight_layout(); fig.savefig(F/'figure_4_horizon.png',dpi=180); plt.close(fig)

# Figure 5: feature selection
fs=pd.read_csv(R/'feature_selection_canonical.csv').groupby('selector').RMSE.agg(['mean','std']).reindex(['All features','Random-k','Correlation-k','MOA'])
fig,ax=plt.subplots(figsize=(6.7,4.2)); ax.bar(range(4),fs['mean'],yerr=fs['std'],capsize=3); ax.set_xticks(range(4),fs.index,rotation=15); ax.set_ylabel('RMSE (kWh)'); ax.set_title('Feature-selection ablation'); fig.tight_layout(); fig.savefig(F/'figure_5_feature_selection.png',dpi=180); plt.close(fig)

# Figure 6: optimizer roles
op=pd.read_csv(R/'optimizer_roles_canonical.csv').groupby('strategy').RMSE.agg(['mean','std']).reindex(['No metaheuristic','MOA only','DOA only - all features','Sequential MOA→DOA'])
fig,ax=plt.subplots(figsize=(6.8,4.2)); ax.bar(range(4),op['mean'],yerr=op['std'],capsize=3); ax.set_xticks(range(4),['None','MOA only','DOA only','MOA→DOA']); ax.set_ylabel('RMSE (kWh)'); ax.set_title('Optimizer-role ablation'); fig.tight_layout(); fig.savefig(F/'figure_6_optimizer_roles.png',dpi=180); plt.close(fig)

# Figure 7: 10-seed encoder-matched initialization variability
s=pd.read_csv(R/'stability_10seeds_encoder_matched.csv'); st=s.groupby('model').RMSE.agg(['mean','std']).reindex(['DOptDRPSNN','DRA-GRU'])
fig,ax=plt.subplots(figsize=(6.8,4.4)); ax.bar(range(2),st['std']); ax.set_xticks(range(2),st.index); ax.set_ylabel('RMSE standard deviation (kWh)'); ax.set_title('10-seed encoder-matched initialization variability')
for i,(m,row) in enumerate(st.iterrows()): ax.text(i,row['std']+0.03,f"SD={row['std']:.3f}\nmean={row['mean']:.2f}",ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(F/'figure_7_stability.png',dpi=180); plt.close(fig)

# Figure 8: tariff-cost reduction, primary controller
c=pd.read_csv(R/'control_results_final.csv'); c=c[c['controller']=='Model forecast + anomaly gate'].copy()
fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.bar(c['archetype'],c['cost_reduction_pct']); ax.set_ylabel('Tariff-cost reduction (%)'); ax.set_title('Anomaly-gated supervisory control on unseen buildings'); ax.tick_params(axis='x',rotation=15)
for i,v in enumerate(c['cost_reduction_pct']): ax.text(i,v+0.004,f'{v:.3f}%',ha='center',fontsize=9)
ax.set_ylim(0,max(c['cost_reduction_pct'].max()*1.22,0.02)); fig.tight_layout(); fig.savefig(F/'figure_8_control.png',dpi=180,bbox_inches='tight'); plt.close(fig)
print('Figures regenerated:', F)
