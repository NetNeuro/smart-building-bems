"""Generate exact R2 facts, manuscript-ready tables, figures and factual report."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import HERE,OUT,protocol,sha,atomic_json,safe_csv,source_root

LABELS={'DRA_PSNN':'DRA-PSNN','DRA_GRU':'DRA-GRU','DRA_LSTM':'DRA-LSTM','Transformer':'Transformer','Persistence':'Persistence'}


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def main():
    p=protocol();dec=json.loads((HERE/'DECISIONS_LOCK.json').read_text())
    env=json.loads((HERE/'ENVIRONMENT.json').read_text())
    curves=pd.read_csv(OUT/'learning_curves_all.csv')
    table=pd.read_csv(OUT/'table_measured_forecasting.csv')
    cis=pd.read_csv(OUT/'paired_block_intervals.csv')
    bd=pd.read_csv(OUT/'paired_building_differences.csv')
    main_table=table[table.panel.eq('R2_fullpool_validation_selected')].copy()
    main_table['display_model']=main_table.model.map(LABELS)
    order=p['models']+['Persistence']
    main_table=main_table.set_index('model').loc[order].reset_index()
    safe_csv(main_table,OUT/'table_main_R2_selected.csv')
    old_table=table[table.panel.eq('R1_fixed6800_5epochs')].copy()
    safe_csv(old_table,OUT/'table_supplement_R1_fixed_budget.csv')
    choices={x['model']:x for x in dec['choices']}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    figs=[]
    for kind in ['objective','validation_RMSE']:
        fig,axs=plt.subplots(2,2,figsize=(9,6.4),layout='constrained')
        for ax,name in zip(axs.flat,p['models']):
            g=curves[curves.model.eq(name)]
            series=[('diagnostic_training','#177B73','Training diagnostic (6,800)') ,('validation','#A25436','Validation (17,296)')] if kind=='objective' else [('validation','#285F89','Validation RMSE')]
            column='normalized_MSE' if kind=='objective' else 'RMSE_kWh'
            for split,color,label in series:
                z=g[g.split.eq(split)].groupby('epoch')[column].agg(['mean','std']).sort_index()
                ax.plot(z.index,z['mean'],color=color,label=label,lw=1.8)
                ax.fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],color=color,alpha=.16)
            epoch=choices[name]['selected_epoch']
            ax.axvline(epoch,color='#666666',ls=':',lw=1.1,label=f'Locked epoch {epoch}')
            ax.set_title(LABELS[name]);ax.set_xlabel('Epoch on all 84,928 training windows')
            ax.set_ylabel('Normalized increment MSE' if kind=='objective' else 'Validation RMSE (kWh)')
            ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False)
        stem='Figure_Measured_Training_Validation_Curves' if kind=='objective' else 'Figure_Measured_Validation_RMSE'
        fig.savefig(HERE/(stem+'.png'),dpi=300);fig.savefig(HERE/(stem+'.pdf'));plt.close(fig)
        figs.append(stem+'.png')
    paired=bd[(bd.panel=='R2_fullpool_validation_selected')&(bd.comparator=='Persistence')].set_index('building_id').loc[p['buildings']]
    fig,ax=plt.subplots(figsize=(8,5.4),layout='constrained')
    y=np.arange(12);v=paired.delta_CVRMSE_percentage_points.to_numpy()
    lower=paired.delta_CVRMSE_percentage_points_lower95.to_numpy();upper=paired.delta_CVRMSE_percentage_points_upper95.to_numpy()
    # Endpoints need not enclose a percentile-bootstrap point estimate; draw segments directly.
    for i in range(12):ax.plot([lower[i],upper[i]],[i,i],color='#6A8B9B',lw=1.5)
    ax.scatter(v,y,color='#164E65',s=23,zorder=3)
    ax.axvline(0,color='#444444',ls='--',lw=1)
    ax.set_yticks(y,p['buildings'],fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('DRA-PSNN − persistence CVRMSE (percentage points)')
    ax.set_title('Paired building differences; fixed-panel 14-day block intervals')
    ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'Figure_Measured_Paired_Buildings.png',dpi=300);fig.savefig(HERE/'Figure_Measured_Paired_Buildings.pdf');plt.close(fig)
    figs.append('Figure_Measured_Paired_Buildings.png')
    selected=main_table.set_index('model');ps=selected.loc['DRA_PSNN'];pers=selected.loc['Persistence']
    primary=cis[(cis.panel=='R2_fullpool_validation_selected')&(cis.block_days==14)]
    legacy=cis[(cis.panel=='R1_fixed6800_5epochs')&(cis.block_days==14)]
    wins={}
    for c in p['uncertainty']['comparators']:
        g=bd[(bd.panel=='R2_fullpool_validation_selected')&(bd.comparator==c)]
        wins[c]={'buildings_with_negative_RMSE_difference':int((g.delta_RMSE_kWh<0).sum()),
                 'buildings_with_negative_MAE_difference':int((g.delta_MAE_kWh<0).sum()),'buildings':12}
    curve_last=curves[curves.epoch.eq(dec['terminal_epoch_cap'])&curves.split.eq('validation')]
    val_budget=[]
    original_validation=pd.read_csv(source_root()/'revision/external/results/forecast_metrics_seed.csv')
    old_names={'DRA_PSNN':'DRA-PSNN (measured telemetry)','DRA_GRU':'DRA-GRU','DRA_LSTM':'DRA-LSTM','Transformer':'Transformer'}
    for name in p['models']:
        q=choices[name]
        oldval=float(original_validation[(original_validation.model==old_names[name])&(original_validation.split=='validation')].RMSE_kWh.mean())
        val_budget.append({'model':name,'selected_epoch':q['selected_epoch'],
                                           'updates':q['updates_at_selected_epoch'],'examples_processed':84928*q['selected_epoch'],
                                           'validation_RMSE_mean_kWh':q['mean_validation_RMSE_kWh'],
                                           'legacy_validation_RMSE_mean_kWh':oldval,
                                           'validation_RMSE_relative_improvement_pct':100*(oldval-q['mean_validation_RMSE_kWh'])/oldval,
                                           'legacy_updates':135,'selected_updates_over_legacy':q['updates_at_selected_epoch']/135})
    safe_csv(pd.DataFrame(val_budget),OUT/'table_budget_checkpoint_choices.csv')
    facts={'study':'Measured BDG2 temporal forecasting after retraining on the same12buildings; no unseen-site/building transfer.',
           'n_buildings':12,'n_sites':6,'n_train_eligible':84928,'n_validation':17296,'n_test_common':98483,
           'expected_test_building_hours':105120,'seeds':p['seeds'],'initial_epoch_cap':10,
           'actual_terminal_epoch_cap':dec['terminal_epoch_cap'],
           'still_budget_sensitive_at_maximum':dec['still_budget_sensitive_at_maximum'],
           'selected_epochs':{k:v['selected_epoch'] for k,v in choices.items()},
           'selected_validation_budget':val_budget,'new_selected_models':main_table.to_dict('records'),
           'relative_RMSE_reduction_vs_persistence_pct':100*(pers.RMSE_kWh_mean-ps.RMSE_kWh_mean)/pers.RMSE_kWh_mean,
           'relative_MAE_reduction_vs_persistence_pct':100*(pers.MAE_kWh_mean-ps.MAE_kWh_mean)/pers.MAE_kWh_mean,
           'legacy_fixed_budget_models':old_table.to_dict('records'),
           'paired_14day_new':primary.to_dict('records'),'paired_14day_legacy':legacy.to_dict('records'),
           'building_direction_counts':wins,'protocol_sha256':sha(HERE/'PROTOCOL.json'),
           'decision_lock_sha256':sha(HERE/'DECISIONS_LOCK.json'),'hardware_software':env,
           'total_training_process_CPU_seconds':float(curve_last.train_cpu_seconds.sum()),
           'training_update_count_per_epoch':332,'legacy_training_update_count':135,
           'table_main':'r2_experiments/measured/results/table_main_R2_selected.csv',
           'table_legacy':'r2_experiments/measured/results/table_supplement_R1_fixed_budget.csv',
           'coverage_csv':'r2_experiments/measured/results/table_building_coverage_exclusions.csv',
           'paired_building_csv':'r2_experiments/measured/results/paired_building_differences.csv',
           'interval_csv':'r2_experiments/measured/results/paired_block_intervals.csv',
           'learning_curve_csv':'r2_experiments/measured/results/learning_curves_all.csv',
           'figures':['r2_experiments/measured/'+x for x in figs],
           'verification':'r2_experiments/measured/results/VERIFICATION.txt',
           'interpretation_constraints':p['uncertainty']['inference_scope'],
           'previously_observed_test':p['prior_holdout_status']}
    atomic_json(clean(facts),HERE/'FACTS.json')
    ci_ps=primary[(primary.scheme=='fixed_panel_synchronized_time')&(primary.comparator=='Persistence')&(primary.metric=='pooled_delta_RMSE_kWh')].iloc[0]
    ci_site=primary[(primary.scheme=='exploratory_site_and_time')&(primary.comparator=='Persistence')&(primary.metric=='pooled_delta_RMSE_kWh')].iloc[0]
    old_ps=old_table[old_table.model.eq('DRA_PSNN')].iloc[0]
    old_pers=old_table[old_table.model.eq('Persistence')].iloc[0]
    olddelta=old_ps.RMSE_kWh_mean-old_pers.RMSE_kWh_mean
    epochtext=', '.join(f'{LABELS[m]}: {choices[m]["selected_epoch"]}' for m in p['models'])
    sensitivity='At least one model still met the predeclared budget-sensitivity trigger at the maximum cap; no full-convergence or global-optimum claim is made.' if dec['still_budget_sensitive_at_maximum'] else 'The predeclared extension trigger was no longer met at the terminal cap; this is an empirical budget-plateau diagnostic, not a global-optimum certificate.'
    report=f'''# Measured forecasting R2 evidence

Protocol SHA-256: {sha(HERE/'PROTOCOL.json')}
Decision-lock SHA-256: {sha(HERE/'DECISIONS_LOCK.json')}

## Exact data/source preservation

The exact R1 architecture-class file, canonical source, external preprocessing
script and numerical configurations were copied and hashed before execution.
The source package was not modified. All four official BDG2 source hashes were
verified. Input scaling and increment normalization are bitwise identical to R1.
All 84,928 eligible training sequences, 17,296 validation sequences and 98,483
test targets have their original row bindings. Training is January–October
2016, validation November–December 2016, and the test grid is calendar 2017.
These are later-year predictions after retraining on the same twelve buildings
at six sites; this is not held-out-building/site or zero-shot transfer.

Selection uses only Jan–Oct2016: >=95% finite nonnegative meter values, median
and SD both >1kWh, and positive published floor area. Take the first six
lexicographic sites with at least two eligible buildings; prefer the first
lexicographic education and office meter where present, then lexicographic
fallback. No test-dependent meter selection or cleaning was added. Of 105,120
expected2017building-hours, 4,417 lack a target,57 an anchor and2,163 a complete
prior window, leaving98,483 common scored targets. Counts are mutually
exclusive in that order. Raw zeros/extremes remain. Some Crow scored series end
in November; full-calendar describes the evaluation grid, not complete measured
coverage in every building. Detailed coverage and exclusions are supplied.

## Training-budget/checkpoint evidence

All four models and all three original seeds used every eligible training
sequence, batch256, and the original AdamW settings without a new search or LR
schedule. Every epoch records evaluation-mode normalized MSE on the fixed6,800
training diagnostic windows and all17,296 validation windows, plus validation
meter RMSE/MAE, updates, exposures and process CPU/wall timing. The maximum
actually executed stage was {dec['terminal_epoch_cap']}epochs; there are332
updates per full-pool epoch, compared with135 total updates in the legacy run.
Selected epochs (same epoch for the three seeds of each model) were {epochtext}.
The rule minimizes mean validation RMSE across seeds, with earliest exact ties.
The predeclared10→20→40 escalation rule was applied only to2016validation:
extend all runs when any model improves >=0.5% in the last5epochs or has its
best mean checkpoint in the last2epochs. Optimizer/RNG states were resumed.
{sensitivity}

All budgets/checkpoints and validation-file hashes were locked before additional
2017 scoring. The2017holdout had already been reported in R1; this follow-up is
therefore explicitly a reused-holdout robustness audit rather than pristine
external confirmation. No further training/selection follows2017scoring.
Equal-update landmarks135/540/1350 are validation diagnostics, not equal-compute
or pure sample-size-effect experiments.

## Final selected-checkpoint results

DRA-PSNN: RMSE {ps.RMSE_kWh_mean:.6f} +/- {ps.RMSE_kWh_SD:.6f}kWh;
MAE {ps.MAE_kWh_mean:.6f} +/- {ps.MAE_kWh_SD:.6f}kWh;
R² {ps.R2_mean:.6f}; median-building CVRMSE
{ps.median_building_CVRMSE_pct:.6f}%.
Persistence: RMSE {pers.RMSE_kWh_mean:.6f}kWh, MAE
{pers.MAE_kWh_mean:.6f}kWh, median-building CVRMSE
{pers.median_building_CVRMSE_pct:.6f}%.
All comparators/seeds are in table_main_R2_selected.csv. The legacy fixed-budget
table is retained separately and is not silently overwritten.

## Paired building/time uncertainty

The12-row-per-comparison building tables report paired DRA-PSNN minus
persistence/GRU/LSTM differences on identical rows; negative favorsPSNN.
Across-seed SD measures training variability and is separate from the
exploratory building/time intervals. Pooled point estimates retain the mean of
seed-specific metrics rather than substituting ensemble-prediction error.

For5,000 replicates, fourteen-calendar-day moving blocks are drawn within each
calendar quarter and synchronized across all buildings/models/seeds. Original
missing masks remain attached to dates; denominators are recomputed. Seven-
and28-day sensitivities were fixed in advance. Fitted seed models are held
fixed. The site/time sensitivity resamples the six site clusters, keeping both
buildings in each chosen site; site draws and time draws are shared between
legacy and new panels. These are pointwise95%percentile intervals conditional
on the selected data and fitted models. Six deterministically selected sites do
not support broad building-population/site-transfer inference. There are no
hour-iid significance tests, confirmatory rankings, equivalence or economic
claims.

For the selected model versus persistence, pooled deltaRMSE is
{ci_ps.estimate:.6f}kWh, fixed-panel14-day interval
[{ci_ps.lower95:.6f},{ci_ps.upper95:.6f}]kWh. The exploratory site/time interval
is [{ci_site.lower95:.6f},{ci_site.upper95:.6f}]kWh. Complete GRU/LSTM comparisons,
MAE and equal-building normalized-error results, and block sensitivities are in
paired_block_intervals.csv. The legacy deltaRMSE is only
{olddelta:.6f}kWh ({100*olddelta/old_pers.RMSE_kWh_mean:.6f}% of persistence RMSE),
which remains numerically negligible and does not establish operational value.

## Verification and portable outputs

Verification recomputes every new metric from predictions, checks validation
checkpoint choices and immutable hashes, and replays the first1,024 test rows
of all12new and all12R1checkpoints at1e-8 tolerance. Paired pooled estimands and
common-row denominators are independently checked. Evidence:
results/VERIFICATION.txt. FACTS.json provides the main table, all selected epochs,
actual budget, paired intervals, relative artifact paths and actual hardware.
Execution uses {env['cpu_model']}, CPU-only, Python{env['python']},
PyTorch{env['torch']}, NumPy{env['numpy']}, pandas{env['pandas']},
scikit-learn{env['scikit_learn']}, SciPy{env['scipy']}. Training uses four worker
processes and two Torch threads per worker. Times exclude validation/I/O and do
not establish edge deployment or architectural compute equality.

## Figure captions

Training/validation curves: Mean normalized-increment MSE across seeds11,23,37
on the fixed6,800-window training diagnostic set and full17,296-window2016
validation set. Bands are sample SD across training seeds, not confidence bands
for buildings/time. Every optimization epoch uses all84,928 eligible training
windows. Vertical lines mark the architecture-specific validation-locked epoch.

ValidationRMSE curves: Meter-scale2016validationRMSE mean and sample SD across
three seeds through the predeclared terminal training stage. No2017score enters
the epoch or budget decision.

Paired-building plot: DRA-PSNN minus persistence CVRMSE differences for the same
12buildings and2017common rows. Negative values favorPSNN. Intervals are
exploratory pointwise95%fixed-panel, synchronized14-day block intervals; training
seeds are held fixed. These are later-year temporal comparisons after retraining,
not unseen-building/site or physical operational validation.
'''
    for joined,spaced in [('Jan–Oct2016','Jan–Oct 2016'),('>1kWh','>1 kWh'),
                           ('expected2017building-hours','expected 2017 building-hours'),
                           ('4,417 lack a target,57 an anchor and2,163','4,417 lack a target, 57 an anchor and 2,163'),
                           ('leaving98,483','leaving 98,483'),('batch256','batch 256'),
                           ('fixed6,800','fixed 6,800'),('all17,296','all 17,296'),
                           ('epochs; there are332','epochs; there are 332'),('with135','with 135'),
                           ('three seeds of each model) were','three seeds of each model) were'),
                           ('predeclared10→20→40','predeclared 10→20→40'),('only to2016validation','only to 2016 validation'),
                           ('last5epochs','last 5 epochs'),('last2epochs','last 2 epochs'),
                           ('2017holdout','2017 holdout'),('2017scoring','2017 scoring'),
                           ('landmarks135/540/1350','landmarks 135/540/1350'),
                           ('kWh;',' kWh;'),('}kWh','} kWh'),
                           ('The12-row','The 12-row'),('negative favorsPSNN','negative favors PSNN'),
                           ('For5,000','For 5,000'),('Seven-\nand28-day','Seven-\nand 28-day'),
                           ('pointwise95%percentile','pointwise 95% percentile'),
                           ('pooled deltaRMSE','pooled delta RMSE'),('fixed-panel14-day','fixed-panel 14-day'),
                           ('legacy deltaRMSE','legacy delta RMSE'),('first1,024','first 1,024'),
                           ('all12new and all12R1checkpoints','all 12 new and all 12 R1 checkpoints'),
                           ('1e-8 tolerance','1e-8 tolerance'),('seeds11,23,37','seeds 11, 23, 37'),
                           ('all84,928','all 84,928'),('three seeds through','three seeds through'),
                           ('No2017score','No 2017 score'),('12buildings and2017common','12 buildings and 2017 common'),
                           ('pointwise95%fixed-panel','pointwise 95% fixed-panel'),('synchronized14-day','synchronized 14-day')]:
        report=report.replace(joined,spaced)
    (HERE/'REPORT.md').write_text(report)
    print(main_table.to_string(index=False),flush=True)
    print('FACTS AND REPORT READY',flush=True)


if __name__=='__main__':main()
