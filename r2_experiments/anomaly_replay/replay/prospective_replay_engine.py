"""Prospective next-interval supervisory replay and matched-curtailment audit.

This is a synthetic load-response replay, not a thermodynamic simulation or a
physical closed loop. Forecast inputs are the archived unmodified meter stream.
Decision at the release of meter interval t uses p_t, occupancy_t, and the
forecast for t+1, whose input window ends at t. The known t+1 tariff is allowed.
Rebound stays in its originating building and has a frozen pre-evaluation
observed-meter ceiling. No holdout outcome is used to tune the control rule.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
def raw_mult(ts):
    if ts.dayofweek<5 and 16<=ts.hour<21:return 1.65
    if ts.dayofweek<5 and (8<=ts.hour<16 or 21<=ts.hour<23):return 1.05
    return .65

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-data',required=True);ap.add_argument('--forecast-model');ap.add_argument('--forecasts',required=True);ap.add_argument('--scores',required=True);ap.add_argument('--threshold',required=True,type=float);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.raw_data,parse_dates=['timestamp'])
    forecast=pd.read_csv(args.forecasts,parse_dates=['timestamp'])
    if 'model' in forecast:
        if args.forecast_model:
            forecast=forecast[forecast.model==args.forecast_model]
        elif forecast.model.nunique()!=1:
            raise ValueError('Combined multi-model forecasts require --forecast-model.')
    if 'seed' in forecast:
        forecast=forecast.groupby(['building_id','timestamp'],as_index=False).forecast.mean()
    fc=forecast[['building_id','timestamp','forecast']].copy()
    if fc.duplicated(['building_id','timestamp']).any():raise ValueError('Forecast keys must be unique after ensemble mean.')
    score=pd.read_csv(args.scores,parse_dates=['timestamp'])
    pn=next(c for c in ['probability','anomaly_probability','score'] if c in score)
    score=score[['building_id','timestamp',pn]].rename(columns={pn:'anomaly_probability'})
    if score.duplicated(['building_id','timestamp']).any():raise ValueError('Anomaly keys must be unique.')
    g=fc.merge(raw,on=['building_id','timestamp'],validate='one_to_one').merge(score,on=['building_id','timestamp'],validate='one_to_one').sort_values(['building_id','timestamp']).reset_index(drop=True)
    if not (g.building_id>=9).all():raise ValueError('Main control requires unseen buildings 9–11 only.')
    train=pd.concat([z.sort_values('timestamp').iloc[:int(.7*len(z))] for b,z in raw[raw.building_id<9].groupby('building_id')])
    ratio=float(np.median([z.clean.quantile(.9)/np.sqrt(z.square_feet.iloc[0]) for b,z in train.groupby('building_id')]))
    g['load_threshold_kWh']=ratio*np.sqrt(g.square_feet)
    g['tariff_multiplier']=[raw_mult(t) for t in g.timestamp]
    g['price_USD_per_kWh']=.1341*g.tariff_multiplier/g.tariff_multiplier.mean()
    prior=score.rename(columns={'anomaly_probability':'gate_probability_available'}).copy()
    prior['timestamp']=prior.timestamp+pd.Timedelta(hours=1)
    prior_occ=raw[['building_id','timestamp','occupancy']].rename(columns={'occupancy':'occupancy_available'}).copy()
    prior_occ['timestamp']=prior_occ.timestamp+pd.Timedelta(hours=1)
    g=g.merge(prior,on=['building_id','timestamp'],how='left',validate='one_to_one').merge(prior_occ,on=['building_id','timestamp'],how='left',validate='one_to_one')
    g['decision_timestamp']=g.timestamp-pd.Timedelta(hours=1)
    g['gate_available']=g.gate_probability_available.notna()&g.occupancy_available.notna()
    g['gate_flagged']=g.gate_available&(g.gate_probability_available>=args.threshold)
    g['first_interval_no_gate']=~g.gate_available
    g['rebound_ceiling_kW']=np.nan
    for b,z in g.groupby('building_id'):
        history=raw[(raw.building_id==b)&(raw.timestamp<z.timestamp.min())]
        if len(history)==0:raise ValueError('A pre-evaluation observed-meter history is required for the rebound ceiling.')
        g.loc[z.index,'rebound_ceiling_kW']=float(history.meter.max())
    baseline=g.clean.to_numpy();hvac=g.hvac.to_numpy();price=g.price_USD_per_kWh.to_numpy()
    def fractions(fc,gate):
        risk=(fc-g.load_threshold_kWh)/np.maximum(g.load_threshold_kWh,1.)
        peak=g.tariff_multiplier>=1.6;shoulder=(g.tariff_multiplier>.7)&~peak
        frac=np.where(peak&(risk>0)&(g.occupancy_available<=.9),np.minimum(.12,.04+.12*risk),np.where(shoulder&(risk>.05)&(g.occupancy_available<=.8),np.minimum(.06,.015+.06*risk),0.))
        frac=np.where(g.gate_available,frac,0.)
        if gate:frac=np.where(g.gate_flagged,0.,frac)
        return frac
    def replay(curtailed):
        rebound=np.zeros(len(g));terminal=np.zeros(len(g));unreturned=0.
        for b,z in g.groupby('building_id'):
            ii=z.index.to_numpy();scheduled={};pending=[]
            # Scheduling uses public tariff order only. Saturation is applied at
            # the actual response interval, never to select a future hour.
            for i in ii:
                q=.35*curtailed[i]
                if q<=0:continue
                eligible=[k for k in ii if i<k<=i+6 and g.timestamp.iloc[k]<=g.timestamp.iloc[i]+pd.Timedelta(hours=6)]
                if not eligible:terminal[i]+=q;continue
                k=min(eligible,key=lambda k:(price[k],k));scheduled.setdefault(k,[]).append([q,i+6])
            for k in ii:
                pending.extend(scheduled.get(k,[]))
                available=max(0.,g.rebound_ceiling_kW.iloc[k]-(baseline[k]-curtailed[k]))
                remain=[]
                for q,deadline in pending:
                    take=min(q,available);rebound[k]+=take;available-=take;q-=take
                    if q>1e-12:
                        if k>=deadline:unreturned+=q
                        else:remain.append([q,deadline])
                pending=remain
            unreturned+=sum(q for q,d in pending)
        controlled=baseline-curtailed+rebound
        return controlled,rebound,terminal,float(unreturned)
    fm=fractions(g.forecast.to_numpy(),True);fo=fractions(baseline,True);fn=fractions(g.forecast.to_numpy(),False);fon=fractions(baseline,False)
    cm=fm*hvac;co=fo*hvac;cn=fn*hvac
    target=min(cm.sum(),co.sum())
    variants={'Prospective model + prior gate':cm,'Prospective perfect + prior gate':co,'Prospective model no gate':cn,'Matched gross model':cm*(target/cm.sum()) if cm.sum()>0 else cm,'Matched gross perfect':co*(target/co.sum()) if co.sum()>0 else co}
    rows=[];traces=[]
    for name,cur in variants.items():
        controlled,rebound,terminal,unreturned=replay(cur)
        trace=g.copy();trace['variant']=name;trace['otherwise_eligible']=(fon if 'perfect' in name.lower() else fn)>0;trace['ungated_candidate_fraction']=fon if 'perfect' in name.lower() else fn;trace['curtailed_kWh']=cur;trace['rebound_kWh']=rebound;trace['terminal_rebound_kWh']=terminal;trace['controlled_load']=controlled;traces.append(trace)
        for b,z in g.groupby('building_id'):
            ii=z.index.to_numpy();be=baseline[ii].sum();bc=(baseline[ii]*price[ii]).sum();ce=controlled[ii].sum();cc=(controlled[ii]*price[ii]).sum()
            rows.append({'variant':name,'building_id':int(b),'archetype':z.archetype.iloc[0],'hours':len(ii),'decision_hours':int(z.gate_available.sum()),'baseline_kWh':be,'controlled_kWh':ce,'net_saved_kWh':be-ce,'energy_reduction_pct':100*(be-ce)/be,'baseline_cost_USD':bc,'controlled_cost_USD':cc,'cost_saved_USD':bc-cc,'cost_reduction_pct':100*(bc-cc)/bc,'gross_curtailed_kWh':cur[ii].sum(),'rebound_kWh':rebound[ii].sum(),'terminal_rebound_kWh':terminal[ii].sum(),'intervention_hours':int((cur[ii]>0).sum()),'gate_flagged_hours':int(z.gate_flagged.sum()),'otherwise_eligible_gate_blocks':int((((fon if 'perfect' in name.lower() else fn)[ii]>0)&z.gate_flagged).sum()) if 'no gate' not in name else 0,'peak_baseline_kW':baseline[ii].max(),'peak_controlled_kW':controlled[ii].max(),'rebound_ceiling_kW':z.rebound_ceiling_kW.iloc[0],'unreturned_due_to_cap_kWh_all':unreturned})
    summaries=pd.DataFrame(rows);summaries.to_csv(out/'prospective_control_results.csv',index=False)
    trace=pd.concat(traces,ignore_index=True);trace.to_csv(out/'prospective_control_trace_all.csv',index=False)
    trace[trace.variant=='Prospective model + prior gate'].to_csv(out/'prospective_control_trace_primary.csv',index=False)
    agg=summaries.groupby('variant',sort=False)[['baseline_kWh','controlled_kWh','net_saved_kWh','baseline_cost_USD','controlled_cost_USD','cost_saved_USD','gross_curtailed_kWh','rebound_kWh','terminal_rebound_kWh','intervention_hours','gate_flagged_hours','otherwise_eligible_gate_blocks']].sum()
    agg['energy_reduction_pct']=100*agg.net_saved_kWh/agg.baseline_kWh;agg['cost_reduction_pct']=100*agg.cost_saved_USD/agg.baseline_cost_USD;agg.to_csv(out/'prospective_control_aggregate.csv')
    # Demand-charge effect over this partial-window replay is descriptive only;
    # it is not a complete utility bill. For unchanged per-building peak,
    # adding any nonnegative demand charge cannot increase absolute savings.
    rate=[]
    for name,z in summaries.groupby('variant'):
        for demand in [0.,10.,20.]:
            be=z.baseline_cost_USD.sum()+demand*z.peak_baseline_kW.sum();ce=z.controlled_cost_USD.sum()+demand*z.peak_controlled_kW.sum()
            rate.append({'variant':name,'illustrative_demand_USD_per_kW':demand,'baseline_partial_bill_USD':be,'controlled_partial_bill_USD':ce,'saved_USD':be-ce,'reduction_pct':100*(be-ce)/be,'context':'partial-window illustrative sensitivity, not a named utility/monthly bill'})
    pd.DataFrame(rate).to_csv(out/'demand_charge_sensitivity.csv',index=False)
    meta={'forecast_file':str(Path(args.forecasts).resolve()),'anomaly_file':str(Path(args.scores).resolve()),'gate_threshold':args.threshold,'evaluation_first_timestamp':str(g.timestamp.min()),'evaluation_last_timestamp':str(g.timestamp.max()),'decision_timing':'At release of meter interval t: gate p_t, occupancy_t, forecast Lhat_(t+1) from windows ending at t, known tariff_(t+1); action at t+1.','feedback_scope':'Sequential supervisory replay: original observed meter contexts are not regenerated after control; no physical building feedback or comfort validation.','rebound':'0.35 of gross curtailment assigned to lowest known tariff hour among next six, tie resolved earliest; same-building frozen pre-evaluation observed-meter peak ceiling; actual saturation at response interval.','matched_comparator':'Diagnostic equal aggregate gross curtailment by scaling the more active policy. No threshold retuning or selection using test accuracy.','training_load_threshold_ratio':ratio,'tariff':'Illustrative common TOU normalized on evaluation timestamps to 0.1341 USD/kWh; no seasonal-rate validation.','first_interval_policy':'No action without prior scored anomaly probability; actual availability follows the supplied score timestamps, including earlier scored history when present.','raw_data_sha256':__import__('hashlib').sha256(Path(args.raw_data).read_bytes()).hexdigest()}
    (out/'prospective_control_protocol.json').write_text(json.dumps(meta,indent=2))
    assert trace.groupby(['variant','building_id']).timestamp.apply(lambda z:z.is_monotonic_increasing).all()
    assert (trace.curtailed_kWh>=0).all()
    primary=summaries[summaries.variant=='Prospective model + prior gate']
    print(agg.to_string());print(primary[['building_id','intervention_hours','gate_flagged_hours','otherwise_eligible_gate_blocks','peak_baseline_kW','peak_controlled_kW']].to_string(index=False))
if __name__=='__main__':main()
