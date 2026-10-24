"""Paired empirical building and synchronized calendar-block uncertainty.

Fitted seed models are held fixed. These are exploratory pointwise intervals,
not independent-hour tests, equivalence tests, or population/site-transfer claims.
"""
import datetime as dt
import json
import numpy as np
import pandas as pd
from common import HERE,OUT,protocol,source_root,sha,atomic_json,safe_csv


def calendar_weights(blockdays,B,seed):
    dates=pd.date_range('2017-01-01','2017-12-31',freq='D')
    rng=np.random.default_rng(np.random.SeedSequence([seed,blockdays]))
    W=np.zeros((B,len(dates)),dtype=np.float64)
    for quarter in [1,2,3,4]:
        q=np.flatnonzero(dates.quarter==quarter);n=len(q);k=int(np.ceil(n/blockdays))
        starts=rng.integers(0,n-blockdays+1,size=(B,k))
        take=(starts[:,:,None]+np.arange(blockdays)).reshape(B,-1)[:,:n]
        for b in range(B):np.add.at(W[b],q[take[b]],1.)
    assert np.all(W.sum(1)==365)
    # The same site multiplicities are used in both panels and all block-length sensitivity runs.
    srng=np.random.default_rng(np.random.SeedSequence([seed,6]))
    siteidx=srng.integers(0,6,size=(B,6))
    cluster=np.stack([np.bincount(v,minlength=6) for v in siteidx])
    return W,cluster


def load_losses(panel,base,p):
    names=p['models']+['Persistence'];B=len(p['buildings']);S=len(p['seeds'])
    squared=np.zeros((365,B,len(names),S));absolute=np.zeros_like(squared)
    count=np.zeros((365,B));actual=np.zeros((365,B))
    day=(base.timestamp.dt.floor('D')-pd.Timestamp('2017-01-01')).dt.days.to_numpy()
    bids=np.array([p['buildings'].index(b) for b in base.building_id])
    yy=base.actual_kWh.to_numpy(float)
    np.add.at(count,(day,bids),1.);np.add.at(actual,(day,bids),yy)
    legacy=source_root()/'revision/external/results'
    for mi,name in enumerate(names):
        for si,seed in enumerate(p['seeds']):
            if name=='Persistence':pp=base.persistence_kWh.to_numpy(float)
            else:
                path=(legacy/f'{name}_seed{seed}_test_predictions.csv.gz' if panel=='R1_fixed6800_5epochs' else
                      OUT/'predictions'/f'{name}_seed{seed}_R2_fullpool_test.csv.gz')
                df=pd.read_csv(path,parse_dates=['timestamp'])
                assert df.building_id.equals(base.building_id) and df.timestamp.equals(base.timestamp)
                assert np.allclose(df.actual_kWh,yy,rtol=0,atol=1e-12)
                pp=df.prediction_kWh.to_numpy(float)
            err=pp-yy
            np.add.at(squared[:,:,mi,si],(day,bids),err*err)
            np.add.at(absolute[:,:,mi,si],(day,bids),np.abs(err))
    return squared,absolute,count,actual,names


def calculate(sq,ab,n,ysum,weights):
    """Inputs: replicates x buildings x models x seeds; paired site weights."""
    assert np.all(n>0),'A bootstrap panel contains an empty building-year'
    denom=(n*weights).sum(axis=1)
    pooled_rmse=np.sqrt((sq*weights[:,:,None,None]).sum(axis=1)/denom[:,None,None]).mean(axis=-1)
    pooled_mae=((ab*weights[:,:,None,None]).sum(axis=1)/denom[:,None,None]).mean(axis=-1)
    brmse=np.sqrt(sq/n[:,:,None,None]).mean(axis=-1)
    bmae=(ab/n[:,:,None,None]).mean(axis=-1)
    bcvrmse=100*brmse/(ysum/n)[:,:,None]
    return {'pooled_delta_RMSE_kWh':pooled_rmse,
            'pooled_delta_MAE_kWh':pooled_mae,
            'equal_building_delta_RMSE_kWh':(brmse*weights[:,:,None]).sum(axis=1)/weights.sum(1)[:,None],
            'equal_building_delta_CVRMSE_percentage_points':(bcvrmse*weights[:,:,None]).sum(axis=1)/weights.sum(1)[:,None]},brmse,bmae,bcvrmse


def main():
    assert (HERE/'TEST_SCORED.json').exists()
    p=protocol();u=p['uncertainty'];reps=u['replicates']
    base=pd.read_csv(OUT/'common_test_rows.csv.gz',parse_dates=['timestamp'])
    results=[];building_rows=[];point_rows=[]
    for panel in ['R1_fixed6800_5epochs','R2_fullpool_validation_selected']:
        sq,ab,n,y,names=load_losses(panel,base,p)
        # Fixed observed-panel point estimands retain mean-of-seed metrics.
        point,brmse,bmae,bcvr=calculate(sq.sum(0)[None],ab.sum(0)[None],n.sum(0)[None],y.sum(0)[None],np.ones((1,12)))
        for comp in u['comparators']:
            ci=names.index(comp)
            for metricname,value in point.items():
                point_rows.append({'panel':panel,'comparator':comp,'metric':metricname,'estimate':float(value[0,0]-value[0,ci])})
            for bi,bid in enumerate(p['buildings']):
                building_rows.append({'panel':panel,'building_id':bid,'site_id':bid.split('_')[0],
                                      'common_valid_hours':int(n[:,bi].sum()),'comparator':comp,
                                      'PSNN_RMSE_kWh':float(brmse[0,bi,0]),'comparator_RMSE_kWh':float(brmse[0,bi,ci]),
                                      'delta_RMSE_kWh':float(brmse[0,bi,0]-brmse[0,bi,ci]),
                                      'PSNN_MAE_kWh':float(bmae[0,bi,0]),'comparator_MAE_kWh':float(bmae[0,bi,ci]),
                                      'delta_MAE_kWh':float(bmae[0,bi,0]-bmae[0,bi,ci]),
                                      'delta_CVRMSE_percentage_points':float(bcvr[0,bi,0]-bcvr[0,bi,ci])})
        for L in u['block_days']:
            W,siteweights=calendar_weights(L,reps,u['bootstrap_seed'])
            daily_sq=(W@sq.reshape(365,-1)).reshape(reps,12,len(names),3)
            daily_ab=(W@ab.reshape(365,-1)).reshape(reps,12,len(names),3)
            daily_n=W@n;daily_y=W@y
            assert np.all(daily_n>0)
            site_to_index={site:i for i,site in enumerate(p['sites'])}
            building_site=np.array([site_to_index[b.split('_')[0]] for b in p['buildings']])
            for scheme,weights in [('fixed_panel_synchronized_time',np.ones((reps,12))),
                                   ('exploratory_site_and_time',siteweights[:,building_site])]:
                metrics,rr,ma,cv=calculate(daily_sq,daily_ab,daily_n,daily_y,weights)
                for comp in u['comparators']:
                    ci=names.index(comp)
                    for metricname,value in metrics.items():
                        delta=value[:,0]-value[:,ci]
                        lower,upper=np.quantile(delta,[.025,.975])
                        results.append({'panel':panel,'scheme':scheme,'block_days':L,'replicates':reps,
                                        'comparator':comp,'metric':metricname,
                                        'estimate':float(point[metricname][0,0]-point[metricname][0,ci]),
                                        'lower95':float(lower),'upper95':float(upper),
                                        'scope':'Exploratory pointwise, conditional fitted seeds; six selected sites; no site-transfer/population/economic claim.'})
                # Time-only per-building intervals retain original building identity.
                if L==14 and scheme=='fixed_panel_synchronized_time':
                    for row in [x for x in building_rows if x['panel']==panel]:
                        bi=p['buildings'].index(row['building_id']);ci=names.index(row['comparator'])
                        for metricname,value in [('delta_RMSE_kWh',rr),('delta_MAE_kWh',ma),('delta_CVRMSE_percentage_points',cv)]:
                            lo,hi=np.quantile(value[:,bi,0]-value[:,bi,ci],[.025,.975])
                            row[metricname+'_lower95']=float(lo);row[metricname+'_upper95']=float(hi)
            np.savez_compressed(OUT/f'bootstrap_draws_block{L}_days.npz',calendar_day_multiplicities=W.astype(np.uint16),site_cluster_multiplicities=siteweights.astype(np.uint8))
            print('BOOTSTRAP COMPLETE',panel,L,'days',reps,'replicates',flush=True)
    safe_csv(pd.DataFrame(results),OUT/'paired_block_intervals.csv')
    safe_csv(pd.DataFrame(building_rows),OUT/'paired_building_differences.csv')
    safe_csv(pd.DataFrame(point_rows),OUT/'paired_point_estimands.csv')
    atomic_json({'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'protocol_sha256':sha(HERE/'PROTOCOL.json'),'paired_interval_sha256':sha(OUT/'paired_block_intervals.csv'),
                 'common_rows_all_models':len(base),'bootstrap_seed':u['bootstrap_seed'],
                 'replicates':reps,'block_days':u['block_days'],
                 'rng':'SeedSequence([20261022,block_days]) for synchronized time, SeedSequence([20261022,6]) for site draws.',
                 'inferential_scope':u['inference_scope'],'all_bootstrap_building_year_denominators_positive':True},OUT/'UNCERTAINTY_LOCK.json')


if __name__=='__main__':main()
