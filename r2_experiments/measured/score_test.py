"""Score the reused 2017 holdout only after all R2 validation choices are locked."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import torch
from common import HERE,SOURCE,CACHE,OUT,STATES,protocol,sha,atomic_json,safe_csv,arrays,source_root,make_model,standard_predict,reconstruct,metric


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    p=protocol();lock=json.loads((HERE/'DECISIONS_LOCK.json').read_text())
    assert lock['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    assert lock['learning_curves_sha256']==sha(OUT/'learning_curves_all.csv')
    assert lock['data_lock_sha256']==sha(CACHE/'DATA_LOCK.json')
    if not (HERE/'TEST_SCORING_STARTED.json').exists():
        atomic_json({'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                     'decision_lock_sha256':sha(HERE/'DECISIONS_LOCK.json'),
                     'protocol_sha256':sha(HERE/'PROTOCOL.json'),
                     'status':'All new budgets/checkpoints fixed before this program loads2017 arrays.'},HERE/'TEST_SCORING_STARTED.json')
    S=arrays('test');legacyroot=source_root()/'revision/external/results'
    pred_dir=OUT/'predictions';pred_dir.mkdir(exist_ok=True)
    base=pd.DataFrame({'building_id':S['bid'],'site_id':S['site'],'timestamp':pd.to_datetime(S['ts']),
                       'actual_kWh':S['actual'],'persistence_kWh':S['anchor']})
    assert len(base)==98483 and not base.duplicated(['building_id','timestamp']).any()
    safe_csv(base,OUT/'common_test_rows.csv.gz')
    rows=[];building=[];replay=[]
    choices={x['model']:x for x in lock['choices']}
    for name in p['models']:
        choice=choices[name]
        for seed in p['seeds']:
            path=STATES/f'{name}_seed{seed}'/f"epoch{choice['selected_epoch']:04d}_weights.pt"
            assert sha(path)==choice['selected_weight_sha256_by_seed'][str(seed)]
            model=make_model(name,seed)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            pred=reconstruct(standard_predict(model,S['X']),S)
            frame=base[['building_id','site_id','timestamp','actual_kWh']].copy()
            frame['prediction_kWh']=pred
            safe_csv(frame,pred_dir/f'{name}_seed{seed}_R2_fullpool_test.csv.gz')
            rows.append({'panel':'R2_fullpool_validation_selected','model':name,'seed':seed,
                         'epoch':choice['selected_epoch'],'n':len(base),**metric(S['actual'],pred)})
            for bid in p['buildings']:
                z=S['bid']==bid
                building.append({'panel':'R2_fullpool_validation_selected','model':name,'seed':seed,
                                 'building_id':bid,'site_id':bid.split('_')[0],'n':int(z.sum()),
                                 **metric(S['actual'][z],pred[z])})
            old=pd.read_csv(legacyroot/f'{name}_seed{seed}_test_predictions.csv.gz',parse_dates=['timestamp'])
            assert old.building_id.equals(base.building_id)
            assert old.timestamp.equals(base.timestamp)
            assert np.allclose(old.actual_kWh,S['actual'],rtol=0,atol=1e-12)
            oldpred=old.prediction_kWh.to_numpy()
            rows.append({'panel':'R1_fixed6800_5epochs','model':name,'seed':seed,'epoch':5,'n':len(base),**metric(S['actual'],oldpred)})
            for bid in p['buildings']:
                z=S['bid']==bid
                building.append({'panel':'R1_fixed6800_5epochs','model':name,'seed':seed,
                                 'building_id':bid,'site_id':bid.split('_')[0],'n':int(z.sum()),
                                 **metric(S['actual'][z],oldpred[z])})
            # R1 fitted-weight replay requested by the source audit; no new fitting.
            oldmodel=make_model(name,seed)
            oldmodel.load_state_dict(torch.load(legacyroot/f'{name}_seed{seed}.pt',map_location='cpu',weights_only=True))
            small={k:np.asarray(v[:1024]) for k,v in S.items()}
            oldreplay=reconstruct(standard_predict(oldmodel,small['X']),small)
            maxdiff=float(np.max(np.abs(oldreplay-oldpred[:1024])))
            assert np.allclose(oldreplay,oldpred[:1024],rtol=1e-8,atol=1e-8),(name,seed,maxdiff)
            replay.append({'model':name,'seed':seed,'replayed_rows':1024,'maximum_absolute_difference_kWh':maxdiff,'passed':True})
            print('SCORED',name,seed,'epoch',choice['selected_epoch'],rows[-2],flush=True)
    for panel in ['R1_fixed6800_5epochs','R2_fullpool_validation_selected']:
        rows.append({'panel':panel,'model':'Persistence','seed':0,'epoch':0,'n':len(base),**metric(S['actual'],S['anchor'])})
        for bid in p['buildings']:
            z=S['bid']==bid
            building.append({'panel':panel,'model':'Persistence','seed':0,'building_id':bid,
                             'site_id':bid.split('_')[0],'n':int(z.sum()),**metric(S['actual'][z],S['anchor'][z])})
    safe_csv(pd.DataFrame(rows),OUT/'test_metrics_seed.csv')
    safe_csv(pd.DataFrame(building),OUT/'test_metrics_building_seed.csv')
    safe_csv(pd.DataFrame(replay),OUT/'r1_checkpoint_replay.csv')
    summary=[]
    seeddf=pd.DataFrame(rows);bdf=pd.DataFrame(building)
    for (panel,name),g in seeddf.groupby(['panel','model'],sort=False):
        b=bdf[(bdf.panel==panel)&(bdf.model==name)]
        median=b.groupby('seed').CVRMSE_pct.median().mean()
        r={'panel':panel,'model':name,'epoch':int(g.epoch.iloc[0]),'seeds':len(g),'test_targets':len(base),
           'median_building_CVRMSE_pct':float(median)}
        for metricname in ['RMSE_kWh','MAE_kWh','R2','CVRMSE_pct']:
            r[metricname+'_mean']=float(g[metricname].mean())
            r[metricname+'_SD']=float(g[metricname].std(ddof=1)) if len(g)>1 else None
        summary.append(r)
    safe_csv(pd.DataFrame(summary),OUT/'table_measured_forecasting.csv')
    # Common-row identity is required across all12 runs, and persistence uses those rows too.
    atomic_json({'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'protocol_sha256':sha(HERE/'PROTOCOL.json'),'decision_lock_sha256':sha(HERE/'DECISIONS_LOCK.json'),
                 'seed_metrics_sha256':sha(OUT/'test_metrics_seed.csv'),
                 'n_common_test_rows':len(base),'r1_checkpoint_replay_all12_passed':True,
                 'no_training_or_selection_after_test_scoring':True},HERE/'TEST_SCORED.json')
    print('ALL FINAL TEST METRICS AND R1 REPLAYS COMPLETE',flush=True)


if __name__=='__main__':main()
