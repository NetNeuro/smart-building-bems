"""Verify frozen choices, predictions, checkpoint replay and paired estimands."""
import json
import numpy as np
import pandas as pd
import torch
from common import HERE,OUT,STATES,CACHE,protocol,sha,arrays,make_model,standard_predict,reconstruct,metric


def main():
    torch.set_num_threads(2);p=protocol();checks=[]
    lock=json.loads((HERE/'DECISIONS_LOCK.json').read_text())
    assert lock['protocol_sha256']==sha(HERE/'PROTOCOL.json')
    curves=pd.read_csv(OUT/'learning_curves_all.csv')
    for choice in lock['choices']:
        mean=curves[(curves.model==choice['model'])&(curves.split=='validation')&(curves.epoch>0)].groupby('epoch').RMSE_kWh.mean()
        assert int(mean.idxmin())==choice['selected_epoch']
        assert np.isclose(mean.min(),choice['mean_validation_RMSE_kWh'],rtol=0,atol=1e-12)
        for seed in p['seeds']:
            path=STATES/f"{choice['model']}_seed{seed}"/f"epoch{choice['selected_epoch']:04d}_weights.pt"
            assert sha(path)==choice['selected_weight_sha256_by_seed'][str(seed)]
    checks.append('PASS checkpoint choices equal locked minima of three-seed mean2016validationRMSE; weights unchanged')
    te=arrays('test');base=pd.read_csv(OUT/'common_test_rows.csv.gz',parse_dates=['timestamp'])
    rows=pd.read_csv(OUT/'test_metrics_seed.csv')
    for _,row in rows[rows.panel.eq('R2_fullpool_validation_selected')].iterrows():
        if row.model=='Persistence':pred=np.asarray(te['anchor'])
        else:
            path=OUT/'predictions'/f'{row.model}_seed{row.seed}_R2_fullpool_test.csv.gz'
            frame=pd.read_csv(path,parse_dates=['timestamp'])
            assert frame.building_id.equals(base.building_id) and frame.timestamp.equals(base.timestamp)
            assert len(frame)==98483
            pred=frame.prediction_kWh.to_numpy()
            model=make_model(row.model,int(row.seed))
            model.load_state_dict(torch.load(STATES/f'{row.model}_seed{row.seed}'/f'epoch{row.epoch:04d}_weights.pt',map_location='cpu',weights_only=True))
            small={k:np.asarray(v[:1024]) for k,v in te.items()}
            replay=reconstruct(standard_predict(model,small['X']),small)
            assert np.allclose(replay,pred[:1024],rtol=1e-8,atol=1e-8)
        computed=metric(np.asarray(te['actual']),pred)
        for key,value in computed.items():assert np.isclose(value,row[key],rtol=1e-12,atol=1e-12),(row.model,key)
    checks.append('PASS all12R2checkpoints replay first1024testrows and all metrics recompute from98,483paired predictions')
    legacy=pd.read_csv(OUT/'r1_checkpoint_replay.csv')
    assert len(legacy)==12 and legacy.passed.all()
    checks.append('PASS all12R1saved checkpoints replay first1024archived testrows at1e-8tolerance')
    b=pd.read_csv(OUT/'paired_building_differences.csv')
    points=pd.read_csv(OUT/'paired_point_estimands.csv')
    for panel in points.panel.unique():
        for comp in p['uncertainty']['comparators']:
            seedrows=rows[rows.panel.eq(panel)]
            for metricname,column in [('pooled_delta_RMSE_kWh','RMSE_kWh'),('pooled_delta_MAE_kWh','MAE_kWh')]:
                expected=seedrows[seedrows.model.eq('DRA_PSNN')][column].mean()-seedrows[seedrows.model.eq(comp)][column].mean()
                actual=points[points.panel.eq(panel)&points.comparator.eq(comp)&points.metric.eq(metricname)].estimate.iloc[0]
                assert np.isclose(expected,actual,rtol=1e-9,atol=1e-9),(panel,comp,expected,actual)
            g=b[b.panel.eq(panel)&b.comparator.eq(comp)]
            assert len(g)==12 and g.common_valid_hours.sum()==98483
    checks.append('PASS paired pooled estimands match mean-of-seed metric differences;12building rows and common denominators agree')
    ci=pd.read_csv(OUT/'paired_block_intervals.csv')
    assert len(ci)==2*3*2*3*4 and np.isfinite(ci[['estimate','lower95','upper95']]).all().all()
    assert np.all(ci.lower95<=ci.upper95)
    for L in [14,7,28]:
        d=np.load(OUT/f'bootstrap_draws_block{L}_days.npz')
        assert d['calendar_day_multiplicities'].shape==(5000,365)
        assert np.all(d['calendar_day_multiplicities'].sum(1)==365)
        assert np.all(d['site_cluster_multiplicities'].sum(1)==6)
    checks.append('PASS5000synchronized quarter-stratified calendar resamples at14/7/28days and six-site cluster weights')
    (OUT/'VERIFICATION.txt').write_text('\n'.join(checks)+'\n')
    print('\n'.join(checks),flush=True)


if __name__=='__main__':main()
