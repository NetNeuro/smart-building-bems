"""Full-pool learning curves. This program never loads the test cache."""
import concurrent.futures as futures
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import HERE,CACHE,OUT,STATES,protocol,sha,atomic_json,safe_csv,arrays,make_model,standard_predict,reconstruct,metric


def save_torch(value,path):
    temp=path.with_suffix(path.suffix+'.partial');torch.save(value,temp);temp.replace(path)


def diagnostic(model,S):
    q=standard_predict(model,S['X'])
    result=metric(S['actual'],reconstruct(q,S))
    result['normalized_MSE']=float(np.mean((q.astype(float)-np.asarray(S['y'],dtype=float))**2))
    return result


def train_one(name,seed,cap):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    p=protocol();tr=arrays('train');va=arrays('validation')
    idx=np.load(CACHE/'diagnostic_training_indices.npy')
    diag={k:np.asarray(v[idx]) for k,v in tr.items()}
    # Only training and validation arrays are available to the fitting routine.
    assert set([tr['ts'].max().astype('datetime64[D]').item().year,
                va['ts'].max().astype('datetime64[D]').item().year])=={2016}
    xt=torch.tensor(np.asarray(tr['X']));yt=torch.tensor(np.asarray(tr['y']))
    folder=STATES/f'{name}_seed{seed}';folder.mkdir(exist_ok=True)
    curve=folder/'learning_curve.csv';landmarks=folder/'update_landmarks.csv'
    model=make_model(name,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=p['hyperparameters']['lr'],weight_decay=1e-4)
    epoch=0;updates=0;cpu=0.;wall=0.;examples=0
    rows=[];marks=[]
    latest=folder/'latest_resume.pt'
    if latest.exists():
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
        assert checkpoint['protocol_sha256']==sha(HERE/'PROTOCOL.json')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        epoch=checkpoint['epoch'];updates=checkpoint['updates'];examples=checkpoint['examples_seen']
        cpu=checkpoint['train_cpu_seconds'];wall=checkpoint['train_wall_seconds']
        torch.set_rng_state(checkpoint['torch_rng']);np.random.set_state(checkpoint['numpy_rng'])
        rows=pd.read_csv(curve).to_dict('records') if curve.exists() else []
        marks=pd.read_csv(landmarks).to_dict('records') if landmarks.exists() else []
        # An interrupted final CSV write cannot introduce records after the saved state.
        rows=[r for r in rows if int(r['epoch'])<=epoch]
        marks=[r for r in marks if int(r['updates'])<=updates]
    else:
        torch.manual_seed(seed);np.random.seed(seed)
        for split,S in [('diagnostic_training',diag),('validation',va)]:
            rows.append({'model':name,'seed':seed,'epoch':0,'updates':0,'examples_seen':0,
                         'split':split,'train_cpu_seconds':0.,'train_wall_seconds':0.,
                         'optimizer_epoch_mean_MSE':None,**diagnostic(model,S)})
        safe_csv(pd.DataFrame(rows),curve)
    for e in range(epoch+1,cap+1):
        model.train();perm=torch.randperm(len(xt));loss_sum=0.;n_sum=0
        for j in range(0,len(xt),p['batch_size']):
            z=perm[j:j+p['batch_size']]
            c0=time.process_time();w0=time.perf_counter()
            pred=model(xt[z]);loss=((pred-yt[z])**2).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            cpu+=time.process_time()-c0;wall+=time.perf_counter()-w0
            updates+=1;examples+=len(z);loss_sum+=float(loss.detach())*len(z);n_sum+=len(z)
            if updates in p['update_landmarks']:
                v=diagnostic(model,va)
                marks.append({'model':name,'seed':seed,'updates':updates,'epoch_in_progress':e,
                              'examples_seen':examples,'train_cpu_seconds':cpu,'train_wall_seconds':wall,**v})
                safe_csv(pd.DataFrame(marks),landmarks);model.train()
        for split,S in [('diagnostic_training',diag),('validation',va)]:
            rows.append({'model':name,'seed':seed,'epoch':e,'updates':updates,'examples_seen':examples,
                         'split':split,'train_cpu_seconds':cpu,'train_wall_seconds':wall,
                         'optimizer_epoch_mean_MSE':loss_sum/n_sum,**diagnostic(model,S)})
        # Every epoch's weight file permits the later architecture-level checkpoint choice.
        save_torch(model.state_dict(),folder/f'epoch{e:04d}_weights.pt')
        safe_csv(pd.DataFrame(rows),curve)
        save_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':e,'updates':updates,
                    'examples_seen':examples,'train_cpu_seconds':cpu,'train_wall_seconds':wall,
                    'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                    'protocol_sha256':sha(HERE/'PROTOCOL.json')},latest)
        if e%5==0:
            r=[x for x in rows if x['split']=='validation'][-1]
            print(name,seed,'epoch',e,'validation_RMSE',round(r['RMSE_kWh'],6),
                  'updates',updates,'train_cpu_s',round(cpu,1),flush=True)
    return {'model':name,'seed':seed,'epoch':cap}


def train_architecture(name,cap):
    return [train_one(name,s,cap) for s in protocol()['seeds']]


def main():
    p=protocol();assert (CACHE/'DATA_LOCK.json').exists()
    assert not (HERE/'TEST_SCORED.json').exists(),'New training after test scoring is prohibited by the protocol'
    if (HERE/'DECISIONS_LOCK.json').exists():
        print('Decisions already locked; no training changes',flush=True);return
    trajectory=[]
    if (OUT/'budget_decision_trace.json').exists():
        trajectory=json.loads((OUT/'budget_decision_trace.json').read_text())
    terminal=None
    for cap in p['epoch_caps']:
        print('BEGIN VALIDATION-ONLY STAGE',cap,flush=True)
        with futures.ProcessPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(train_architecture,name,cap):name for name in p['models']}
            for job in futures.as_completed(jobs):
                print('STAGE FINISHED',cap,jobs[job],job.result(),flush=True)
        allrows=pd.concat([pd.read_csv(STATES/f'{m}_seed{s}/learning_curve.csv')
                           for m in p['models'] for s in p['seeds']],ignore_index=True)
        safe_csv(allrows,OUT/'learning_curves_all.csv')
        allmarks=pd.concat([pd.read_csv(STATES/f'{m}_seed{s}/update_landmarks.csv')
                            for m in p['models'] for s in p['seeds']],ignore_index=True)
        safe_csv(allmarks,OUT/'equal_update_validation.csv')
        checks=[]
        for name in p['models']:
            mean=allrows[(allrows.model==name)&(allrows.split=='validation')&(allrows.epoch>0)&(allrows.epoch<=cap)].groupby('epoch').RMSE_kWh.mean()
            earlier=float(mean.loc[:cap-5].min());tail=float(mean.loc[cap-4:cap].min())
            gain=max(0.,(earlier-tail)/earlier);best=int(mean.idxmin())
            needs=gain>=.005 or best>=cap-1
            checks.append({'model':name,'cap_epoch':cap,'best_mean_validation_epoch':best,
                           'best_mean_validation_RMSE_kWh':float(mean.min()),'relative_last5_best_gain':gain,
                           'best_in_last2_epochs':best>=cap-1,'criterion_requests_extension':bool(needs)})
        extend=any(x['criterion_requests_extension'] for x in checks) and cap<p['epoch_caps'][-1]
        trajectory.append({'cap_epoch':cap,'models':checks,'extend_all':extend})
        atomic_json(trajectory,OUT/'budget_decision_trace.json')
        print('VALIDATION DECISION',json.dumps(trajectory[-1]),flush=True)
        if not extend:terminal=cap;break
    assert terminal is not None
    choices=[]
    for name in p['models']:
        mean=allrows[(allrows.model==name)&(allrows.split=='validation')&(allrows.epoch>0)&(allrows.epoch<=terminal)].groupby('epoch').RMSE_kWh.mean()
        epoch=int(mean.idxmin())
        files={str(seed):sha(STATES/f'{name}_seed{seed}/epoch{epoch:04d}_weights.pt') for seed in p['seeds']}
        choices.append({'model':name,'selected_epoch':epoch,'mean_validation_RMSE_kWh':float(mean.min()),
                        'updates_at_selected_epoch':int(allrows[(allrows.model==name)&(allrows.epoch==epoch)].updates.iloc[0]),
                        'selected_weight_sha256_by_seed':files})
    lock={'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'protocol_sha256':sha(HERE/'PROTOCOL.json'),
          'data_lock_sha256':sha(CACHE/'DATA_LOCK.json'),'learning_curves_sha256':sha(OUT/'learning_curves_all.csv'),
          'budget_trace_sha256':sha(OUT/'budget_decision_trace.json'),'terminal_epoch_cap':terminal,
          'choices':choices,'no_new_2017_scores_used_for_decisions':True,
          'still_budget_sensitive_at_maximum':bool(terminal==40 and any(x['criterion_requests_extension'] for x in checks)),
          'prior_2017_status':p['prior_holdout_status']}
    atomic_json(lock,HERE/'DECISIONS_LOCK.json')
    print('DECISIONS LOCKED BEFORE NEW TEST SCORING',json.dumps(lock),flush=True)


if __name__=='__main__':main()
