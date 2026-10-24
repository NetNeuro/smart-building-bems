"""Shared audited I/O for the second-revision measured-meter experiment."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
CACHE = HERE / 'cache'
OUT = HERE / 'results'
STATES = HERE / 'states'
for folder in [SOURCE,CACHE,OUT,STATES]: folder.mkdir(exist_ok=True,parents=True)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


def atomic_json(value,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def safe_csv(frame,path):
    path=Path(path)
    temp=path.with_suffix('.partial.gz' if path.suffix=='.gz' else path.suffix+'.partial')
    frame.to_csv(temp,index=False)
    if path.suffix=='.gz':
        with gzip.open(temp,'rb') as f:
            while f.read(1024*1024): pass
    temp.replace(path)


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def protocol():
    p=HERE/'PROTOCOL.json'
    expected=(HERE/'PROTOCOL.sha256').read_text().split()[0]
    assert sha(p)==expected,'Frozen scientific protocol changed'
    value=json.loads(p.read_text())
    assert sha(SOURCE/'SOURCE_LOCK.json')==value['source_lock_sha256'],'Frozen source lock changed'
    return value


def source_root():
    lock=json.loads((SOURCE/'SOURCE_LOCK.json').read_text())
    candidates=[]
    if os.environ.get('JER_R1_RESEARCH'): candidates.append(Path(os.environ['JER_R1_RESEARCH']))
    candidates.extend([HERE.parents[1]/'research',
                       HERE.parents[1]/'sources/r1_package/JER_S105_Revision_20261012/research',
                       Path(lock['source_research_during_execution'])])
    for p in candidates:
        if (p/'revision/external/raw/bdg2_electricity.csv').exists():return p
    raise FileNotFoundError('Set JER_R1_RESEARCH to the supplied R1 research package root')


def arrays(split):
    d=CACHE/split
    return {p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in d.glob('*.npy')}


def metric(y,p):
    e=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    rmse=float(np.sqrt(np.mean(e*e)))
    den=float(np.sum((np.asarray(y)-np.mean(y))**2))
    return {'RMSE_kWh':rmse,'MAE_kWh':float(np.mean(np.abs(e))),
            'R2':float(1-np.sum(e*e)/den),'CVRMSE_pct':float(100*rmse/np.mean(y))}


def model_class_module():
    lock=json.loads((SOURCE/'SOURCE_LOCK.json').read_text())
    assert sha(SOURCE/'exact_architecture.py')==lock['files']['exact_architecture.py']['sha256']
    return load_module(SOURCE/'exact_architecture.py','r1_exact_architecture')


def make_model(name,seed):
    import torch
    torch.manual_seed(seed);np.random.seed(seed)
    a=model_class_module();p=protocol();h=p['hyperparameters'];nf=len(p['features'])
    if name=='DRA_PSNN':return a.Proposed(nf,h['channels'],h['dropout'],h['beta'],h['theta'],h['tau'],h['kappa'])
    if name=='DRA_GRU':return a.DRAGRU(nf,ch=20,drop=.08,h=20)
    if name=='DRA_LSTM':return a.DRALSTM(nf,ch=20,drop=.08,h=20)
    if name=='Transformer':return a.Transformer(nf,d=24,drop=.08)
    raise ValueError(name)


def standard_predict(model,X,batch=1024):
    import torch
    model.eval();out=[]
    with torch.no_grad():
        for j in range(0,len(X),batch):out.append(model(torch.tensor(np.asarray(X[j:j+batch]))).numpy())
    return np.concatenate(out)


def reconstruct(q,S):
    z=np.load(CACHE/'normalization.npz')
    return np.asarray(S['anchor'])+(q*float(z['delta_sd'])+float(z['delta_mean']))*np.sqrt(np.asarray(S['area']))
