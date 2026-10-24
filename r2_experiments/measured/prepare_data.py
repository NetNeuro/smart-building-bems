"""Reproduce exact R1 row bindings into separately accessible train/val/test caches."""
import json
import numpy as np
import pandas as pd
from common import HERE,SOURCE,CACHE,OUT,source_root,protocol,load_module,sha,atomic_json


def main():
    p=protocol();root=source_root();old=root/'revision/external'
    for item in json.loads((SOURCE/'bdg2_source_manifest.json').read_text()):
        assert sha(old/'raw'/item['local_file'])==item['sha256'],item['local_file']
    m=load_module(SOURCE/'r1_run_external_bdg2.py','r1_exact_data_preparation')
    m.HERE=CACHE;m.OUT=OUT;m.RAW=old/'raw'
    parts,mu,sd,buildings,sites=m.prepare()
    assert buildings==p['buildings'] and sites==p['sites']
    assert [len(parts[x]['X']) for x in ['train','validation','test']]==[84928,17296,98483]
    original=np.load(old/'normalization.npz');current=np.load(CACHE/'normalization.npz')
    for key in original.files:assert np.array_equal(original[key],current[key]),key
    bindings=[];manifest={}
    for split,S in parts.items():
        d=CACHE/split;d.mkdir(exist_ok=True)
        for key,value in S.items():
            destkey={'target':'actual','building_id':'bid','site_id':'site','timestamp':'ts'}.get(key,key)
            if key=='timestamp':value=np.asarray(value,dtype='datetime64[ns]')
            np.save(d/(destkey+'.npy'),np.asarray(value),allow_pickle=False)
            manifest[str((d/(destkey+'.npy')).relative_to(HERE))]=sha(d/(destkey+'.npy'))
        for bid in buildings:
            z=S['building_id']==bid
            bindings.append({'building_id':bid,'site_id':bid.split('_')[0],'split':split,'valid_sequences':int(z.sum()),
                             'first_target':str(S['timestamp'][z].min()),'last_target':str(S['timestamp'][z].max())})
    # This verifies row identity only; no 2017 forecast losses are computed.
    legacy=pd.read_csv(old/'results/test_rows.csv.gz',parse_dates=['timestamp'])
    te=parts['test']
    assert np.array_equal(te['building_id'],legacy.building_id.to_numpy())
    assert np.array_equal(np.asarray(te['timestamp'],dtype='datetime64[ns]'),legacy.timestamp.to_numpy())
    assert np.array_equal(te['target'],legacy.actual_kWh.to_numpy())
    assert np.array_equal(te['anchor'],legacy.persistence_kWh.to_numpy())
    pd.DataFrame(bindings).to_csv(OUT/'building_split_coverage.csv',index=False)
    idx=np.linspace(0,len(parts['train']['X'])-1,6800,dtype=int)
    np.save(CACHE/'diagnostic_training_indices.npy',idx)
    atomic_json({'protocol_sha256':sha(HERE/'PROTOCOL.json'),'source_r1_row_identity':True,
                 'normalization_bitwise_identical':True,'parts':manifest,
                 'rows':{x:len(s['X']) for x,s in parts.items()},
                 'note':'Prepared2017 target cache is not loaded by training/checkpoint-selection code.'},CACHE/'DATA_LOCK.json')
    print('EXACT DATA PREPARATION PASS', {x:len(s['X']) for x,s in parts.items()},flush=True)


if __name__=='__main__':main()
