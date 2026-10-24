"""Freeze the R2 rules and exact R1 sources before new training/test scoring."""
import argparse
import datetime as dt
import json
import platform
import shutil

import numpy as np
import pandas as pd
import torch
from common import HERE,SOURCE,sha,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-research',required=True)
    arg=parser.parse_args()
    from pathlib import Path
    root=Path(arg.source_research).resolve();ext=root/'revision/external'
    assert not (HERE/'PROTOCOL.json').exists(),'Refusing to overwrite frozen protocol'
    prior=json.loads((ext/'protocol.json').read_text())
    files={}
    for src,dest in [(ext/'run_external_bdg2.py','r1_run_external_bdg2.py'),
                     (ext/'exact_architecture.py','exact_architecture.py'),
                     (root/'scripts/run_canonical_s105.py','r1_run_canonical_s105.py'),
                     (ext/'protocol.json','r1_protocol.json'),
                     (ext/'raw/source_manifest.json','bdg2_source_manifest.json'),
                     (ext/'raw/BDG2_LICENSE.txt','BDG2_LICENSE.txt')]:
        shutil.copyfile(src,SOURCE/dest)
        files[dest]={'source_relative_path':str(src.relative_to(root)),'sha256':sha(SOURCE/dest)}
    assert files['r1_run_canonical_s105.py']['sha256']==prior['canonical_source_sha256']
    for item in json.loads((SOURCE/'bdg2_source_manifest.json').read_text()):
        assert sha(ext/'raw'/item['local_file'])==item['sha256'],item
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    lock={'frozen_utc':now,'source_research_during_execution':str(root),'files':files,
          'raw_source_hashes_verified':True,'architecture':'Unmodified R1 exact class definitions; no reconstructed model'}
    atomic_json(lock,SOURCE/'SOURCE_LOCK.json')
    p={'protocol_name':'R2 measured temporal forecasting budget and paired-uncertainty audit',
       'frozen_utc':now,'source_lock_sha256':sha(SOURCE/'SOURCE_LOCK.json'),
       'prior_holdout_status':'2017 results were already reported in R1; this is a reused-holdout robustness audit, not pristine external confirmation.',
       'dataset_source':prior['repository_commit'],'dataset_doi':prior['dataset_doi'],
       'buildings':prior['selected_buildings'],'sites':prior['sites'],
       'meter_selection':prior['selection'],'features':prior['features'],
       'hyperparameters':prior['hyperparameters'],'models':['DRA_PSNN','DRA_GRU','DRA_LSTM','Transformer'],
       'seeds':[11,23,37],'window_hours':24,'horizon_hours':1,
       'train':'2016-01-01 through 2016-10-31','validation':'2016-11-01 through 2016-12-31','test':'full calendar 2017',
       'data':'Exact R1 preprocessing and valid-window bindings. No changed buildings, cleaning, targets, imputation, normalizers or historical inputs.',
       'training_pool':'All eligible Jan-Oct2016 sequences, expected84928; verified after exact preparation.',
       'diagnostic_train_pool':'The original6800 evenly spaced eligible training sequence indices; eval mode only for learning curves.',
       'batch_size':256,'epochs_initial':10,'epoch_caps':[10,20,40],
       'optimizer':'AdamW, original learning rate, weight_decay=0.0001, gradient_clip=5; no schedule or new hyperparameter search.',
       'construction_and_training_seed_reset':True,'threads':2,
       'validation_curve':'Every epoch: fixed diagnostic training and entire validation normalized objective MSE, meter RMSE and MAE; no 2017 losses in training script.',
       'extension_rule':{'mean_curve':'Arithmetic mean validation RMSE of seeds11,23,37, per architecture.',
                         'gain':'max(0,(min(m[1:E-5])-min(m[E-4:E]))/min(m[1:E-5]))',
                         'extend_if':'Any model gains at least0.005 in last5epochs, or earliest global minimum lies in last2epochs. Extend ALL models/seeds to next cap, preserving optimizer/RNG states.',
                         'maximum':40,'at_cap':'Record unresolved budget sensitivity and retain validation-selected checkpoint without a convergence claim.'},
       'checkpoint_rule':'For each architecture choose earliest epoch attaining minimal mean validation RMSE across all3seeds. Use the same epoch for its3seeds. Never drop a seed or select using test.',
       'update_landmarks':[135,540,1350],
       'update_landmark_scope':'Validation-only equal-update diagnostic; not equal-compute or pure sample-size causal attribution.',
       'final_test_gate':'Create DECISIONS_LOCK.json with all budgets/checkpoints/validation hashes before score_test.py can read predictions or score2017.',
       'uncertainty':{'comparators':['Persistence','DRA_GRU','DRA_LSTM'],
                      'difference_direction':'DRA_PSNN minus comparator; negative favorsPSNN.',
                      'point_metrics':'Mean of three seed-specific pooled metrics, and paired per-building differences. Never substitute ensemble error.',
                      'primary_interval':'5000 paired moving-calendar-block resamples;14days; within calendar quarter; synchronized time draws across buildings/models/seeds; fixed12building panel.',
                      'site_time_sensitivity':'Resample six site clusters with replacement, retaining both buildings per chosen site; same synchronized quarter time blocks.',
                      'block_days':[14,7,28],'replicates':5000,'bootstrap_seed':20261022,
                      'seed_handling':'Fitted seed models held fixed; initialization SD reported separately, not treated as building/time uncertainty.',
                      'missingness':'Resample complete calendar days including original common-row masks; recompute denominators. No hour-iid resampling or compression across gaps.',
                      'estimands':['pooled_delta_RMSE_kWh','pooled_delta_MAE_kWh','equal_building_delta_RMSE_kWh','equal_building_delta_CVRMSE_percentage_points'],
                      'inference_scope':'Pointwise95% exploratory percentile intervals conditional on selected data, fitted models and quarter-stratified2017 time process. Six deterministic sites do not justify population/site-transfer inference.',
                      'multiplicity':'No confirmatory ranking or superiority tests; comparators/secondary metrics exploratory; no economic or formal equivalence claim.'},
       'legacy_preservation':'R1 five-epoch6800-window outputs remain separate fixed-budget reference; paired uncertainty is reported separately for legacy and validation-selected extended runs.',
       'software':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    atomic_json(p,HERE/'PROTOCOL.json')
    (HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'  PROTOCOL.json\n')
    print('FROZEN',sha(HERE/'PROTOCOL.json'),flush=True)


if __name__=='__main__':main()
