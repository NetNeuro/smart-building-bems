"""Portable R2 anomaly/pipeline/replay regeneration against a frozen R1 research directory."""
from pathlib import Path
import argparse,subprocess,sys,json
P=argparse.ArgumentParser();P.add_argument('--source',required=True,type=Path);P.add_argument('--skip-cost',action='store_true');a=P.parse_args();H=Path(__file__).resolve().parent;SRC=a.source.resolve();D=H/'results'
def run(script,*args):subprocess.run([sys.executable,'-B',str(H/script),*map(str,args)],check=True)
run(Path('calibration_audit/prepare_audit.py'),SRC)
run(Path('run_detector_experiments.py'),'--source',SRC)
run(Path('replay/run_pipeline_replays.py'),'--manifest',D/'primary_replay_jobs.json','--raw-data',SRC/'data/synthetic_benchmark.csv','--out',H/'replay/pipeline_comparison','--detector-metrics',D/'primary_pipeline_prediction_detection.csv')
run(Path('verify_detector_outputs.py'),'--results',D)
if not a.skip_cost:
 names={'canonical_psnn12':'PSNN_12','psnn_common11':'PSNN_11','persistence11':'Persistence_11','standalone_gru11':'GRU_11'}
 heads={k:str(D/'calibration/cut60_causal_unpurged'/f'{v}_GB.joblib') for k,v in names.items()};hp=H/'calibration_audit/pipeline_heads.json';hp.write_text(json.dumps(heads,indent=2))
 run(Path('calibration_audit/benchmark_pipeline_cost.py'),SRC,'--heads-json',hp)
run(Path('make_summaries.py'))
run(Path('make_requested_supplement_tables.py'))
