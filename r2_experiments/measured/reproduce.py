"""Reproduce into a new directory without overwriting supplied audit evidence."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-research',required=True);p.add_argument('--output-root',required=True)
    args=p.parse_args();source=Path(args.source_research).resolve();out=Path(args.output_root).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output directory must be new or empty; supplied evidence is never overwritten')
    out.mkdir(exist_ok=True,parents=True)
    for script in HERE.glob('*.py'):shutil.copy2(script,out/script.name)
    for name in ['PROTOCOL.json','PROTOCOL.sha256']:shutil.copy2(HERE/name,out/name)
    shutil.copytree(HERE/'source',out/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','results'))
    env=os.environ.copy();env.update({'JER_R1_RESEARCH':str(source),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    scripts=['environment.py','prepare_data.py','coverage_counts.py','train_budget.py',
             'score_test.py','paired_uncertainty.py','verify_final.py','summarize_report.py']
    for script in scripts:
        print('REPRODUCTION PHASE',script,flush=True)
        subprocess.run([sys.executable,str(out/script)],check=True,env=env)


if __name__=='__main__':main()
