from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
order=[
    'run_canonical_s105.py',
    'run_6h_all.py',
    'run_stability_10seeds.py',
    'run_anomaly_control.py',
    'make_figures.py',
]
for name in order:
    print(f'\n=== {name} ===', flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts'/name)], cwd=ROOT, check=True)
print('\nAll Round-3 experiments completed.', flush=True)
