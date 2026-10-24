"""Wait for validation-only training completion, then finalize in locked order."""
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
start=time.monotonic()
while not (HERE/'DECISIONS_LOCK.json').exists():
    if time.monotonic()-start>4*3600:raise TimeoutError('Validation decisions not locked within four hours')
    time.sleep(5)
for script in ['score_test.py','paired_uncertainty.py','verify_final.py','summarize_report.py']:
    print('FINALIZATION PHASE',script,flush=True)
    subprocess.run([sys.executable,str(HERE/script)],check=True)
print('FINALIZATION COMPLETE',flush=True)
