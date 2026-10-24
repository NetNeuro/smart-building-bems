"""Record actual execution hardware/software without changing frozen science rules."""
import json
import os
import platform
from pathlib import Path
import numpy,pandas,scipy,sklearn,torch,matplotlib
from common import HERE,atomic_json


def main():
    cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')
    d={'cpu_model':cpu,'logical_cpu_count':os.cpu_count(),'platform':platform.platform(),
       'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__,
       'pandas':pandas.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
       'matplotlib':matplotlib.__version__,'training_torch_threads':2,
       'OMP_NUM_THREADS_training':2,'MKL_NUM_THREADS_training':2,'OPENBLAS_NUM_THREADS_training':1,
       'training_worker_processes':4,'accelerator':'CPU only',
       'timing_scope':'Per-worker process CPU and wall time around forward/backward/AdamW steps; excludes validation, serialization and telemetry I/O; parallel workers may affect wall time.'}
    atomic_json(d,HERE/'ENVIRONMENT.json');print(json.dumps(d,indent=2))


if __name__=='__main__':main()
