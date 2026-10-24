"""Public numerical audit wrapper; journal document checks are separate."""
from pathlib import Path
import contextlib,io,json,os,runpy,sys,traceback
import pandas as pd

SOURCE=Path('sources/r1_package/JER_S105_Revision_20261012').resolve()
OUT=Path('r2_experiments/provenance/r1_verifiers').resolve()
OUT.mkdir(parents=True,exist_ok=True)
orig_write=Path.write_text
orig_csv=pd.DataFrame.to_csv
CURRENT=''
redirects=[]
def destination(path):
    p=Path(path)
    resolved=p.resolve()
    if resolved.is_relative_to(SOURCE):
        target=OUT/CURRENT/resolved.relative_to(SOURCE)
        target.parent.mkdir(parents=True,exist_ok=True)
        redirects.append({'source_path':str(resolved),'verification_output':str(target)})
        return target
    return p
def safe_write(self,*args,**kwargs):
    return orig_write(destination(self),*args,**kwargs)
def safe_csv(self,path_or_buf=None,*args,**kwargs):
    if isinstance(path_or_buf,(str,Path)):path_or_buf=destination(path_or_buf)
    return orig_csv(self,path_or_buf,*args,**kwargs)
Path.write_text=safe_write
pd.DataFrame.to_csv=safe_csv
jobs=[('stability','research/revision/stability/verify_stability.py',[]),
      ('anomaly','research/revision/anomaly/verify_anomaly_outputs.py',[str(SOURCE/'research/revision/anomaly/main_chronological')]),
      ('control','research/revision/control/verify_control.py',[])]
results=[]
for name,rel,args in jobs:
    CURRENT=name
    capture=io.StringIO()
    sys.argv=[str(SOURCE/rel),*args]
    try:
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            runpy.run_path(str(SOURCE/rel),run_name='__main__')
        status='PASS';error=None
    except BaseException:
        status='FAIL';error=traceback.format_exc()
    log=OUT/f'{name}.log'
    orig_write(log,capture.getvalue()+(error or ''))
    results.append({'component':name,'status':status,'log':str(log),'error':error})
    print(name,status,flush=True)
orig_write(OUT/'summary.json',json.dumps({'results':results,'source_output_writes_redirected':redirects},indent=2))
print(json.dumps(results,indent=2))
