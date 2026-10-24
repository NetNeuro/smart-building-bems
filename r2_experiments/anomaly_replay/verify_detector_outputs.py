from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix
P=argparse.ArgumentParser();P.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results');args=P.parse_args();D=args.results.resolve()
sel=pd.read_csv(D/'selection_manifest.csv');metrics=pd.read_csv(D/'all_detector_metrics.csv');checks=[]
freeze=json.loads((D/'selection_freeze.json').read_text())
assert hashlib.sha256((D/'selection_manifest.csv').read_bytes()).hexdigest()==freeze['selection_manifest_sha256']
for _,r in sel.iterrows():
 obj=joblib.load(D/r.model_file);model=obj['model'] if isinstance(obj,dict) else obj;scaler=obj.get('scaler') if isinstance(obj,dict) else None
 for split in ['seen_test','unseen_full','unseen_temporal']:
  z=np.load(D/'inputs'/r.pipeline/f'features_{split}.npz',allow_pickle=True);X=z['X'];y=z['y'];x=scaler.transform(X) if scaler is not None else X;s=model.predict_proba(x)[:,1]
  f=pd.read_csv(D/'predictions'/r.condition/r.pipeline/r.method/f'scores_{split}.csv');p=s>=r.threshold
  assert np.allclose(s,f.score,atol=1e-14,rtol=1e-13)
  assert np.array_equal(p.astype(int),f.predicted)
  expected=metrics[(metrics.condition==r.condition)&(metrics.pipeline==r.pipeline)&(metrics.method==r.method)&(metrics.split==split)].iloc[0]
  vals=[precision_score(y,p,zero_division=0),recall_score(y,p,zero_division=0),f1_score(y,p,zero_division=0),average_precision_score(y,s),roc_auc_score(y,s)]
  assert np.allclose(vals,expected[['Precision','Recall','F1','AP','AUROC']].to_numpy(float),rtol=0,atol=1e-14)
  assert np.array_equal(confusion_matrix(y,p,labels=[0,1]).ravel(),expected[['TN','FP','FN','TP']].to_numpy(int))
  checks.append({'condition':r.condition,'pipeline':r.pipeline,'method':r.method,'split':split,'serialized_scores_and_metrics_verified':True})
for split in ['validation','seen_test','unseen_full','unseen_temporal']:
 z12=np.load(D/'inputs/PSNN_12'/f'features_{split}.npz',allow_pickle=True)['X'];z11=np.load(D/'inputs/PSNN_11'/f'features_{split}.npz',allow_pickle=True)['X']
 assert np.array_equal(np.delete(z12,7,axis=1),z11)
 for pipe in ['Persistence_11','GRU_11']:
  z=np.load(D/'inputs'/pipe/f'features_{split}.npz',allow_pickle=True)['X']
  # Only relative residual and its square may depend on the forecaster.
  assert np.array_equal(z[:,[2,3,4,6,7,8,9,10]],z11[:,[2,3,4,6,7,8,9,10]])
pd.DataFrame(checks).to_csv(D/'verification_report.csv',index=False)
print(f'PASS: {len(checks)} saved detector/split combinations, frozen manifest, and common-feature alignment.')
