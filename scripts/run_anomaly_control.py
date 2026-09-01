from pathlib import Path
import math,json
import numpy as np,pandas as pd
import torch,torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,average_precision_score,roc_auc_score,confusion_matrix

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
DATA=ROOT/'data'/'synthetic_benchmark.csv';F=['temp','humidity','solar','occupancy','log_area','hour_sin','hour_cos','dow_sin','dow_cos','lag1','lag2','lag6','lag12','lag24','roll24'];SEL=['humidity','solar','occupancy','log_area','dow_sin','dow_cos','lag1','lag2','lag24'];idx=np.array([F.index(x) for x in SEL]);L=24;SEEDS=[11,23,37]
CAN={'channels':20,'lr':0.0015287143259818435,'dropout':0.08,'beta':0.9431572529176961,'theta':0.36879562783560627,'tau':0.18,'kappa':0.35}
df=pd.read_csv(DATA,parse_dates=['timestamp']);df['log_area']=np.log1p(df.square_feet);df['hour_sin']=np.sin(2*np.pi*df.hour/24);df['hour_cos']=np.cos(2*np.pi*df.hour/24);df['dow_sin']=np.sin(2*np.pi*df.dow/7);df['dow_cos']=np.cos(2*np.pi*df.dow/7)
seen=df[df.building_id<9].copy();unseen=df[df.building_id>=9].copy();trp=[];vap=[];tep=[]
for b,g in seen.groupby('building_id'):
 g=g.sort_values('timestamp').reset_index(drop=True);n=len(g);a=int(.70*n);c=int(.85*n);trp.append(g.iloc[:a]);vap.append(g.iloc[a:c]);tep.append(g.iloc[c:])
tr=pd.concat(trp,ignore_index=True);va=pd.concat(vap,ignore_index=True);te=pd.concat(tep,ignore_index=True);unseen_temporal=unseen[unseen.timestamp>=te.timestamp.min()].copy();sc=StandardScaler().fit(tr[F]);delta=(tr.clean-tr.lag1)/np.sqrt(tr.square_feet);MU=float(delta.mean());SD=float(delta.std()+1e-9)
def seq(frame):
 X=[];y=[];anchor=[];clean=[];area=[];obs=[];lab=[];bid=[];ts=[];occ=[];temp=[];hum=[];solar=[];lag2=[];lag24=[];atype=[];hvac=[]
 for b,g in frame.groupby('building_id'):
  g=g.sort_values('timestamp').reset_index(drop=True);z=sc.transform(g[F]).astype('float32')
  for i in range(L,len(g)):
   last=float(g.meter.iloc[i-1]);target=float(g.clean.iloc[i]);yy=((target-last)/math.sqrt(float(g.square_feet.iloc[i]))-MU)/SD
   X.append(z[i-L:i]);y.append(yy);anchor.append(last);clean.append(target);area.append(float(g.square_feet.iloc[i]));obs.append(float(g.meter.iloc[i]));lab.append(int(g.anomaly.iloc[i]));bid.append(int(b));ts.append(g.timestamp.iloc[i]);occ.append(float(g.occupancy.iloc[i]));temp.append(float(g.temp.iloc[i]));hum.append(float(g.humidity.iloc[i]));solar.append(float(g.solar.iloc[i]));lag2.append(float(g.lag2.iloc[i]));lag24.append(float(g.lag24.iloc[i]));atype.append(g.atype.iloc[i]);hvac.append(float(g.hvac.iloc[i]))
 return {k:np.asarray(v,np.float32 if k in ['X','y'] else (object if k in ['ts','atype'] else float)) for k,v in locals().items() if k in ['X','y','anchor','clean','area','obs','lab','bid','ts','occ','temp','hum','solar','lag2','lag24','atype','hvac']}
TR=seq(tr);VA=seq(va);TE=seq(te);UN=seq(unseen);UNT=seq(unseen_temporal);ii=np.linspace(0,len(TR['X'])-1,min(6800,len(TR['X'])),dtype=int)
class RAB(nn.Module):
 def __init__(self,ch,drop):
  super().__init__();self.c1=nn.Conv1d(ch,ch,3,padding=1);self.c2=nn.Conv1d(ch,ch,3,padding=1);self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid());self.ta=nn.Sequential(nn.Conv1d(2,1,5,padding=2),nn.Sigmoid());self.dr=nn.Dropout(drop)
 def forward(self,x):h=torch.relu(self.c1(x));h=self.dr(self.c2(h));h=h*self.ca(h);h=h*self.ta(torch.cat([h.mean(1,keepdim=True),h.max(1,keepdim=True).values],1));return torch.relu(x+h)
class HDRAB(nn.Module):
 def __init__(self,ch,drop):super().__init__();self.ds=nn.ModuleList([nn.Conv1d(ch,ch,3,padding=d,dilation=d) for d in [1,2,4]]);self.mix=nn.Conv1d(3*ch,ch,1);self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid());self.dr=nn.Dropout(drop)
 def forward(self,x):h=self.dr(self.mix(torch.cat([torch.relu(c(x)) for c in self.ds],1)));return torch.relu(x+h*self.ca(h))
class DRA(nn.Module):
 def __init__(self,nf):super().__init__();self.i=nn.Conv1d(nf,CAN['channels'],1);self.a=RAB(CAN['channels'],CAN['dropout']);self.b=HDRAB(CAN['channels'],CAN['dropout']);self.f=nn.Conv1d(2*CAN['channels'],CAN['channels'],1)
 def forward(self,x):z=torch.relu(self.i(x.transpose(1,2)));return torch.relu(self.f(torch.cat([self.a(z),self.b(z)],1))).transpose(1,2)
class PSNN(nn.Module):
 def __init__(self):super().__init__();h=max(12,CAN['channels']);self.w=nn.Linear(CAN['channels'],h);self.o=nn.Linear(h,1)
 def forward(self,z,return_rate=False):
  u=torch.zeros(z.shape[0],self.w.out_features,device=z.device);sp=[]
  for t in range(z.shape[1]):u=CAN['beta']*u+self.w(z[:,t]);p=torch.sigmoid((u-CAN['theta'])/CAN['tau']);u=u*(1-CAN['kappa']*p);sp.append(p)
  rate=torch.stack(sp,1).mean(1);out=self.o(rate).squeeze(-1);return (out,rate.mean(1)) if return_rate else out
class Prop(nn.Module):
 def __init__(self):super().__init__();self.d=DRA(len(idx));self.p=PSNN()
 def forward(self,x,return_rate=False):return self.p(self.d(x),return_rate)
def train(seed):
 torch.manual_seed(seed);np.random.seed(seed);m=Prop();opt=torch.optim.AdamW(m.parameters(),lr=CAN['lr'],weight_decay=1e-4);xt=torch.tensor(TR['X'][ii][:,:,idx]);yt=torch.tensor(TR['y'][ii]);m.train()
 for _ in range(5):
  perm=torch.randperm(len(xt))
  for j in range(0,len(xt),256):
   q=perm[j:j+256];p=m(xt[q]);loss=((p-yt[q])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 return m
def pred(m,S):
 m.eval();o=[];sr=[]
 with torch.no_grad():
  for j in range(0,len(S['X']),1024):
   a,b=m(torch.tensor(S['X'][j:j+1024,:,idx]),True);o.append(a.numpy());sr.append(b.numpy())
 rr=np.concatenate(o);sp=np.concatenate(sr);return S['anchor']+(rr*SD+MU)*np.sqrt(S['area']),sp
mods=[train(s) for s in SEEDS]
def ens(S):
 ps=[];ss=[]
 for m in mods:
  p,s=pred(m,S);ps.append(p);ss.append(s)
 return np.mean(ps,0),np.mean(ss,0)
PV,SV=ens(VA);PTE,STE=ens(TE);PUN,SUN=ens(UN);PUNT,SUNT=ens(UNT)
def feat(S,p,s):
 scale=np.maximum(np.abs(p),10.0);signed=(S['obs']-p)/scale;absr=np.abs(signed);jump=(S['obs']-S['anchor'])/np.maximum(np.abs(S['anchor']),10.0);dev2=(S['obs']-S['lag2'])/np.maximum(np.abs(S['lag2']),10.0);dev24=(S['obs']-S['lag24'])/np.maximum(np.abs(S['lag24']),10.0)
 return np.c_[absr,signed,jump,dev2,dev24,absr**2,jump**2,s,S['occ'],S['temp'],S['hum'],S['solar']]
ZV=feat(VA,PV,SV);ZTE=feat(TE,PTE,STE);ZUN=feat(UN,PUN,SUN);ZUNT=feat(UNT,PUNT,SUNT);cut=int(.60*len(ZV));clf=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=.05,random_state=11).fit(ZV[:cut],VA['lab'][:cut]);pt=clf.predict_proba(ZV[cut:])[:,1];yt=VA['lab'][cut:]
best=(-1,.5)
for th in np.linspace(.03,.60,230):
 z=(pt>=th).astype(int);_,_,f1,_=precision_recall_fscore_support(yt,z,average='binary',zero_division=0)
 if f1>best[0]:best=(f1,float(th))
TH=best[1];print('threshold',TH,'valf1',best[0],flush=True)
json.dump({'classifier':'GradientBoostingClassifier','n_estimators':150,'max_depth':3,'learning_rate':0.05,'calibration_partition':'first 60% of validation','threshold_partition':'last 40% of validation','threshold':TH,'features':['abs_rel_residual','signed_rel_residual','jump_from_meter_anchor','dev_lag2','dev_lag24','abs_residual_sq','jump_sq','mean_PSNN_spike_probability','occupancy','temperature','humidity','solar']},open(OUT/'anomaly_head_config.json','w'),indent=2)
def ev(name,S,Z):
 prob=clf.predict_proba(Z)[:,1];z=(prob>=TH).astype(int);P,R,F1,_=precision_recall_fscore_support(S['lab'],z,average='binary',zero_division=0);vals=[name,accuracy_score(S['lab'],z),P,R,F1,average_precision_score(S['lab'],prob),roc_auc_score(S['lab'],prob),100*S['lab'].mean()];pd.DataFrame(confusion_matrix(S['lab'],z)).to_csv(OUT/f'anomaly_confusion_{name}.csv',index=False);per=[]
 for typ in sorted(set(S['atype'])):
  if typ=='normal':continue
  q=S['atype']==typ;per.append([typ,int(q.sum()),float(z[q].mean()),float(prob[q].mean())])
 pd.DataFrame(per,columns=['fault_type','hours','detection_rate','mean_probability']).to_csv(OUT/f'anomaly_per_type_{name}.csv',index=False);return vals,prob
rows=[]
for nm,S,Z in [('seen_test',TE,ZTE),('unseen_full',UN,ZUN),('unseen_temporal',UNT,ZUNT)]:
 v,p=ev(nm,S,Z);rows.append(v);print(v,flush=True)
pd.DataFrame(rows,columns=['split','Accuracy','Precision','Recall','F1','AUPRC','AUROC','anomaly_pct']).to_csv(OUT/'anomaly_results_final.csv',index=False)
for nm,S,Z in [('seen_test',TE,ZTE),('unseen_full',UN,ZUN),('unseen_temporal',UNT,ZUNT)]:
 prob=clf.predict_proba(Z)[:,1]; pd.DataFrame({'building_id':S['bid'].astype(int),'timestamp':S['ts'],'label':S['lab'].astype(int),'probability':prob,'predicted':(prob>=TH).astype(int)}).to_csv(OUT/f'anomaly_row_predictions_{nm}.csv',index=False)
# Spike rate ablation - same head, remove column 7, same split/threshold retuning on validation
ZV0=np.delete(ZV,7,axis=1);ZUN0=np.delete(ZUN,7,axis=1);cl0=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=.05,random_state=11).fit(ZV0[:cut],VA['lab'][:cut]);pv0=cl0.predict_proba(ZV0[cut:])[:,1];best0=(-1,.5)
for th in np.linspace(.03,.60,230):
 z=(pv0>=th).astype(int);_,_,f1,_=precision_recall_fscore_support(yt,z,average='binary',zero_division=0)
 if f1>best0[0]:best0=(f1,float(th))
prob0=cl0.predict_proba(ZUN0)[:,1];z0=(prob0>=best0[1]);P,R,F1,_=precision_recall_fscore_support(UN['lab'],z0,average='binary',zero_division=0)
abl=pd.DataFrame([['Full 12 features incl. spike rate',rows[1][4],rows[1][5],rows[1][6],TH],['Without spike rate',F1,average_precision_score(UN['lab'],prob0),roc_auc_score(UN['lab'],prob0),best0[1]]],columns=['configuration','F1','AUPRC','AUROC','threshold']);abl.to_csv(OUT/'spike_rate_ablation.csv',index=False);print(abl,flush=True)
# threshold sensitivity on unseen full
sen=[];probun=clf.predict_proba(ZUN)[:,1]
for th in [max(.01,TH-.08),max(.01,TH-.04),TH,min(.95,TH+.04),min(.95,TH+.08)]:
 z=(probun>=th);P,R,F1,_=precision_recall_fscore_support(UN['lab'],z,average='binary',zero_division=0);sen.append([th,P,R,F1])
pd.DataFrame(sen,columns=['threshold','Precision','Recall','F1']).to_csv(OUT/'anomaly_threshold_sensitivity_final.csv',index=False)
# control on temporally aligned unseen
PROB=clf.predict_proba(ZUNT)[:,1];EIA=.1341
def raw_mult(ts):
 h=ts.hour;wd=ts.dayofweek<5
 if wd and 16<=h<21:return 1.65
 if wd and ((8<=h<16) or (21<=h<23)):return 1.05
 return .65
mult=np.array([raw_mult(pd.Timestamp(t)) for t in UNT['ts']],float);price=EIA*mult/mult.mean();p90={int(b):float(tr[tr.building_id==b].clean.quantile(.90)) for b in tr.building_id.unique()};ratio=np.median([p90[b]/math.sqrt(float(tr[tr.building_id==b].square_feet.iloc[0])) for b in p90]);thr=np.array([ratio*math.sqrt(a) for a in UNT['area']])
def sim(fc,prob,gate=True):
 base=UNT['clean'].copy();controlled=base.copy();cur=np.zeros(len(base));gated=np.zeros(len(base),bool)
 for i,(t,yh,ap,oc,hv,th0) in enumerate(zip(UNT['ts'],fc,prob,UNT['occ'],UNT['hvac'],thr)):
  ts=pd.Timestamp(t);rm=raw_mult(ts);risk=(yh-th0)/max(th0,1)
  if gate and ap>=TH:gated[i]=1;frac=0
  elif rm>=1.6 and risk>0 and oc<=.90:frac=min(.12,.04+.12*risk)
  elif rm>.7 and risk>.05 and oc<=.80:frac=min(.06,.015+.06*risk)
  else:frac=0
  cur[i]=frac*hv;controlled[i]-=cur[i]
 for i,e in enumerate(cur):
  if e<=0:continue
  cand=list(range(i+1,min(len(base),i+7)))
  if cand:controlled[min(cand,key=lambda k:price[k])]+=.35*e
 return controlled,gated
cm,gate=sim(PUNT,PROB,True);co,_=sim(UNT['clean'],PROB,True);cng,_=sim(PUNT,PROB,False)
ctrl=[]
for bid in sorted(np.unique(UNT['bid'])):
 q=UNT['bid']==bid;arch=df[df.building_id==bid].archetype.iloc[0]
 for nm,c in [('Model forecast + anomaly gate',cm),('Perfect load + anomaly gate',co),('Model forecast, no gate',cng)]:
  be=UNT['clean'][q].sum();ce=c[q].sum();bc=(UNT['clean'][q]*price[q]).sum();cc=(c[q]*price[q]).sum();ctrl.append([bid,arch,nm,be,ce,100*(be-ce)/be,bc,cc,100*(bc-cc)/bc,int(gate[q].sum()) if nm.startswith('Model forecast + anomaly') else np.nan])
out=pd.DataFrame(ctrl,columns=['building_id','archetype','controller','baseline_kWh','controlled_kWh','energy_reduction_pct','baseline_cost_USD','controlled_cost_USD','cost_reduction_pct','gated_hours']);out.to_csv(OUT/'control_results_final.csv',index=False);print(out,flush=True)
pd.DataFrame({'building_id':UNT['bid'].astype(int),'timestamp':UNT['ts'],'clean_load':UNT['clean'],'forecast':PUNT,'anomaly_probability':PROB,'gated':gate.astype(int),'price_USD_per_kWh':price,'controlled_load':cm}).to_csv(OUT/'control_trace_primary.csv',index=False)
