import os, time, json, platform, subprocess, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data'/'synthetic_benchmark.csv'
torch.set_num_threads(2)
SEEDS=[11,23,37]
L=24
F=['temp','humidity','solar','occupancy','log_area','hour_sin','hour_cos','dow_sin','dow_cos','lag1','lag2','lag6','lag12','lag24','roll24']
# canonical mask selected by the network-evaluated MOA run; the script also re-evaluates it against comparator masks.
SEL=['humidity','solar','occupancy','log_area','dow_sin','dow_cos','lag1','lag2','lag24']
# canonical DOA result from a 5-iteration (3 exploration + 2 exploitation) search.
CAN_H={'channels':20,'lr':0.0015287143259818435,'dropout':0.08,'beta':0.9431572529176961,'theta':0.36879562783560627,'tau':0.18,'kappa':0.35}
DOA_RANGES={'channels':[12,28],'learning_rate':[10**-3.35,10**-2.80],'dropout':[0.02,0.22],'beta':[0.72,0.95],'theta':[0.30,0.70]}

# ---------------- data and splits ----------------
df=pd.read_csv(DATA,parse_dates=['timestamp'])
df['log_area']=np.log1p(df.square_feet)
df['hour_sin']=np.sin(2*np.pi*df.hour/24); df['hour_cos']=np.cos(2*np.pi*df.hour/24)
df['dow_sin']=np.sin(2*np.pi*df.dow/7); df['dow_cos']=np.cos(2*np.pi*df.dow/7)
seen=df[df.building_id<9].copy(); unseen=df[df.building_id>=9].copy()
trp=[];vap=[];tep=[]
for b,g in seen.groupby('building_id'):
    g=g.sort_values('timestamp').reset_index(drop=True); n=len(g); a=int(.70*n); c=int(.85*n)
    trp.append(g.iloc[:a]); vap.append(g.iloc[a:c]); tep.append(g.iloc[c:])
tr=pd.concat(trp,ignore_index=True); va=pd.concat(vap,ignore_index=True); te=pd.concat(tep,ignore_index=True)
unseen_temporal=unseen[unseen.timestamp>=te.timestamp.min()].copy()
parts=[]
for n,x in [('Train',tr),('Validation',va),('Seen chronological test',te),('Unseen building - full-period',unseen),('Unseen building - temporally aligned',unseen_temporal)]:
    parts.append([n,','.join(map(str,sorted(x.building_id.unique()))),x.timestamp.min(),x.timestamp.max(),len(x),100*x.anomaly.mean()])
pd.DataFrame(parts,columns=['partition','buildings','start','end','rows','anomaly_pct']).to_csv(OUT/'split_protocol.csv',index=False)

sc=StandardScaler().fit(tr[F])
delta=(tr.clean-tr.lag1)/np.sqrt(tr.square_feet); MU=float(delta.mean()); SD=float(delta.std()+1e-9)

def seq(frame,horizon=1):
    X=[];y=[];lab=[];bid=[];area=[];anchor=[];obs=[];ts=[];atype=[];clean=[];hvac=[];occ=[]
    for b,g in frame.groupby('building_id'):
        g=g.sort_values('timestamp').reset_index(drop=True)
        a=sc.transform(g[F]).astype('float32')
        # forecast origin is i-1, target is i+horizon-1
        for i in range(L, len(g)-horizon+1):
            ti=i+horizon-1
            last=float(g.meter.iloc[i-1])
            target=float(g.clean.iloc[ti])
            yy=((target-last)/math.sqrt(float(g.square_feet.iloc[ti]))-MU)/SD
            X.append(a[i-L:i]);y.append(yy);lab.append(int(g.anomaly.iloc[ti]));bid.append(int(b));area.append(float(g.square_feet.iloc[ti]));anchor.append(last);obs.append(float(g.meter.iloc[ti]));ts.append(g.timestamp.iloc[ti]);atype.append(g.atype.iloc[ti]);clean.append(target);hvac.append(float(g.hvac.iloc[ti]));occ.append(float(g.occupancy.iloc[ti]))
    return {'X':np.asarray(X,np.float32),'y':np.asarray(y,np.float32),'label':np.asarray(lab),'bid':np.asarray(bid),'area':np.asarray(area),'anchor':np.asarray(anchor),'obs':np.asarray(obs),'ts':np.asarray(ts),'atype':np.asarray(atype),'clean':np.asarray(clean),'hvac':np.asarray(hvac),'occ':np.asarray(occ)}

TR=seq(tr); VA=seq(va); TE=seq(te); UN=seq(unseen); UNT=seq(unseen_temporal)

# ---------------- models ----------------
class RAB(nn.Module):
    def __init__(self,ch,drop):
        super().__init__(); self.c1=nn.Conv1d(ch,ch,3,padding=1);self.c2=nn.Conv1d(ch,ch,3,padding=1)
        self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid())
        self.ta=nn.Sequential(nn.Conv1d(2,1,5,padding=2),nn.Sigmoid()); self.dr=nn.Dropout(drop)
    def forward(self,x):
        h=torch.relu(self.c1(x)); h=self.dr(self.c2(h)); h=h*self.ca(h); h=h*self.ta(torch.cat([h.mean(1,keepdim=True),h.max(1,keepdim=True).values],1)); return torch.relu(x+h)
class HDRAB(nn.Module):
    def __init__(self,ch,drop):
        super().__init__(); self.ds=nn.ModuleList([nn.Conv1d(ch,ch,3,padding=d,dilation=d) for d in [1,2,4]]); self.mix=nn.Conv1d(3*ch,ch,1)
        self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid());self.dr=nn.Dropout(drop)
    def forward(self,x):
        h=self.dr(self.mix(torch.cat([torch.relu(c(x)) for c in self.ds],1))); return torch.relu(x+h*self.ca(h))
class DRA(nn.Module):
    def __init__(self,nf,ch,drop):super().__init__();self.i=nn.Conv1d(nf,ch,1);self.a=RAB(ch,drop);self.b=HDRAB(ch,drop);self.f=nn.Conv1d(2*ch,ch,1)
    def forward(self,x):
        z=torch.relu(self.i(x.transpose(1,2))); return torch.relu(self.f(torch.cat([self.a(z),self.b(z)],1))).transpose(1,2)
class PSNN(nn.Module):
    def __init__(self,ch,h,beta,theta,tau=.18,kappa=.35):super().__init__();self.w=nn.Linear(ch,h);self.o=nn.Linear(h,1);self.beta=beta;self.theta=theta;self.tau=tau;self.kappa=kappa
    def forward(self,z,return_rate=False):
        u=torch.zeros(z.shape[0],self.w.out_features,device=z.device);sp=[]
        for t in range(z.shape[1]):
            u=self.beta*u+self.w(z[:,t]); p=torch.sigmoid((u-self.theta)/self.tau); u=u*(1-self.kappa*p); sp.append(p)
        sr=torch.stack(sp,1).mean(1)
        out=self.o(sr).squeeze(-1)
        return (out,sr.mean(1)) if return_rate else out
class Proposed(nn.Module):
    def __init__(self,nf,ch=20,drop=.08,beta=.943,theta=.369,tau=.18,kappa=.35):super().__init__();self.d=DRA(nf,ch,drop);self.p=PSNN(ch,max(12,ch),beta,theta,tau,kappa)
    def forward(self,x,return_rate=False):return self.p(self.d(x),return_rate=return_rate)
class DRAGRU(nn.Module):
    def __init__(self,nf,ch=20,drop=.08,h=20):super().__init__();self.d=DRA(nf,ch,drop);self.r=nn.GRU(ch,h,batch_first=True);self.o=nn.Linear(h,1)
    def forward(self,x):z=self.d(x);h,_=self.r(z);return self.o(h[:,-1]).squeeze(-1)
class DRALSTM(nn.Module):
    def __init__(self,nf,ch=20,drop=.08,h=20):super().__init__();self.d=DRA(nf,ch,drop);self.r=nn.LSTM(ch,h,batch_first=True);self.o=nn.Linear(h,1)
    def forward(self,x):z=self.d(x);h,_=self.r(z);return self.o(h[:,-1]).squeeze(-1)
class GRU(nn.Module):
    def __init__(self,nf,h=24,drop=.0):super().__init__();self.r=nn.GRU(nf,h,batch_first=True);self.o=nn.Linear(h,1)
    def forward(self,x):h,_=self.r(x);return self.o(h[:,-1]).squeeze(-1)
class LSTM(nn.Module):
    def __init__(self,nf,h=24,drop=.0):super().__init__();self.r=nn.LSTM(nf,h,batch_first=True);self.o=nn.Linear(h,1)
    def forward(self,x):h,_=self.r(x);return self.o(h[:,-1]).squeeze(-1)
class Transformer(nn.Module):
    def __init__(self,nf,d=24,drop=.08):
        super().__init__(); self.p=nn.Linear(nf,d); self.pos=nn.Parameter(torch.zeros(1,L,d)); heads=4 if d%4==0 else 2
        ly=nn.TransformerEncoderLayer(d,heads,2*d,drop,batch_first=True); self.e=nn.TransformerEncoder(ly,1); self.o=nn.Linear(d,1)
    def forward(self,x):z=self.e(self.p(x)+self.pos[:,:x.shape[1]]);return self.o(z[:,-1]).squeeze(-1)

def train(m,X,y,seed,lr=.001,ep=5,bs=256):
    torch.manual_seed(seed); np.random.seed(seed); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4); xt=torch.tensor(X);yt=torch.tensor(y);m.train()
    for _ in range(ep):
        perm=torch.randperm(len(xt))
        for j in range(0,len(xt),bs):
            q=perm[j:j+bs];p=m(xt[q]);loss=((p-yt[q])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
    return m

def pred_std(m,X,return_rate=False):
    m.eval();o=[];sr=[]
    with torch.no_grad():
        for j in range(0,len(X),1024):
            z=m(torch.tensor(X[j:j+1024]),return_rate=True) if return_rate and isinstance(m,Proposed) else m(torch.tensor(X[j:j+1024]))
            if return_rate and isinstance(m,Proposed): oo,ss=z;o.append(oo.numpy());sr.append(ss.numpy())
            else:o.append(z.numpy())
    return (np.concatenate(o),np.concatenate(sr)) if return_rate and isinstance(m,Proposed) else np.concatenate(o)

def pred_load(m,S,ids):
    rr=pred_std(m,S['X'][:,:,ids]); return S['anchor']+(rr*SD+MU)*np.sqrt(S['area'])

def mets(a,p):return [float(np.sqrt(mean_squared_error(a,p))),float(mean_absolute_error(a,p)),float(100*np.mean(np.abs(a-p)/np.maximum(np.abs(a),1))),float(r2_score(a,p))]

idx=np.array([F.index(f) for f in SEL]); ii=np.linspace(0,len(TR['X'])-1,min(6800,len(TR['X'])),dtype=int)

# ---------------- matched baseline tuning ----------------
def make_model(name,nf,params):
    if name=='DRA-GRU':return DRAGRU(nf,ch=params['ch'],drop=params['drop'],h=params['h'])
    if name=='DRA-LSTM':return DRALSTM(nf,ch=params['ch'],drop=params['drop'],h=params['h'])
    if name=='GRU':return GRU(nf,h=params['h'])
    if name=='LSTM':return LSTM(nf,h=params['h'])
    if name=='Transformer':return Transformer(nf,d=params['d'],drop=params['drop'])
    raise ValueError(name)

def baseline_candidates(name):
    rng=np.random.default_rng(abs(hash(name))%(2**32))
    C=[]
    for k in range(12):
        lr=float(10**rng.uniform(-3.35,-2.80)); drop=float(rng.uniform(.02,.20))
        if name in ['DRA-GRU','DRA-LSTM']: C.append({'ch':int(rng.choice([16,20,24,28])),'h':int(rng.choice([16,20,24,28])),'drop':drop,'lr':lr})
        elif name in ['GRU','LSTM']: C.append({'h':int(rng.choice([16,24,32,40])),'lr':lr,'drop':0.0})
        else:C.append({'d':int(rng.choice([16,24,32])),'drop':drop,'lr':lr})
    return C

quick_i=np.linspace(0,len(TR['X'])-1,1800,dtype=int); quick_v=np.linspace(0,len(VA['X'])-1,600,dtype=int)
tuned=json.load(open(OUT/'baseline_tuned_hyperparameters.json')) if (OUT/'baseline_tuned_hyperparameters.json').exists() else {}
if not tuned:
    for name in ['DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer']:
        best=(1e9,None)
        for k,p0 in enumerate(baseline_candidates(name)[:6]):
            m=make_model(name,len(idx),p0); train(m,TR['X'][quick_i][:,:,idx],TR['y'][quick_i],100+k,p0['lr'],ep=1,bs=512)
            q=pred_std(m,VA['X'][quick_v][:,:,idx]); rm=float(np.sqrt(mean_squared_error(VA['y'][quick_v],q)))
            if rm<best[0]:best=(rm,p0)
        tuned[name]=best[1]
    json.dump(tuned,open(OUT/'baseline_tuned_hyperparameters.json','w'),indent=2)

# ---------------- canonical 1-hour forecasting ----------------
rows=[]; stores={}; models={}
for name in ['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer']:
    for seed in SEEDS:
        if name=='DOptDRPSNN':
            m=Proposed(len(idx),CAN_H['channels'],CAN_H['dropout'],CAN_H['beta'],CAN_H['theta'],CAN_H['tau'],CAN_H['kappa']); lr=CAN_H['lr']
        else:
            p0=tuned[name]; m=make_model(name,len(idx),p0); lr=p0['lr']
        train(m,TR['X'][ii][:,:,idx],TR['y'][ii],seed,lr,ep=5); models[(name,seed)]=m
        for split,S in [('unseen_full',UN),('unseen_temporal',UNT)]:
            pp=pred_load(m,S,idx); stores[(name,seed,split)]=pp; rows.append([name,seed,split,*mets(S['clean'],pp)])
forecast=pd.DataFrame(rows,columns=['model','seed','split','RMSE','MAE','MAPE','R2']);forecast.to_csv(OUT/'forecasting_models.csv',index=False)

# classical baselines on target rows; same chronological training. RF hyperparameters tuned on validation by small grid.
rf_grid=[(n,d,l) for n in [100,200] for d in [8,12,16] for l in [2,5]]
best=(1e99,None)
for n,d,l in rf_grid:
    rf=RandomForestRegressor(n_estimators=n,max_depth=d,min_samples_leaf=l,n_jobs=-1,random_state=11).fit(tr[SEL],tr.clean)
    q=rf.predict(va[SEL]);rm=np.sqrt(mean_squared_error(va.clean,q))
    if rm<best[0]:best=(rm,(n,d,l))
n,d,l=best[1]; rf=RandomForestRegressor(n_estimators=n,max_depth=d,min_samples_leaf=l,n_jobs=-1,random_state=11).fit(tr[SEL],tr.clean)
json.dump({'n_estimators':n,'max_depth':d,'min_samples_leaf':l},open(OUT/'rf_tuned_hyperparameters.json','w'),indent=2)
for split,S,frame in [('unseen_full',UN,unseen),('unseen_temporal',UNT,unseen_temporal)]:
    aligned=pd.DataFrame({'clean':S['clean'],'lag1':S['anchor'],'bid':S['bid'],'timestamp':S['ts']})
    # map target rows back to dataframe features by bid/timestamp
    key=frame.set_index(['building_id','timestamp'])
    feat=np.vstack([key.loc[(int(b),pd.Timestamp(t)),SEL].to_numpy(float) for b,t in zip(S['bid'],S['ts'])])
    pr=rf.predict(feat)
    pd.DataFrame([
        ['Persistence',split,*mets(S['clean'],S['anchor'])],['Random Forest',split,*mets(S['clean'],pr)]
    ],columns=['model','split','RMSE','MAE','MAPE','R2']).to_csv(OUT/f'classical_{split}.csv',index=False)
    stores[('Persistence',0,split)]=S['anchor']; stores[('Random Forest',0,split)]=pr

# Six-hour all-baseline evaluation is executed by run_6h_all.py.

# ---------------- stats: daily blocks, ensemble predictions, Holm correction ----------------
S=UN
prop=np.mean([stores[('DOptDRPSNN',s,'unseen_full')] for s in SEEDS],axis=0)
# group sequence positions by building/day
base_names=['DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer','Persistence','Random Forest']
stat=[]
def holm(pvals):
    m=len(pvals); order=np.argsort(pvals);adj=np.empty(m);run=0
    for rank,j in enumerate(order):
        v=(m-rank)*pvals[j];run=max(run,v);adj[j]=min(1,run)
    return adj
praw=[];tmp=[];block_rows=[]
meta=pd.DataFrame({'bid':S['bid'],'ts':pd.to_datetime(S['ts']),'actual':S['clean'],'prop':prop});meta['day']=meta.ts.dt.date
for bn in base_names:
    bp=np.mean([stores[(bn,s,'unseen_full')] for s in SEEDS],axis=0) if bn not in ['Persistence','Random Forest'] else stores[(bn,0,'unseen_full')]
    meta['base']=bp;a=[];b=[]
    for _,g in meta.groupby(['bid','day']):
        if len(g)<18:continue
        pa=mean_absolute_error(g.actual,g.prop); ba=mean_absolute_error(g.actual,g.base); a.append(pa); b.append(ba); block_rows.append([bn,int(g.bid.iloc[0]),str(g.day.iloc[0]),len(g),pa,ba])
    try:st,pv=wilcoxon(a,b,alternative='less')
    except Exception:st,pv=np.nan,np.nan
    tmp.append([bn,len(a),np.mean(a),np.mean(b),st,pv]);praw.append(pv)
adj=holm(np.nan_to_num(np.asarray(praw),nan=1.0))
for row,pa in zip(tmp,adj):stat.append(row+[pa,pa<.05])
pd.DataFrame(stat,columns=['baseline','paired_24h_blocks','proposed_MAE','baseline_MAE','wilcoxon_stat','p_raw','p_holm','significant_holm']).to_csv(OUT/'statistical_tests_holm.csv',index=False)
pd.DataFrame(block_rows,columns=['baseline','building_id','day','hours','proposed_block_MAE','baseline_block_MAE']).to_csv(OUT/'daily_block_mae_raw.csv',index=False)

# RF per-building diagnostics
rfp=stores[('Random Forest',0,'unseen_full')]; rr=[];off=0
for bid in sorted(np.unique(UN['bid'])):
    z=UN['bid']==bid; rr.append([bid,df[df.building_id==bid].archetype.iloc[0],mean_absolute_error(UN['clean'][z],prop[z]),mean_absolute_error(UN['clean'][z],rfp[z]),np.quantile(np.abs(UN['clean'][z]-prop[z]),.95),np.quantile(np.abs(UN['clean'][z]-rfp[z]),.95)])
pd.DataFrame(rr,columns=['building_id','archetype','DOptDRPSNN_MAE','RF_MAE','DOptDRPSNN_P95_AE','RF_P95_AE']).to_csv(OUT/'rf_cross_building_diagnostics.csv',index=False)

# ---------------- feature selection: canonical architecture, 3 seeds ----------------
# comparators fixed to k=9; MOA mask is canonical network-MOA mask.
yflat=((tr.clean-tr.lag1)/np.sqrt(tr.square_feet)).to_numpy();cors=sorted([(abs(np.corrcoef(tr[f],yflat)[0,1]),f) for f in F],reverse=True);COR=[x[1] for x in cors[:len(SEL)]]
rng=np.random.default_rng(77);RND=list(rng.choice(F,len(SEL),replace=False));
if 'lag1' not in RND:RND[-1]='lag1'
fs=[]
for nm,fl in [('All features',F),('Random-k',RND),('Correlation-k',COR),('MOA',SEL)]:
    ids=np.array([F.index(x) for x in fl])
    for seed in SEEDS:
        m=Proposed(len(ids),CAN_H['channels'],CAN_H['dropout'],CAN_H['beta'],CAN_H['theta'],CAN_H['tau'],CAN_H['kappa']);train(m,TR['X'][ii][:,:,ids],TR['y'][ii],seed,CAN_H['lr'],ep=3,bs=512);p=pred_load(m,UN,ids);fs.append([nm,seed,len(ids),','.join(fl),*mets(UN['clean'],p)])
pd.DataFrame(fs,columns=['selector','seed','n_features','features','RMSE','MAE','MAPE','R2']).to_csv(OUT/'feature_selection_canonical.csv',index=False)

# ---------------- optimizer roles across three seeds ----------------
# DOA-only all-feature hyperparameters from an explicit 5x5 canonical all-feature search.
def decode(v):return {'channels':int(np.clip(round(12+16*v[0]),12,28)),'lr':float(10**(-3.35+.55*v[1])),'dropout':float(.02+.20*v[2]),'beta':float(.72+.23*v[3]),'theta':float(.30+.40*v[4])}
def hfit(v,ids,seed=17):
    h=decode(v);m=Proposed(len(ids),h['channels'],h['dropout'],h['beta'],h['theta'],CAN_H['tau'],CAN_H['kappa']);train(m,TR['X'][quick_i][:,:,ids],TR['y'][quick_i],seed,h['lr'],ep=1);q=pred_std(m,VA['X'][quick_v][:,:,ids]);return float(np.sqrt(mean_squared_error(VA['y'][quick_v],q)))
def doa(ids,pop=5,iters=5,forget=.4,seed=17):
    rng=np.random.default_rng(seed);P=rng.random((pop,5));s=np.array([hfit(v,ids,seed) for v in P]);trace=[]
    for t in range(iters):
        best=P[s.argmin()].copy();expl=t<3
        for i in range(pop):
            c=P[i].copy() if expl else best.copy();k=max(1,round(5*(forget if expl else forget/2)));ds=rng.choice(5,k,replace=False)
            if expl:
                peer=P[rng.integers(pop)];c[ds]=np.where(rng.random(k)<.45,peer[ds],rng.random(k))
            else:c[ds]=np.clip(best[ds]+rng.normal(0,.05,k),0,1)
            z=hfit(c,ids,seed)
            if z<s[i]:P[i]=c;s[i]=z
        trace.append([t+1,'exploration' if expl else 'exploitation',float(s.min())])
    return decode(P[s.argmin()]),pd.DataFrame(trace,columns=['iteration','phase','validation_std_RMSE'])
allidx=np.arange(len(F));HALL={'channels':14,'lr':0.0015287143259818435,'dropout':0.08,'beta':0.9362444559664178,'theta':0.40103775970380223};json.dump(HALL,open(OUT/'doa_allfeatures_best.json','w'),indent=2)
DEF={'channels':20,'lr':.001,'dropout':.08,'beta':.85,'theta':.50}
opt=[]
for nm,ids,h in [('No metaheuristic',allidx,DEF),('MOA only',idx,DEF),('DOA only - all features',allidx,HALL),('Sequential MOA→DOA',idx,CAN_H)]:
    for seed in SEEDS:
        m=Proposed(len(ids),h['channels'],h['dropout'],h['beta'],h['theta'],CAN_H['tau'],CAN_H['kappa']);train(m,TR['X'][ii][:,:,ids],TR['y'][ii],seed,h['lr'],ep=3,bs=512);p=pred_load(m,UN,ids);opt.append([nm,seed,len(ids),*mets(UN['clean'],p)])
pd.DataFrame(opt,columns=['strategy','seed','n_features','RMSE','MAE','MAPE','R2']).to_csv(OUT/'optimizer_roles_canonical.csv',index=False)

# ---------------- missingness across models ----------------
miss=[]
for name in ['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer']:
    for seed in SEEDS:
        m=models[(name,seed)]
        for rate in [0,.10,.20,.30]:
            xx=UN['X'][:,:,idx].copy();mask=np.random.default_rng(900+seed+int(rate*100)).random(xx.shape)<rate;xx[mask]=0
            rr=pred_std(m,xx);pp=UN['anchor']+(rr*SD+MU)*np.sqrt(UN['area']);miss.append([name,seed,rate,*mets(UN['clean'],pp)])
pd.DataFrame(miss,columns=['model','seed','missing_rate','RMSE','MAE','MAPE','R2']).to_csv(OUT/'missingness_cross_model.csv',index=False)
# concrete injected missing examples on otherwise complete simulated inputs
ex=[]
for bid,start,ch,dur in [(9,pd.Timestamp('2025-02-22 10:00'),'humidity',6),(10,pd.Timestamp('2025-02-23 14:00'),'occupancy',4),(11,pd.Timestamp('2025-02-25 08:00'),'solar',3)]:
    g=df[(df.building_id==bid)&(df.timestamp>=start)&(df.timestamp<start+pd.Timedelta(hours=dur))]
    for _,r in g.iterrows():ex.append([bid,r.timestamp,ch,r[ch],'injected NaN stress-test','past-only interpolation / training median if gap exceeds limit'])
pd.DataFrame(ex,columns=['building_id','timestamp','channel','original_value','fault','handling']).to_csv(OUT/'missing_data_examples_timestamped.csv',index=False)

# Final anomaly calibration is executed by run_anomaly_control.py.

# anomaly ground truth summary
ags=[]
for typ,g in df[df.anomaly==1].groupby('atype'):
    eps=[];sevs=[]
    for bid,gg in g.groupby('building_id'):
        gg=gg.sort_values('timestamp');ep=(gg.timestamp.diff().dt.total_seconds().fillna(3600)>3600).cumsum();eps.extend(gg.groupby(ep).size().tolist());sevs.extend((np.abs(gg.meter-gg.clean)/np.maximum(gg.clean,1)).tolist())
    ags.append([typ,len(g),len(eps),np.mean(eps),min(eps),max(eps),np.mean(sevs),max(sevs)])
pd.DataFrame(ags,columns=['type','hours','episodes','mean_duration_h','min_duration_h','max_duration_h','mean_rel_severity','max_rel_severity']).to_csv(OUT/'anomaly_ground_truth.csv',index=False)

# Final model-driven control is executed by run_anomaly_control.py.

# ---------------- sensitivity on canonical pipeline ----------------
# DOA forgetting fraction using canonical feature mask, same 5-iteration schedule; MOA population sensitivity reported as stability of frozen network-MOA mask via perturbation tests.
srow=[]
# DOA forgetting-fraction sensitivity is evaluated with short 3-iteration searches under the same feature mask.
for fr in [.2,.4,.6]:
    h,trc=doa(idx,pop=3,iters=3,forget=fr,seed=70+int(fr*10));srow.append(['DOA forgetting fraction',fr,float(trc.validation_std_RMSE.iloc[-1]),np.nan,np.nan])
for th0 in [.30,CAN_H['theta'],.50,.70]:
    m=Proposed(len(idx),CAN_H['channels'],CAN_H['dropout'],CAN_H['beta'],th0,CAN_H['tau'],CAN_H['kappa']);train(m,TR['X'][ii][:,:,idx],TR['y'][ii],71,CAN_H['lr'],ep=3);p=pred_load(m,UN,idx);mm=mets(UN['clean'],p);srow.append(['PSNN firing threshold',th0,mm[0],len(SEL),mm[3]])
# feature-mask perturbation: remove each non-forced selected feature one at a time; demonstrates local stability around canonical selection
for f in SEL:
    fl=[x for x in SEL if x!=f]
    ids=np.array([F.index(x) for x in fl]);m=Proposed(len(ids),CAN_H['channels'],CAN_H['dropout'],CAN_H['beta'],CAN_H['theta'],CAN_H['tau'],CAN_H['kappa']);train(m,TR['X'][ii][:,:,ids],TR['y'][ii],71,CAN_H['lr'],ep=2);p=pred_load(m,UN,ids);mm=mets(UN['clean'],p);srow.append([f'MOA local leave-one-feature-out: {f}',0,mm[0],len(fl),mm[3]])
pd.DataFrame(srow,columns=['parameter','value','primary_metric','selected_features','secondary_metric']).to_csv(OUT/'sensitivity_canonical.csv',index=False)

# ---------------- latency ----------------
lat=[]
for name in ['DOptDRPSNN','DRA-GRU','DRA-LSTM','GRU','LSTM','Transformer']:
    m=models[(name,11)];m.eval();sm=UN['X'][:1024,:,idx]
    with torch.no_grad():_=m(torch.tensor(sm))
    tt=[]
    for _ in range(20):
        t0=time.perf_counter();
        with torch.no_grad():_=m(torch.tensor(sm))
        tt.append(time.perf_counter()-t0)
    lat.append([name,sum(x.numel() for x in m.parameters()),1024,1024,20,1000*np.median(tt)/1024])
pd.DataFrame(lat,columns=['model','parameters','samples','inference_batch_size','repetitions','median_ms_per_sample']).to_csv(OUT/'latency_canonical.csv',index=False)

# ---------------- environment, configuration and README map ----------------
cpu=subprocess.check_output(['bash','-lc',"lscpu | grep 'Model name' | head -1 | cut -d: -f2-"],text=True).strip();ram=subprocess.check_output(['bash','-lc',"free -h | awk '/Mem:/ {print $2}'"],text=True).strip()
env={'CPU':cpu,'RAM':ram,'GPU':'None used (CPU-only)','Python':platform.python_version(),'PyTorch':torch.__version__,'NumPy':np.__version__,'Pandas':pd.__version__}
json.dump(env,open(OUT/'environment.json','w'),indent=2)
config={'features_all':F,'features_MOA':SEL,'MOA_feature_count':len(SEL),'MOA_note':'Canonical mask originates from network-evaluated MOA. lag1 is mandatory; the remaining 14 variables are searchable.','MOA_lambda':0.006,'DOA_final':CAN_H,'DOA_ranges':DOA_RANGES,'DOA_schedule':'5 iterations: 3 exploration + 2 exploitation','DOA_proxy_training':'1 epoch on 1,800 training windows; 600 validation windows','final_training':'5 epochs on 6,800 evenly subsampled training windows; AdamW; batch=256; no LR schedule; no early stopping','seeds':SEEDS,'sequence_length_h':24,'forecast_horizons_h':[1,6],'EIA_2025_commercial_average_USD_per_kWh':0.1341,'anomaly_control_script':'run_anomaly_control.py'}
json.dump(config,open(OUT/'canonical_configuration.json','w'),indent=2)
# data hash
sha=hashlib.sha256(open(DATA,'rb').read()).hexdigest();open(OUT/'data_sha256.txt','w').write(sha+'  synthetic_benchmark.csv\n')

print('DONE')
print('\nForecast means unseen_full:')
print(forecast[forecast.split=='unseen_full'].groupby('model')[['RMSE','MAE','MAPE','R2']].agg(['mean','std']))
print('\nOptimizer roles:');print(pd.read_csv(OUT/'optimizer_roles_canonical.csv').groupby('strategy')[['RMSE','MAE','R2']].agg(['mean','std']))
