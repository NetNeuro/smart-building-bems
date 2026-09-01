from pathlib import Path
import math, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import f, levene

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'synthetic_benchmark.csv'
F=['temp','humidity','solar','occupancy','log_area','hour_sin','hour_cos','dow_sin','dow_cos','lag1','lag2','lag6','lag12','lag24','roll24']
SEL=['humidity','solar','occupancy','log_area','dow_sin','dow_cos','lag1','lag2','lag24']; idx=np.array([F.index(x) for x in SEL]); L=24
CAN={'ch':20,'lr':0.0015287143259818435,'drop':0.08,'beta':0.9431572529176961,'theta':0.36879562783560627,'tau':0.18,'kappa':0.35}
GRU_H={'ch':20,'h':20,'drop':0.08,'lr':0.0015287143259818435}
SEEDS=[11,23,37,41,53,67,71,83,97,109]
df=pd.read_csv(DATA,parse_dates=['timestamp']); df['log_area']=np.log1p(df.square_feet); df['hour_sin']=np.sin(2*np.pi*df.hour/24); df['hour_cos']=np.cos(2*np.pi*df.hour/24); df['dow_sin']=np.sin(2*np.pi*df.dow/7); df['dow_cos']=np.cos(2*np.pi*df.dow/7)
seen=df[df.building_id<9].copy(); unseen=df[df.building_id>=9].copy(); trp=[];vap=[];tep=[]
for b,g in seen.groupby('building_id'):
    g=g.sort_values('timestamp').reset_index(drop=True); n=len(g); a=int(.70*n); c=int(.85*n); trp.append(g.iloc[:a]);vap.append(g.iloc[a:c]);tep.append(g.iloc[c:])
tr=pd.concat(trp,ignore_index=True); sc=StandardScaler().fit(tr[F]); delta=(tr.clean-tr.lag1)/np.sqrt(tr.square_feet); MU=float(delta.mean());SD=float(delta.std()+1e-9)
def seq(frame):
    X=[];y=[];anchor=[];clean=[];area=[]
    for b,g in frame.groupby('building_id'):
        g=g.sort_values('timestamp').reset_index(drop=True); z=sc.transform(g[F]).astype('float32')
        for i in range(L,len(g)):
            last=float(g.meter.iloc[i-1]); target=float(g.clean.iloc[i]); yy=((target-last)/math.sqrt(float(g.square_feet.iloc[i]))-MU)/SD
            X.append(z[i-L:i]);y.append(yy);anchor.append(last);clean.append(target);area.append(float(g.square_feet.iloc[i]))
    return {k:np.asarray(v,np.float32 if k in ['X','y'] else float) for k,v in dict(X=X,y=y,anchor=anchor,clean=clean,area=area).items()}
TR=seq(tr);UN=seq(unseen); ii=np.linspace(0,len(TR['X'])-1,min(6800,len(TR['X'])),dtype=int)
class RAB(nn.Module):
    def __init__(self,ch,drop):
        super().__init__();self.c1=nn.Conv1d(ch,ch,3,padding=1);self.c2=nn.Conv1d(ch,ch,3,padding=1);self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid());self.ta=nn.Sequential(nn.Conv1d(2,1,5,padding=2),nn.Sigmoid());self.dr=nn.Dropout(drop)
    def forward(self,x):
        h=torch.relu(self.c1(x));h=self.dr(self.c2(h));h=h*self.ca(h);h=h*self.ta(torch.cat([h.mean(1,keepdim=True),h.max(1,keepdim=True).values],1));return torch.relu(x+h)
class HDRAB(nn.Module):
    def __init__(self,ch,drop):
        super().__init__();self.ds=nn.ModuleList([nn.Conv1d(ch,ch,3,padding=d,dilation=d) for d in [1,2,4]]);self.mix=nn.Conv1d(3*ch,ch,1);self.ca=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(ch,max(4,ch//4),1),nn.ReLU(),nn.Conv1d(max(4,ch//4),ch,1),nn.Sigmoid());self.dr=nn.Dropout(drop)
    def forward(self,x):
        h=self.dr(self.mix(torch.cat([torch.relu(c(x)) for c in self.ds],1)));return torch.relu(x+h*self.ca(h))
class DRA(nn.Module):
    def __init__(self,nf,ch,drop):super().__init__();self.i=nn.Conv1d(nf,ch,1);self.a=RAB(ch,drop);self.b=HDRAB(ch,drop);self.f=nn.Conv1d(2*ch,ch,1)
    def forward(self,x):z=torch.relu(self.i(x.transpose(1,2)));return torch.relu(self.f(torch.cat([self.a(z),self.b(z)],1))).transpose(1,2)
class PSNN(nn.Module):
    def __init__(self,ch):super().__init__();self.w=nn.Linear(ch,max(12,ch));self.o=nn.Linear(max(12,ch),1)
    def forward(self,z):
        u=torch.zeros(z.shape[0],self.w.out_features,device=z.device);sp=[]
        for t in range(z.shape[1]):u=CAN['beta']*u+self.w(z[:,t]);p=torch.sigmoid((u-CAN['theta'])/CAN['tau']);u=u*(1-CAN['kappa']*p);sp.append(p)
        return self.o(torch.stack(sp,1).mean(1)).squeeze(-1)
class Prop(nn.Module):
    def __init__(self):super().__init__();self.d=DRA(len(idx),CAN['ch'],CAN['drop']);self.p=PSNN(CAN['ch'])
    def forward(self,x):return self.p(self.d(x))
class DRAGRU(nn.Module):
    def __init__(self):super().__init__();self.d=DRA(len(idx),GRU_H['ch'],GRU_H['drop']);self.r=nn.GRU(GRU_H['ch'],GRU_H['h'],batch_first=True);self.o=nn.Linear(GRU_H['h'],1)
    def forward(self,x):z=self.d(x);h,_=self.r(z);return self.o(h[:,-1]).squeeze(-1)
def train(m,seed,lr):
    torch.manual_seed(seed);np.random.seed(seed);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);xt=torch.tensor(TR['X'][ii][:,:,idx]);yt=torch.tensor(TR['y'][ii])
    m.train()
    for _ in range(5):
        perm=torch.randperm(len(xt))
        for j in range(0,len(xt),256):
            q=perm[j:j+256];p=m(xt[q]);loss=((p-yt[q])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
    return m
def pred(m):
    m.eval();o=[]
    with torch.no_grad():
        for j in range(0,len(UN['X']),1024):o.append(m(torch.tensor(UN['X'][j:j+1024,:,idx])).numpy())
    rr=np.concatenate(o);return UN['anchor']+(rr*SD+MU)*np.sqrt(UN['area'])
rows=[]
for seed in SEEDS:
    for name,ctor,lr in [('DOptDRPSNN',Prop,CAN['lr']),('DRA-GRU',DRAGRU,GRU_H['lr'])]:
        p=pred(train(ctor(),seed,lr));rm=float(np.sqrt(mean_squared_error(UN['clean'],p)));ma=float(mean_absolute_error(UN['clean'],p));r2=float(r2_score(UN['clean'],p));rows.append([name,seed,rm,ma,r2]);print(name,seed,rm,flush=True)
out=pd.DataFrame(rows,columns=['model','seed','RMSE','MAE','R2']);out.to_csv(ROOT/'results'/'stability_10seeds_encoder_matched.csv',index=False)
print(out.groupby('model').agg(['mean','std']))
a=out[out.model=='DOptDRPSNN'].RMSE.values;b=out[out.model=='DRA-GRU'].RMSE.values
Fv=np.var(b,ddof=1)/np.var(a,ddof=1); p=1-f.cdf(Fv,len(b)-1,len(a)-1)
print('F DRA-GRU/PSNN',Fv,'one-sided p',p)
bf_stat,bf_p=levene(a,b,center='median')
print('Brown-Forsythe',bf_stat,bf_p)
pd.DataFrame([['DRA-GRU vs DOptDRPSNN',Fv,p,bf_stat,bf_p]],columns=['comparison','variance_ratio_F','one_sided_p','brown_forsythe_stat','brown_forsythe_p']).to_csv(ROOT/'results'/'stability_variance_tests.csv',index=False)
