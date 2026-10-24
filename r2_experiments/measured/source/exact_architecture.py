# Exact class definitions extracted from scripts/run_canonical_s105.py
import torch
import torch.nn as nn
L=24

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

class Transformer(nn.Module):
    def __init__(self,nf,d=24,drop=.08):
        super().__init__(); self.p=nn.Linear(nf,d); self.pos=nn.Parameter(torch.zeros(1,L,d)); heads=4 if d%4==0 else 2
        ly=nn.TransformerEncoderLayer(d,heads,2*d,drop,batch_first=True); self.e=nn.TransformerEncoder(ly,1); self.o=nn.Linear(d,1)
    def forward(self,x):z=self.e(self.p(x)+self.pos[:,:x.shape[1]]);return self.o(z[:,-1]).squeeze(-1)
