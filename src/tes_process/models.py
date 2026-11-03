"""Segment-observation adaptations. Names explicitly distinguish these from old TPP likelihoods.

All models generate END/STAY/BOUNDARY, segment length and multi-label events.
Text variants additionally learn the next lexical sketch, allowing content-state rollouts.
The shared outputs are an experimental adapter, not a claim of unchanged historical models.
"""
import torch
from torch import nn
from torch.nn import functional as F

class Process(nn.Module):
 def __init__(self,family,K,E,L=12,H=32):
  super().__init__();self.family=family;self.K=K;self.E=E;self.L=L;self.H=H
  self.with_text=family.endswith('_text');kind=family.removesuffix('_text');self.kind=kind
  B=K+16+5
  if kind=='poisson':dim=1
  elif kind=='markov':dim=K+16
  elif kind=='semimarkov':dim=B
  elif kind in ['hawkes_binned','signed_hawkes_binned','semimarkov_events']:dim=B+4*E
  elif kind=='ingarch':
   dim=B+E;self.omega=nn.Parameter(torch.full((E,),-2.));self.alpha=nn.Parameter(torch.zeros(E,E)-3);self.beta=nn.Parameter(torch.zeros(E)-1)
  elif kind in ['neural_hawkes_observed','gru']:
   dim=B+H;inp=B+2*E+(64 if self.with_text else 0)
   if kind=='gru':self.recur=nn.GRUCell(inp,H)
   else:self.recur=nn.Linear(inp+H,7*H)
  else:raise ValueError(family)
  if self.with_text:dim+=64
  self.gate=nn.Linear(dim,K+2);self.events=nn.Linear(dim,E);self.lengths=nn.Linear(dim,L)
  if self.with_text:self.future_text=nn.Linear(dim,64)
  if kind=='hawkes_binned':
   self.hawkes_base=nn.Parameter(torch.full((E,),-3.));self.hawkes_A=nn.Parameter(torch.full((4*E,E),-5.))
   self.mode_rate=nn.Parameter(torch.full((K,E),-3.))
 def step(self,b,state=None):
  base=b['base'];n=len(base);e=b['event'];mask=b['known'];txt=torch.cat([b['text'],b['text_ema']],-1)
  if self.kind=='poisson':x=torch.ones(n,1,device=base.device)
  elif self.kind=='markov':x=base[:,:self.K+16]
  elif self.kind=='semimarkov':x=base
  elif self.kind in ['hawkes_binned','signed_hawkes_binned','semimarkov_events']:x=torch.cat([base,b['memory']],-1)
  elif self.kind=='ingarch':
   old=torch.zeros_like(e) if state is None else state
   a=F.softplus(self.alpha);beta=torch.sigmoid(self.beta)*.9
   a=a/(a.sum(0,keepdim=True)+1)*(1-beta)[None,:]*.9
   state=F.softplus(self.omega)+e@a+old*beta
   x=torch.cat([base,state],-1)
  else:
   inp=torch.cat([base,e,mask]+([txt] if self.with_text else []),-1)
   h=torch.zeros(n,self.H,device=base.device) if state is None else (state if self.kind=='gru' else state[0])
   if self.kind=='gru':h=self.recur(inp,h);state=h
   else:
    c=torch.zeros_like(h) if state is None else state[1];cb=torch.zeros_like(h) if state is None else state[2]
    # Decay uses only the already observed current segment's length, never the future target.
    if state is not None:
     c=cb+(c-cb)*torch.exp(-state[3]*(b['length'][:,None]/256).clamp(max=1000));h=state[4]*torch.tanh(c)
    gi,gf,go,gz,gib,gfb,d=self.recur(torch.cat([inp,h],-1)).chunk(7,-1)
    c=torch.sigmoid(gf)*c+torch.sigmoid(gi)*torch.tanh(gz)
    cb=torch.sigmoid(gfb)*cb+torch.sigmoid(gib)*torch.tanh(gz)
    go=torch.sigmoid(go);h=go*torch.tanh(c);state=(h,c,cb,F.softplus(d)+.001,go)
   x=torch.cat([base,h],-1)
  if self.with_text:x=torch.cat([x,txt],-1)
  eventlogits=self.events(x)
  if self.kind=='hawkes_binned':
   rate=F.softplus(self.hawkes_base)+base[:,:self.K]@F.softplus(self.mode_rate)+b['memory']@F.softplus(self.hawkes_A)
   prob=(-torch.expm1(-rate)).clamp(1e-6,1-1e-6);eventlogits=torch.logit(prob)
  elif self.kind in ['signed_hawkes_binned','ingarch']:
   rate=state if self.kind=='ingarch' else F.softplus(eventlogits)
   prob=(-torch.expm1(-rate)).clamp(1e-6,1-1e-6);eventlogits=torch.logit(prob)
  out=dict(gate=self.gate(x),event=eventlogits,length=self.lengths(x))
  if self.with_text:
   z=self.future_text(x);out['text_mean']=z[:,:32];out['text_logsd']=z[:,32:].clamp(-4,1)
  return out,state
 def forward(self,b):
  if self.kind not in ['ingarch','neural_hawkes_observed','gru']:
   n,t=b['base'].shape[:2];flat={k:v.reshape(n*t,*v.shape[2:]) for k,v in b.items()}
   out,_=self.step(flat)
   return {k:v.reshape(n,t,*v.shape[1:]) for k,v in out.items()}
  outs=[];state=None
  for j in range(b['base'].shape[1]):
   step={k:v[:,j] for k,v in b.items()};reset=step['reset'][:,None]
   if state is not None:state=tuple(v*(~reset) for v in state) if isinstance(state,tuple) else state*(~reset)
   out,state=self.step(step,state);outs.append(out)
  return {k:torch.stack([o[k] for o in outs],1) for k in outs[0]}

def losses(out,b):
 shape=b['gate'].shape
 gate=F.cross_entropy(out['gate'].reshape(-1,out['gate'].shape[-1]),b['gate'].clamp_min(0).long().reshape(-1),reduction='none').reshape(shape)
 events=F.binary_cross_entropy_with_logits(out['event'],b['next_event'],reduction='none')
 length=F.cross_entropy(out['length'].reshape(-1,out['length'].shape[-1]),b['next_length'].long().reshape(-1),reduction='none').reshape(shape)
 mask=b['next_known'];den=mask.sum(-1);ev=(events*mask).sum(-1)/den.clamp_min(1)
 total=gate*b['gknown']+ev+(length*b['cont'])
 if 'text_mean' in out:
  logsd=out['text_logsd'];nll=(.5*((b['next_text']-out['text_mean'])/logsd.exp()).square()+logsd).mean(-1)
  total=total+.1*nll*b['cont']
 return total,dict(gate=gate,events=ev,length=length,event_valid=den>0)
