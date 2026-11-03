"""Text-conditioned marked renewal distribution with a causal span-history Transformer."""
import json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from . import base as B
ROOT=Path(__file__).resolve().parent
CONFIG = {'width': 512, 'context': 256, 'heads': 8, 'layers': 12, 'batch': 64, 'epochs': 8, 'lr': .0003, 'warmup_steps': 100}

class Predictor(nn.Module):
 def __init__(self,variant,meta,width=None,layers=None):
  super().__init__();self.variant=variant;self.K=len(meta['operators']);w=width or CONFIG['width'];depth=layers or CONFIG['layers']
  self.mode=nn.Embedding(self.K+1,24);self.prev=nn.Embedding(self.K+1,16);self.solver=nn.Embedding(len(meta['solvers']),16);self.bench=nn.Embedding(len(meta['benchmarks']),8)
  self.proj=nn.Linear(41+64+2*384,w);self.position=nn.Embedding(CONFIG['context'],w)
  if variant.startswith('transformer'):
   layer=nn.TransformerEncoderLayer(w,CONFIG['heads'],w*4,.1,batch_first=True,norm_first=True,activation='gelu');self.core=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
   # Cloned Encoder layers must not start with identical weights.
   for module in self.core.modules():
    if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight)
  else:self.core=nn.Sequential(nn.GELU(),nn.Linear(w,w),nn.GELU(),nn.Linear(w,w),nn.GELU())
  self.norm=nn.LayerNorm(w);self.gate=nn.Linear(w,3);self.dest=nn.Linear(w,self.K);self.events=nn.Linear(w,6);self.length=nn.Linear(w,1);self.wait=nn.Linear(w,3*(3+self.K+1));self.semantic=nn.Linear(w,384)
  with torch.no_grad():
   bias=self.wait.bias.view(3,3+self.K+1);bias[:,1]=7.;bias[:,2]=0.
 def forward(self,b):
  x=b['x'];text=b['text'];prompt=b['prompt']
  if self.variant=='transformer_no_events':x=torch.cat((x[...,:5],torch.zeros_like(x[...,5:])),dim=-1)
  if self.variant=='transformer_no_text':text=torch.zeros_like(text);prompt=torch.zeros_like(prompt)
  h=self.proj(torch.cat((x,self.mode(b['mode']),self.prev(b['prev_mode']),self.solver(b['solver']),self.bench(b['bench']),text,prompt),dim=-1))
  if self.variant.startswith('transformer'):
   n=h.shape[1];h=h+self.position(torch.arange(n,device=h.device))[None];mask=torch.triu(torch.ones((n,n),dtype=torch.bool,device=h.device),1);h=self.core(h,mask=mask,src_key_padding_mask=b['pad'])
  else:h=self.core(h)
  h=self.norm(h);p=self.wait(h).view(*h.shape[:-1],3,3+self.K+1)
  return dict(gate=self.gate(h),dest=self.dest(h),events=self.events(h),length=self.length(h).squeeze(-1),wait_pi=p[...,0],wait_logscale=p[...,1].clamp(-3,16),wait_shape=(F.softplus(p[...,2])+.2).clamp(max=5),wait_mark=p[...,3:],semantic=F.normalize(self.semantic(h),dim=-1))

def wait_nll(o,t,observed,mark):
 logt=t.clamp_min(1e-5).log().unsqueeze(-1);ls=o['wait_logscale'];shape=o['wait_shape'];exponent=(shape*(logt-ls)).clamp(max=20);power=exponent.exp();lp=F.log_softmax(o['wait_pi'],dim=-1)
 ld=shape.log()-ls+(shape-1)*(logt-ls)-power
 lm=F.log_softmax(o['wait_mark'],dim=-1).gather(-1,mark.clamp_min(0)[...,None,None].expand(*mark.shape,3,1)).squeeze(-1)
 density=torch.logsumexp(lp+ld+lm,dim=-1);survival=torch.logsumexp(lp-power,dim=-1)
 return -torch.where(observed,density,survival)

def losses(o,b):
 base=B.losses(o,b);valid=b['score']&(b['wait_y']>0);w=b['weight']*valid
 nll=wait_nll(o,b['wait_y'],b['wait_observed'],b['next_span']);loss=base+.2*(nll*w).sum()/w.sum().clamp_min(1e-8)
 if 'next_text' in b:
  valid=b['score']&((b['gate']==0)|(b['gate']==1));where=valid.flatten().nonzero().flatten()
  if len(where)>512:where=where[torch.randperm(len(where),device=where.device)[:512]]
  if len(where)>1:
   pred=o['semantic'].flatten(0,1)[where];target=F.normalize(b['next_text'].flatten(0,1)[where],dim=-1);ids=b['next_text_id'].flatten()[where];logits=(pred@target.T)/.1;positive=ids[:,None]==ids[None,:]
   # Identical texts count as multiple positives, never as false negatives.
   cpc=torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~positive,-1e9),-1);loss=loss+.1*cpc.mean()
 return loss

def load_data(device='mps'):
 d,meta=B.load_data();d['embedding']=None
 if device:
  manifest=json.loads((ROOT/'data/embeddings_manifest.json').read_text());assert B.sha(ROOT/'data/embeddings.npy')==manifest['sha256']
  emb=np.load(ROOT/'data/embeddings.npy',mmap_mode='r');d['embedding']=torch.tensor(emb,device=device,dtype=torch.float32)
 return d,meta

def batch(d,ids,device):
 chunks=d['chunks'][ids];size=len(chunks);T=CONFIG['context'];index=np.zeros((size,T),np.int64);pad=np.ones((size,T),bool);score=np.zeros((size,T),bool)
 for i,(s,n,w,ti,sp) in enumerate(chunks):index[i,:n]=np.arange(s,s+n);pad[i,:n]=False;score[i,w:n]=True
 fields=['x','mode','prev_mode','gate','dest','event_y','event_mask','length_y','solver','bench','weight','wait_y','wait_observed','next_span']
 b={k:torch.from_numpy(d[k][index]).to(device) for k in fields}
 for k in ['mode','prev_mode','gate','dest','solver','bench','next_span']:b[k]=b[k].long()
 b['pad']=torch.from_numpy(pad).to(device);b['score']=torch.from_numpy(score).to(device);b['index']=index
 for source,target in [('text_id','text'),('prompt_id','prompt')]:
  idx=torch.from_numpy(d[source][index].astype(np.int64)).to(device);b[target]=d['embedding'][idx]
 future=np.minimum(index+1,len(d['mode'])-1);ni=torch.from_numpy(d['text_id'][future].astype(np.int64)).to(device);b['next_text_id']=ni;b['next_text']=d['embedding'][ni]
 return b

@torch.no_grad()
def evaluate(model,d,meta,split,device,batch_size=128,save_to=None,progress=None):
 chunks=np.flatnonzero(d['chunks'][:,4]==split);pieces=[];outs={k:[] for k in ['gate','dest','events','length','wait_pi','wait_logscale','wait_shape','wait_mark']}
 model.eval()
 for start in range(0,len(chunks),batch_size):
  b=batch(d,chunks[start:start+batch_size],device);o=model(b);m=(b['score']&(b['gate']>=0)).cpu().numpy();pieces.append(b['index'][m])
  for k in outs:
   v=o[k];v=F.softmax(v,-1) if k in ('gate','dest') else torch.sigmoid(v) if k=='events' else v;outs[k].append(v.cpu().numpy()[m])
  if progress and start%(batch_size*10)==0:progress('evaluation',chunks_done=start,total_chunks=len(chunks))
 ids=np.concatenate(pieces);pred={k:np.concatenate(v) for k,v in outs.items()};order=np.argsort(ids);ids=ids[order];pred={k:v[order] for k,v in pred.items()}
 assert np.array_equal(ids,np.flatnonzero((d['split']==split)&(d['gate']>=0)))
 result=B.summarize_predictions(d,ids,pred)
 cpu={k:torch.from_numpy(v) for k,v in pred.items() if k.startswith('wait_')};times=torch.from_numpy(d['wait_y'][ids]);obs=torch.from_numpy(d['wait_observed'][ids]);marks=torch.from_numpy(d['next_span'][ids]);nll=wait_nll(cpu,times,obs,marks).numpy()
 pi=torch.softmax(cpu['wait_pi'],-1);pm=torch.softmax(cpu['wait_mark'],-1);mp=(pi[...,None]*pm).sum(-2).numpy();valid=(d['wait_y'][ids]>0);observed=valid&d['wait_observed'][ids];question=d['question'][ids]
 for row in result['question_metrics']:
  q=question==row['question'];v=q&valid;ob=q&observed
  if v.any():row['wait_joint_nll']=float(nll[v].mean())
  if ob.any():
   yy=d['next_span'][ids][ob];probs=mp[ob];row['next_span_nll']=float(-np.log(np.clip(probs[np.arange(len(yy)),yy],1e-9,1)).mean());row['next_span_accuracy']=float((probs.argmax(-1)==yy).mean())
 for key in ['wait_joint_nll','next_span_nll','next_span_accuracy']:result['scores'][key]=float(np.mean([r[key] for r in result['question_metrics'] if key in r]))
 reliability=[]
 confidence=mp[observed].max(-1);correct=(mp[observed].argmax(-1)==d['next_span'][ids][observed]);ece=0.
 for lo in np.linspace(0,0.9,10):
  use=(confidence>=lo)&(confidence<(lo+.1) if lo<.89 else confidence<=1)
  if use.any():
   acc=float(correct[use].mean());conf=float(confidence[use].mean());ece+=float(use.mean())*abs(acc-conf);reliability.append(dict(lower=float(lo),count=int(use.sum()),accuracy=acc,confidence=conf))
 result['next_span_reliability_pooled']=reliability;result['scores']['next_span_ece_pooled']=ece
 result['scores']['selection_objective']=result['scores']['joint_nll']+.2*result['scores']['wait_joint_nll']
 if save_to:
  tmp=save_to.with_suffix('.tmp.npz');np.savez_compressed(tmp,index=ids,**pred);tmp.replace(save_to);result['predictions_sha256']=B.sha(save_to)
 return result
