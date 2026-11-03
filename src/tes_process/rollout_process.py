"""Paired DEV prefix rollouts in character windows, including absorbing natural END."""
import collections,hashlib,math
import numpy as np
import torch
from scipy.spatial.distance import pdist
from .common import ROOT,atomic,read,beat
from .structure_diagnostics import features

H=2048
def vector(mode,length,event,boundary,natural,K,E,verify_mode=3):
 v=np.zeros(2+2*K+3+1+E+1);total=sum(length);ended=natural and total<=H
 v[0]=min(total,H)/H;v[1]=ended
 seen=set();last=None;prev=None;run=0;switches=returns=aba=0;longest=0;spent=0
 for m,l,e,b in zip(mode,length,event,boundary):
  if spent>=H:break
  amount=min(l,H-spent);new=last is None or b or m!=last
  if new:
   v[2+m]+=1;run=0
   if last is not None:
    switches+=1;returns+=int(m in seen);aba+=int(prev==m)
   prev,last=last,m;seen.add(m)
  run+=amount
  if m==verify_mode:longest=max(longest,run)
  v[2+K+m]+=amount/H
  if spent+l<=H:v[2+2*K+4:2+2*K+4+E]+=e
  spent+=l
 v[2+2*K:2+2*K+3]=switches,returns,aba;v[2+2*K+3]=longest/H;v[-1]=bool(ended and aba)
 ix=list(range(2,2+K))+list(range(2+2*K,2+2*K+3))+list(range(2+2*K+4,2+2*K+4+E))
 v[ix]=v[ix]/(1+v[ix]);return v

def run(ni,family,seed,net,data,spec,out):
 if (out/'ROLLOUT.json').exists():return
 net.eval();K,E=spec['K'],spec['E'];M=read(ROOT/'RUN.json')['mc_paths'];rng=np.random.default_rng(20261026+seed)
 verify_mode=spec['mode_vocab'].index('VERIFYING') if 'VERIFYING' in spec['mode_vocab'] else -1
 groups=collections.defaultdict(list)
 for m in spec['sequences']:
  if m['role']==2:groups[m['question']].append(m)
 prefixes=[]
 for q,ms in sorted(groups.items()):
  for meta in sorted(ms,key=lambda m:hashlib.sha256(m['id'].encode()).hexdigest())[:2]:
   for j in [4,19]:
    if j>=meta['n']-1 or data['mode'][meta['start']+j]<0:continue
    lo=meta['start'];a=lo+j+1;z=lo+meta['n'];length=data['length'][a:z]
    if data['reset'][a:z].any() or (data['mode'][a:z]<0).any():continue
    if not meta['natural'] and sum(length)<=H:continue
    cumulative=np.cumsum(length);inside=cumulative<=H
    known=data['known'][a:z][inside].all()
    y=vector(data['mode'][a:z],length,data['event'][a:z],data['start'][a:z],meta['natural'],K,E,verify_mode)
    extra=features(data['mode'][a:z],length,data['event'][a:z],data['start'][a:z],meta['natural'],K,E,H)
    prefixes.append(dict(question=q,id=meta['id'],index=lo+j,trace_start=lo,observed=y,event_observed=bool(known),extra=extra))
 results=[];samples=[];actual=[]
 # Each prefix is vectorized over its Monte Carlo paths; the state includes generated events/text.
 keys=['base','memory','event','known','text','text_ema','length','reset','start','true_start']
 for pi,pre in enumerate(prefixes):
  state=None;lastout=None;ix=pre['index']
  with torch.no_grad():
   for j in range(pre['trace_start'],ix+1):
    b={key:torch.from_numpy(np.repeat(data[key][j:j+1],M,axis=0).copy()) for key in keys}
    if bool(data['reset'][j]) and not hasattr(net,'rules'):state=None
    lastout,state=net.step(b,state)
   oldmode=data['mode'][ix];spent=np.zeros(M);ended=np.zeros(M,bool);alive=np.ones(M,bool)
   modes=[[] for _ in range(M)];lengths=[[] for _ in range(M)];events=[[] for _ in range(M)];starts=[[] for _ in range(M)]
   current=np.full(M,oldmode,int);ema=b['memory'].numpy().reshape(M,4,E).copy();textema=b['text_ema'].numpy().copy()
   priorbase=b['base'].numpy().copy();counters=np.expm1(priorbase[:,K+16:]*10)
   edges=np.array([1,9,17,33,65,129,257,513,1025,2049,4097,8193,16385])
   def draw(p):return (rng.random(len(p))[:,None]>np.cumsum(p,axis=1)).sum(1).clip(max=p.shape[1]-1)
   for step in range(H+1):
    gate=draw(lastout['gate'].softmax(-1).numpy());stop=alive&(gate==1);ended[stop]=True;alive[stop]=False
    if not alive.any():break
    move=gate>=2;nextmode=np.where(move,gate-2,current)
    if 'length_cond' in lastout:
     lp=lastout['length_cond'][np.arange(M),nextmode].softmax(-1).numpy()
     ep=lastout['event_cond'][np.arange(M),nextmode].sigmoid().numpy()
    else:lp=lastout['length'].softmax(-1).numpy();ep=lastout['event'].sigmoid().numpy()
    bins=draw(lp)
    size=np.floor(edges[bins]+rng.random(M)*(edges[bins+1]-edges[bins])).astype(int)
    event=(rng.random((M,E))<ep).astype(np.float32)
    for j in np.flatnonzero(alive):
     modes[j].append(int(nextmode[j]));lengths[j].append(int(size[j]));events[j].append(event[j]);starts[j].append(bool(move[j]))
    spent[alive]+=size[alive];alive&=spent<H
    if not alive.any():break
    counters[:,0]+=1;counters[:,1]+=size;counters[:,2]=size
    counters[:,3]=np.where(move,1,counters[:,3]+1);counters[:,4]=np.where(move,size,counters[:,4]+size)
    base=priorbase.copy();base[:,:K]=0;base[np.arange(M),nextmode]=1;base[:,K+16:]=np.log1p(counters)/10
    ema=ema*np.exp(-1/np.array([1,4,16,64]))[None,:,None]+event[:,None,:]
    text=b['text'].numpy().copy()
    if net.with_text:
     text=lastout['text_mean'].numpy()+rng.normal(size=(M,32))*lastout['text_logsd'].exp().numpy()
     text=text/np.maximum(1,np.linalg.norm(text,axis=1))[:,None]
    textema=.9*textema+.1*text
    b={k:torch.from_numpy(np.asarray(v,np.float32)) for k,v in dict(base=base,memory=ema.reshape(M,-1),event=event,known=np.ones_like(event),
         text=text,text_ema=textema,length=size).items()}
    b['reset']=torch.zeros(M,dtype=torch.bool);b['true_start']=torch.zeros(M,dtype=torch.bool)
    b['start']=torch.from_numpy(move);lastout,state=net.step(b,state);current=nextmode
   assert not alive.any(),'rollout did not reach END or H; never turn a computational cutoff into END'
  x=np.stack([vector(m,l,e,s,bool(end),K,E,verify_mode) for m,l,e,s,end in zip(modes,lengths,events,starts,ended)])
  y=pre['observed'];samples.append(x.astype(np.float32));actual.append(y)
  row={k:pre[k] for k in ['question','id','index','event_observed']}
  for metric,cols in [('remaining_END',[0,1]),('span',list(range(2+2*K+4))+[len(y)-1]),('full',list(range(len(y))))]:
   if metric=='full' and not pre['event_observed']:continue
   xx=x[:,cols];yy=y[cols];scale=1 if metric=='remaining_END' else math.sqrt(len(cols))
   row[metric]=(np.linalg.norm(xx-yy,axis=1).mean()-pdist(xx).sum()/(M*(M-1)))/scale
  row['END_pred']=float(x[:,1].mean());row['END_true']=float(y[1]);results.append(row)
  simulated=[features(m,l,e,s,bool(end),K,E,H) for m,l,e,s,end in zip(modes,lengths,events,starts,ended)]
  for name in ['duration','mode_events','transitions','event_counts','event_pairs']:
   if name in ['mode_events','event_counts','event_pairs'] and not pre['event_observed']:continue
   xx=np.stack([v[name] for v in simulated]);yy=pre['extra'][name]
   row['structure_'+name]=float((np.linalg.norm(xx-yy,axis=1).mean()-pdist(xx).sum()/(M*(M-1)))/math.sqrt(len(yy)))
  if pi%10==0:beat('rollout',namespace=ni,family=family,seed=seed,done=pi,total=len(prefixes))
 perq={}
 for metric in ['remaining_END','span','full','structure_duration','structure_mode_events','structure_transitions','structure_event_counts','structure_event_pairs']:
  temp=collections.defaultdict(lambda:collections.defaultdict(list))
  for r in results:
   if metric in r:temp[r['question']][r['id']].append(r[metric])
  perq[metric]={q:float(np.mean([np.mean(v) for v in traces.values()])) for q,traces in temp.items()}
 atomic(out/'ROLLOUT.json',dict(horizon_characters=H,paths=M,prefixes=len(prefixes),
  per_question=perq,means={k:float(np.mean(list(v.values()))) if v else None for k,v in perq.items()},
  prefixes_rule='First two SHA-sorted DEV traces per question; fixed segment positions5,20; original observation masks',
  text_future='Text variants simulate the trained Gaussian lexical-sketch transition; not generated natural-language text',
  length_sampling='uniform integer within predicted bins; final bin8193..16384; scoring restricted to2048 chars',
  limitations=['Finite Monte Carlo,32 paths by default; intervals conditional on fixed fits.',
   'Native event presence per completed segment, not exact count of repeated same-type mentions.',
   'Some layers have no complete event windows; full score absent rather than treating unknown as zero.'],rows=results))
 if samples:np.savez_compressed(out/'ROLLOUT_SAMPLES.npz',predicted=np.stack(samples),observed=np.stack(actual))
