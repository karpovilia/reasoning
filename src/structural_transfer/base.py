"""Causal segment Transformer with explicit span-boundary and destination heads."""
import argparse, hashlib, json, math, os, platform, random, site, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
VARIANTS=['summary_mlp','transformer_no_events','transformer']

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def atomic_torch(path,obj):
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def environment():
 return dict(python=sys.version,executable=sys.executable,user_site_enabled=site.ENABLE_USER_SITE,sys_path=sys.path,setup={k:os.environ.get(k) for k in ['PYTHONNOUSERSITE','PYTHONPATH','PATH','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK']},torch=torch.__version__,numpy=np.__version__,torch_path=torch.__file__,numpy_path=np.__file__,machine=platform.machine(),mps_built=torch.backends.mps.is_built(),mps_available=torch.backends.mps.is_available())
def configure(device):
 torch.set_num_threads(2 if device=='mps' else 1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 if device=='mps':
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; CPU fallback is not automatic')
  torch.mps.set_per_process_memory_fraction(.15)

def check_environment():
 actual=environment();expected=json.loads((ROOT/'environment_expected.json').read_text());return verify_environment(actual,expected,ROOT/'outputs')

def verify_environment(actual,expected,out):
 diff={k:{'expected':v,'actual':actual.get(k)} for k,v in expected.items() if actual.get(k)!=v}
 atomic_json(out/'environment_actual.json',actual)
 if diff:
  atomic_json(out/'environment_mismatch.json',diff);raise RuntimeError('Environment mismatch: '+', '.join(diff))

def load_data():
 manifest=json.loads((ROOT/'data/manifest.json').read_text())
 for fn,key in [('sequences.npz','dataset_sha256'),('metadata.json','metadata_sha256')]:
  assert sha(ROOT/'data'/fn)==manifest[key],fn
 with np.load(ROOT/'data/sequences.npz') as z:data={k:z[k] for k in z.files}
 metadata=json.loads((ROOT/'data/metadata.json').read_text())
 counts=np.bincount(data['question'][data['split']==0],minlength=len(metadata['questions']))
 data['weight']=np.array([1/max(1,counts[q]) for q in data['question']],np.float32)
 # scale cancels in normalized objectives, but avoids tiny floating point sums.
 data['weight']*=max(1,counts[counts>0].mean())
 return data,metadata

def batch(data,chunk_ids,device):
 chunks=data['chunks'][chunk_ids];size=len(chunks);T=64
 index=np.zeros((size,T),np.int64);pad=np.ones((size,T),bool);score=np.zeros((size,T),bool)
 for i,(start,n,warm,ti,sp) in enumerate(chunks):
  index[i,:n]=np.arange(start,start+n);pad[i,:n]=False;score[i,warm:n]=True
 b={k:torch.from_numpy(data[k][index]).to(device) for k in ['x','mode','prev_mode','gate','dest','event_y','event_mask','length_y','solver','bench','weight']}
 for k in ['mode','prev_mode','gate','dest','solver','bench']:b[k]=b[k].long()
 b['pad']=torch.from_numpy(pad).to(device);b['score']=torch.from_numpy(score).to(device);b['index']=index
 return b

class Predictor(nn.Module):
 def __init__(self,variant,meta,width=64):
  super().__init__();self.variant=variant;self.K=len(meta['operators'])
  self.mode=nn.Embedding(self.K+1,16);self.prev=nn.Embedding(self.K+1,8)
  self.solver=nn.Embedding(len(meta['solvers']),8);self.bench=nn.Embedding(len(meta['benchmarks']),4)
  self.proj=nn.Linear(41+36,width)
  if variant.startswith('transformer'):
   layer=nn.TransformerEncoderLayer(width,4,128,.1,batch_first=True,norm_first=True,activation='gelu')
   self.core=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   self.position=nn.Embedding(64,width)
  else:self.core=nn.Sequential(nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
  self.norm=nn.LayerNorm(width);self.gate=nn.Linear(width,3);self.dest=nn.Linear(width,self.K);self.events=nn.Linear(width,6);self.length=nn.Linear(width,1)
 def forward(self,b):
  x=b['x']
  if self.variant=='transformer_no_events':x=torch.cat([x[...,:5],torch.zeros_like(x[...,5:])],-1)
  h=self.proj(torch.cat([x,self.mode(b['mode']),self.prev(b['prev_mode']),self.solver(b['solver']),self.bench(b['bench'])],-1))
  if self.variant.startswith('transformer'):
   n=h.shape[1];h=h+self.position(torch.arange(n,device=h.device))[None]
   mask=torch.triu(torch.ones((n,n),device=h.device,dtype=torch.bool),1)
   h=self.core(h,mask=mask,src_key_padding_mask=b['pad'])
  else:h=self.core(h)
  h=self.norm(h);dest=self.dest(h)
  dest=dest.scatter(-1,b['mode'].unsqueeze(-1),-1e4)
  return dict(gate=self.gate(h),dest=dest,events=self.events(h),length=self.length(h).squeeze(-1))

def losses(o,b):
 mask=b['score'] & (b['gate']>=0);sw=mask & (b['gate']==1);cont=b['score'] & ((b['gate']==0)|(b['gate']==1))
 w=b['weight'];g=F.cross_entropy(o['gate'].reshape(-1,3),b['gate'].clamp_min(0).reshape(-1),reduction='none').reshape_as(w)
 d=F.cross_entropy(o['dest'].reshape(-1,o['dest'].shape[-1]),b['dest'].clamp_min(0).reshape(-1),reduction='none').reshape_as(w)
 joint=((g+d*sw)*w*mask).sum()/(w*mask).sum().clamp_min(1e-8)
 dest=(d*w*sw).sum()/(w*sw).sum().clamp_min(1e-8)
 em=b['event_mask'] & cont.unsqueeze(-1);el=F.binary_cross_entropy_with_logits(o['events'],b['event_y'],reduction='none')
 event=(el*w.unsqueeze(-1)*em).sum()/(w.unsqueeze(-1)*em).sum().clamp_min(1e-8)
 length=(F.smooth_l1_loss(o['length'],b['length_y'],reduction='none')*w*cont).sum()/(w*cont).sum().clamp_min(1e-8)
 return joint+.5*dest+.25*event+.05*length

class Markov:
 def __init__(self,data,meta):
  self.K=len(meta['operators']);self.S=len(meta['solvers']);self.B=len(meta['benchmarks'])
  self.cells=self.K*self.S*self.B;keys=self.keys(data['mode'],data['solver'],data['bench']);train=(data['split']==0)&(data['gate']>=0);w=data['weight']
  self.g=np.zeros((self.cells,3));self.d=np.zeros((self.cells,self.K));self.e=np.zeros((self.cells,6));self.ec=np.zeros((self.cells,6));self.l=np.zeros(self.cells);self.lc=np.zeros(self.cells)
  np.add.at(self.g,(keys[train],data['gate'][train]),w[train])
  sw=train&(data['gate']==1);np.add.at(self.d,(keys[sw],data['dest'][sw]),w[sw])
  co=train&(data['gate']!=2)
  for k in range(6):
   use=co&data['event_mask'][:,k];np.add.at(self.e[:,k],keys[use],w[use]*data['event_y'][use,k]);np.add.at(self.ec[:,k],keys[use],w[use])
  np.add.at(self.l,keys[co],w[co]*data['length_y'][co]);np.add.at(self.lc,keys[co],w[co])
  # Shrink cells to the training distribution for the same current mode.
  for k in range(self.K):
   ids=np.arange(self.cells).reshape(self.K,self.S,self.B)[k].ravel();gp=self.g[ids].sum(0)+.5;gp/=gp.sum();dp=self.d[ids].sum(0)+.5;dp[k]=0;dp/=dp.sum();ep=(self.e[ids].sum(0)+.5)/(self.ec[ids].sum(0)+1);lp=self.l[ids].sum()/max(1e-8,self.lc[ids].sum())
   self.g[ids]=(self.g[ids]+10*gp)/(self.g[ids].sum(1,keepdims=True)+10)
   self.d[ids]=(self.d[ids]+10*dp)/(self.d[ids].sum(1,keepdims=True)+10)
   self.e[ids]=(self.e[ids]+10*ep)/(self.ec[ids]+10);self.l[ids]=(self.l[ids]+10*lp)/(self.lc[ids]+10)
 def keys(self,m,s,b):return (m.astype(np.int64)*self.S+s)*self.B+b
 def predict(self,data,ids):
  k=self.keys(data['mode'][ids],data['solver'][ids],data['bench'][ids])
  return dict(gate=self.g[k],dest=self.d[k],events=self.e[k],length=self.l[k])

def summarize_predictions(data,ids,pred):
 y=data['gate'][ids];sw=y==1;co=y!=2;dy=data['dest'][ids];q=data['question'][ids]
 pg=np.clip(pred['gate'],1e-8,1);pd=np.clip(pred['dest'],1e-8,1);pe=np.clip(pred['events'],1e-8,1-1e-8)
 gn=-np.log(pg[np.arange(len(ids)),y]);dn=-np.log(pd[np.arange(len(ids)),np.maximum(dy,0)])
 joint=gn+sw*dn;mask=data['event_mask'][ids]&co[:,None];ey=data['event_y'][ids]
 eb=-(ey*np.log(pe)+(1-ey)*np.log1p(-pe));ln=np.expm1(np.minimum(pred['length'],15));ly=np.expm1(data['length_y'][ids])
 rows=[]
 for qi in np.unique(q):
  m=q==qi;s=m&sw;c=m&co;e=mask&m[:,None];z=dict(question=int(qi),n=int(m.sum()),switches=int(s.sum()),joint_nll=float(joint[m].mean()),gate_nll=float(gn[m].mean()),
   gate_brier=float(((pg[m]-np.eye(3)[y[m]])**2).sum(1).mean()),end_brier=float(((pg[m,2]-(y[m]==2))**2).mean()))
  if s.any():z.update(destination_nll=float(dn[s].mean()),destination_accuracy=float((pd[s].argmax(1)==dy[s]).mean()),destination_entropy=float((-(pd[s]*np.log(pd[s])).sum(1)).mean()))
  if e.any():z.update(event_bce=float(eb[e].mean()),event_brier=float(((pe-ey)**2)[e].mean()))
  if c.any():z['length_mae_chars']=float(np.abs(ln[c]-ly[c]).mean())
  rows.append(z)
 keys=sorted({k for r in rows for k in r}-{ 'question','n','switches'})
 scores={k:float(np.mean([r[k] for r in rows if k in r])) for k in keys}
 conf=np.zeros((pd.shape[1],pd.shape[1]),int)
 np.add.at(conf,(dy[sw],pd[sw].argmax(1)),1)
 den=conf.sum(0)+conf.sum(1);good=den>0;scores['destination_macro_f1_pooled']=float((2*np.diag(conf)[good]/den[good]).mean()) if good.any() else None
 return dict(scores=scores,question_metrics=rows,rows=len(ids),questions=len(rows),switches=int(sw.sum()),natural_ends=int((y==2).sum()),destination_confusion=conf.tolist())

@torch.no_grad()
def evaluate(model,data,meta,split,device,batch_size=32,save_to=None,progress=None):
 chunks=np.flatnonzero(data['chunks'][:,4]==split);pieces=[];outs={k:[] for k in ['gate','dest','events','length']}
 if isinstance(model,Markov):
  ids=np.flatnonzero((data['split']==split)&(data['gate']>=0));pred=model.predict(data,ids)
 else:
  model.eval()
  for start in range(0,len(chunks),batch_size):
   b=batch(data,chunks[start:start+batch_size],device);o=model(b);m=(b['score']&(b['gate']>=0)).cpu().numpy();pieces.append(b['index'][m])
   for k in outs:
    v=o[k];v=F.softmax(v,-1) if k in ('gate','dest') else torch.sigmoid(v) if k=='events' else v
    outs[k].append(v.cpu().numpy()[m])
   if progress and start%(batch_size*50)==0:progress('evaluation',chunks_done=start,total_chunks=len(chunks))
  ids=np.concatenate(pieces);pred={k:np.concatenate(v) for k,v in outs.items()}
 order=np.argsort(ids);ids=ids[order];pred={k:v[order] for k,v in pred.items()}
 expected=np.flatnonzero((data['split']==split)&(data['gate']>=0));assert np.array_equal(ids,expected),'duplicate or missing test targets'
 result=summarize_predictions(data,ids,pred)
 if save_to:
  tmp=save_to.with_suffix('.tmp.npz');np.savez_compressed(tmp,index=ids,**pred);os.replace(tmp,save_to)
  result['predictions_sha256']=sha(save_to)
 return result

def paired(a,b,key,confidence=.95):
 da={r['question']:r[key] for r in a['question_metrics'] if key in r};db={r['question']:r[key] for r in b['question_metrics'] if key in r};qs=sorted(da.keys()&db.keys())
 v=np.array([db[q]-da[q] for q in qs]);rng=np.random.default_rng(20261007);boots=[]
 for _ in range(2000):boots.append(float(rng.choice(v,len(v),replace=True).mean()))
 alpha=(1-confidence)/2
 return dict(new_minus_base=float(v.mean()),interval=np.quantile(boots,[alpha,1-alpha]).tolist(),confidence=confidence,questions=len(qs))

def train_all():
 configure('mps');check_environment();config=json.loads((ROOT/'config.json').read_text());clock=json.loads((ROOT/'outputs/clock.json').read_text());deadline=clock['deadline'];out=ROOT/'outputs';out.mkdir(exist_ok=True)
 frozen=json.loads((ROOT/'freeze.json').read_text())
 for p,digest in frozen['files'].items():assert sha(ROOT/p)==digest,'changed frozen file '+p
 signature=sha(ROOT/'freeze.json');data,meta=load_data();trainchunks=np.flatnonzero(data['chunks'][:,4]==0)
 def progress(state,**kw):
  atomic_json(out/'progress.json',dict(state=state,time=time.time(),deadline=deadline,**kw))
  if time.time()>deadline-25:raise TimeoutError('immutable batch runtime deadline reached')
 for variant in VARIANTS:
  folder=out/variant;folder.mkdir(exist_ok=True)
  if (folder/'TRAIN_DONE.json').exists():continue
  torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.mps.manual_seed(config['seed']);rng=np.random.default_rng(config['seed']);model=Predictor(variant,meta).to('mps');opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
  params=sum(p.numel() for p in model.parameters());step=0;best=float('inf');order=rng.permutation(trainchunks);pos=0;ckpt=folder/'resume.pt'
  if ckpt.exists():
   state=torch.load(ckpt,map_location='cpu',weights_only=False);assert state['signature']==signature
   model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step'];best=state['best'];order=state['order'];pos=state['position'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.mps.set_rng_state(state['mps_rng'])
   for st in opt.state.values():
    for k,v in st.items():
     if torch.is_tensor(v) and k!='step':st[k]=v.to('mps')
  def checkpoint():
   atomic_torch(ckpt,dict(signature=signature,model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,order=order,position=pos,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))
  progress('training',variant=variant,step=step,parameters=params)
  while step<config['steps']:
   if pos>=len(order):order=rng.permutation(trainchunks);pos=0
   ids=order[pos:pos+config['batch']];pos+=len(ids);b=batch(data,ids,'mps');model.train();opt.zero_grad(set_to_none=True);loss=losses(model(b),b)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
   if step%25==0:progress('training',variant=variant,step=step,total_steps=config['steps'],loss=float(loss.detach().cpu()),parameters=params)
   if step%50==0:checkpoint()
   if step%config['validate_every']==0 or step==config['steps']:
    val=evaluate(model,data,meta,1,'mps',config['batch'],progress=lambda state,**kw:progress(state,variant=variant,step=step,split='validation',**kw));score=val['scores']['joint_nll']
    entry=dict(step=step,training_loss=float(loss.detach().cpu()),validation=val['scores'],time=time.time())
    with (folder/'curve.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    print(variant,step,json.dumps(val['scores']),flush=True)
    if score<best:
     best=score;atomic_torch(folder/'best.pt',dict(model=model.state_dict(),step=step,signature=signature));atomic_json(folder/'best_validation.json',val)
    checkpoint()
  atomic_json(folder/'TRAIN_DONE.json',dict(signature=signature,steps=step,parameters=params,best_validation=best,best_checkpoint_sha256=sha(folder/'best.pt')))
  del model,opt
  torch.mps.empty_cache()
 # No test outcomes have been read to select any checkpoint above.
 results={};baseline=Markov(data,meta)
 results['markov']=evaluate(baseline,data,meta,2,'cpu',save_to=out/'markov_predictions.npz')
 for variant in VARIANTS:
  progress('test_evaluation',variant=variant)
  model=Predictor(variant,meta).to('mps');state=torch.load(out/variant/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
  results[variant]=evaluate(model,data,meta,2,'mps',config['batch'],out/f'{variant}_predictions.npz',lambda state,**kw:progress(state,variant=variant,split='test',**kw));results[variant]['selected_step']=state['step']
  atomic_json(out/f'{variant}_metrics.json',results[variant]);del model;torch.mps.empty_cache()
 contrasts={}
 for base in ['summary_mlp','transformer_no_events','markov']:
  contrasts['transformer_minus_'+base]={k:paired(results[base],results['transformer'],k,.975 if k=='destination_nll' and base!='markov' else .95) for k in ['joint_nll','destination_nll','destination_accuracy','event_bce']}
 summary=dict(status='complete',scope='retrospective annotation dynamics',signature=signature,config=config,data=meta['stats'],results=results,contrasts=contrasts,finished_at=time.time())
 atomic_json(out/'RESULTS.json',summary)
 lines=['Малый трансформер спанов и событий на Heraclides','', 'Результат на отложенных вопросах прежнего корпуса. Ретроспективные метки; не проверка онлайн-детектора или управления.','', 'Метод | joint NLL | NLL нового режима при SWITCH | accuracy нового режима | число вопросов']
 for v,r in results.items():
  s=r['scores'];lines.append(f"{v} | {s['joint_nll']:.5f} | {s['destination_nll']:.5f} | {s['destination_accuracy']:.4f} | {r['questions']}")
 lines+=['','Основные парные сравнения NLL нового режима (отрицательное — улучшение):']
 for k,c in contrasts.items():lines.append(k+' '+json.dumps(c['destination_nll'],ensure_ascii=False))
 lines+=['','Проверены все допустимые тестовые сегменты без повторного счёта.','Один seed; интервалы по вопросам не включают неопределённость переобучения.','END оценивается только там, где источник явно подтверждает естественное окончание.','Вход и прогноз длины измерены в символах, не в токенах.','Сохранены checkpoints, learning curves, полные прогнозы и метрики по вопросам.']
 (out/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n')
 progress('complete',report=str(out/'REPORT_RU.txt'))
 atomic_json(out/'COMPLETE.json',dict(status='complete',signature=signature,expected_variants=VARIANTS,results_sha256=sha(out/'RESULTS.json'),completed_at=time.time()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','environment']);args=ap.parse_args()
 if args.action=='environment':print(json.dumps(environment(),indent=2))
 else:train_all()
