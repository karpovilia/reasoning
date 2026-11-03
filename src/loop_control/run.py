import os,sys,json,hashlib,time,math,copy,fcntl,subprocess,argparse
from pathlib import Path
ROOT=Path(os.environ.get('REASONING_LOOP_RUN', 'outputs/loops'))
REF=Path(os.environ.get('REASONING_LOOP_WEIGHTS', 'models/loop_source'))
PACK=Path(os.environ.get('REASONING_LOOP_PACK', 'data/local/loop_pack'))
BASE=Path(os.environ.get('REASONING_QWEN_BASE', 'models/base/qwen3-4b-instruct-2507'))
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def t_save(p,x):
 import torch
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 if p.name=='checkpoint.pt' and p.exists() and p.with_suffix('.sha.json').exists():
  if sha(p)==load(p.with_suffix('.sha.json'))['sha256']:
   prev=p.with_name('previous_checkpoint.pt');link=prev.with_suffix('.tmp')
   if link.exists():link.unlink()
   os.link(p,link);os.replace(link,prev);save(prev.with_suffix('.sha.json'),{'sha256':sha(prev)})
 with tmp.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);save(p.with_suffix('.sha.json'),{'sha256':sha(p)})
def t_load(p):
 import torch
 p=Path(p)
 try:assert sha(p)==load(p.with_suffix('.sha.json'))['sha256']
 except (AssertionError,FileNotFoundError,json.JSONDecodeError):
  if p.name=='checkpoint.pt' and p.with_name('previous_checkpoint.pt').exists():return t_load(p.with_name('previous_checkpoint.pt'))
  raise
 return torch.load(p,map_location='cpu',weights_only=False)
def beat(stage,**kw):save(ROOT/'STATUS.json',dict(time=time.time(),stage=stage,pid=os.getpid(),**kw))
def tree_cpu(x):
 import torch
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:tree_cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [tree_cpu(v) for v in x]
 return x

def main():
 import torch,numpy as np
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_float32_matmul_precision('highest')
 clock=ROOT/'CLOCK.json'
 if not clock.exists():save(clock,{'start':time.time(),'max_seconds':1800})
 def budget():
  if time.time()-load(clock)['start']>1800:save(ROOT/'TERMINAL.json',{'status':'INCOMPLETE','reason':'30 minute cumulative runtime budget'});raise SystemExit(0)
 budget();beat('LOAD_FROZEN_MODEL')
 assert torch.cuda.device_count()==1;torch.cuda.set_per_process_memory_fraction(.60)
 for p,h in load(ROOT/'SPEC.json')['pins'].items():assert sha(p)==h,('PIN_CHANGED',p)
 binding=load(REF/'weights/B_CT_seed42/RUN_BINDING.json');cfg=binding['config']
 from ptrain import model_b,model_a
 from ptrain.data import to_device
 from .losses import native_total_loss
 from transformers import AutoTokenizer
 os.environ['PTRAIN_QWEN_DIR']=str(BASE)
 # Reuse the accepted old pack; no source census or namespace redefinition.
 data=model_b.BData(PACK,cfg)
 model=model_b.build(cfg,data.nsolvers,'cuda',namespaces=data.namespaces)
 payload=torch.load(REF/'weights/B_CT_seed42/best.pt',map_location='cpu',weights_only=False);model.load_trainable_state_dict(payload['model']);del payload
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 cached={}
 hook=model.norm.register_forward_pre_hook(lambda mod,args:cached.update(raw=args[0].detach().cpu()))
 loopdata=load(ROOT/'LOOP_DATA.json');heldout={x['question_component'] for x in loopdata['negative_holdout']};positiveq=loopdata['positive_train'][0]['question_component']
 # Freeze exact question membership before inference or optimization.
 inventory={}
 for role in ('TRAIN','DEV'):
  candidate=[];seen=set()
  for wi in data.ids(role):
   ui,lo,hi,sc=data.windows[wi];u=data.units[ui];q=u['question_component']
   if lo!=0 or u['n']>32 or q==positiveq or q in heldout or q in seen:continue
   if not any(data.a[f'ns{h}_vstep'][u['offset']:u['offset']+u['n']].any() for h in range(len(data.namespaces))):continue
   seen.add(q);candidate.append((hashlib.sha256(q.encode()).hexdigest(),wi,u))
  candidate.sort();assert len(candidate)>=16
  inventory[role]=[{'window':wi,'question_component':u['question_component'],'trace_id':u['physical_id'],'segments':u['n']} for _,wi,u in candidate[:16]]
 if (ROOT/'REPLAY_MEMBERSHIP.json').exists():assert inventory==load(ROOT/'REPLAY_MEMBERSHIP.json')
 else:save(ROOT/'REPLAY_MEMBERSHIP.json',inventory)
 assert not ({x['question_component'] for x in inventory['TRAIN']} & ({x['question_component'] for x in inventory['DEV']}|heldout))
 cache=ROOT/'cache';cache.mkdir(exist_ok=True)
 accepted=[n['status']=='accepted' for n in data.namespaces];events=[n.get('event_supervision_eligible',True) for n in data.namespaces]
 timings=[]
 def extract(key,b,meta):
  p=cache/(key+'.pt')
  if p.exists():
   try:return t_load(p)
   except (AssertionError,FileNotFoundError,json.JSONDecodeError):os.replace(p,p.with_name(p.name+'.interrupted'))
  budget();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():o=model(b)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  result={'raw':cached['raw'].clone(),'batch':tree_cpu(b),'meta':meta,'forward_seconds':elapsed};t_save(p,result);timings.append({'key':key,'seconds':elapsed,'positions':int(b['score'].sum()),'tokens':int(b['attention_mask'].sum())});beat('EXTRACT',key=key,seconds=elapsed)
  return result
 old={role:[] for role in ('TRAIN','DEV')}
 for role in old:
  for i,row in enumerate(inventory[role]):old[role].append(extract(f'old_{role}_{i:02d}',data.make_batch([row['window']],'cuda'),row))
 def custom(t,end=None):
  text=t['text'][:end] if end else t['text'];segs=[text[i:i+256] for i in range(0,len(text),256)];ends=[min(i+256,len(text)) for i in range(0,len(text),256)]
  # First 10k characters of the positive and complete <=6k negatives, all chosen before scores.
  q=tok(t['question'],add_special_tokens=False)['input_ids'][:1024];ids=tok(model_b.CHAT_PREFIX,add_special_tokens=False)['input_ids']+q+tok(model_b.CHAT_MID,add_special_tokens=False)['input_ids'];marks=[];trunc=0
  for s in segs:
   z=tok(s+'\n',add_special_tokens=False)['input_ids'];trunc+=len(z)>512;ids+=z[-512:];marks.append(len(ids)-1)
  assert len(ids)<=8192,('LOOP_CONTEXT_TOO_LONG',len(ids));assert trunc==0
  n=len(segs);b={'xin':torch.zeros(1,n,41),'mode':torch.full((1,n),7,dtype=torch.long),'prev_mode':torch.full((1,n),7,dtype=torch.long),'solver':torch.full((1,n),data.manifest['solver_vocab'].index(t['solver']) if t['solver'] in data.manifest['solver_vocab'] else 0,dtype=torch.long),'es_ok':torch.zeros(1,n,dtype=torch.bool),'text':torch.zeros(1,n,384),'prompt':torch.zeros(1,n,384),'pad':torch.zeros(1,n,dtype=torch.bool),'score':torch.ones(1,n,dtype=torch.bool),'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),'marker_pos':torch.tensor([marks])}
  for j,(s,endchar) in enumerate(zip(segs,ends)):b['xin'][0,j,:3]=torch.tensor([math.log1p(j+1)/10,math.log1p(len(s))/10,math.log1p(endchar)/10])
  return to_device(b,'cuda'),ends
 loop=[]
 for i,t in enumerate(loopdata['positive_train']+loopdata['negative_holdout']):
  b,ends=custom(t,10000 if i==0 else None)
  if i==0:
   # Same-prefix invariance: append/change hidden future, keep past side features and tokens exact.
   cut=10;bb={k:v.clone() for k,v in b.items()};pos=int(b['marker_pos'][0,cut]);bb['input_ids'][0,pos+1:]=1;bb['xin'][0,cut+1:]=.123
   with torch.inference_mode():a=model(b);z=model(bb)
   err=max(float((x['gate'][0,:cut+1]-y['gate'][0,:cut+1]).abs().max()) for x,y in zip(a['ns'],z['ns']));save(ROOT/'CAUSAL_SUFFIX_CHECK.json',{'max_abs':err,'tolerance':1e-4});assert err<1e-4
  meta={k:v for k,v in t.items() if k not in ('text','question')};meta.update(char_ends=ends,loop_y=[int(i==0 and e>t['loop_start_char']) for e in ends],token_ends=[len(tok(t['text'][:e],add_special_tokens=False)['input_ids']) for e in ends],onset_token=len(tok(t['text'][:t['loop_start_char']],add_special_tokens=False)['input_ids']) if i==0 else None)
  loop.append(extract(f'loop_{i:02d}',b,meta))
 hook.remove();save(ROOT/'TIMINGS.json',timings)
 class Heads(torch.nn.Module):
  def __init__(self):
   super().__init__();self.norm=copy.deepcopy(model.norm).cpu();self.down=copy.deepcopy(model.down).cpu();self.ns_heads=copy.deepcopy(model.ns_heads).cpu();self.ns_dims=model.ns_dims;self.loop=torch.nn.Linear(cfg['model']['head_width'],1)
   for p in self.parameters():p.requires_grad_(True)
  def forward(self,raw):
   h=torch.nn.functional.gelu(self.down(self.norm(raw)));o=model_a.native_outputs(self,h);o['loop']=self.loop(h).squeeze(-1);return o
 heads=Heads();del model;torch.cuda.empty_cache()
 optim=torch.optim.AdamW(heads.parameters(),lr=3e-4,weight_decay=.01)
 def oldloss(item):return native_total_loss(heads(item['raw']),item['batch'],accepted,event_eligible=events,group1=False)[0]
 def evaluate_old():
  with torch.no_grad():return float(torch.stack([oldloss(x) for x in old['DEV']]).mean())
 def old_components():
  totals={k:[0.,0] for k in ('gate_ce','destination_ce','event_bce')}
  with torch.no_grad():
   for item in old['DEV']:
    b=item['batch'];o=heads(item['raw'])
    for h,(t,out) in enumerate(zip(b['ns'],o['ns'])):
     if not accepted[h]:continue
     for key,values,mask in (
       ('gate_ce',torch.nn.functional.cross_entropy(out['gate'].reshape(-1,3),t['gate'].clamp_min(0).reshape(-1),reduction='none').reshape(b['score'].shape),b['score']&t['vstep']),
       ('destination_ce',torch.nn.functional.cross_entropy(out['dest'].reshape(-1,out['dest'].shape[-1]),t['dest'].clamp(0,out['dest'].shape[-1]-1).reshape(-1),reduction='none').reshape(b['score'].shape),b['score']&t['vnt']),
       ('event_bce',torch.nn.functional.binary_cross_entropy_with_logits(out['ev'],t['ev'],reduction='none'),b['score'][...,None]&t['vev']&bool(events[h]))):
      totals[key][0]+=float(values[mask].sum());totals[key][1]+=int(mask.sum())
  return {k:{'mean':v[0]/v[1] if v[1] else None,'labeled_cells':v[1]} for k,v in totals.items()}
 components_before=old_components();before=evaluate_old();startstep=0
 if (ROOT/'checkpoint.pt').exists():
  ck=t_load(ROOT/'checkpoint.pt');heads.load_state_dict(ck['heads']);optim.load_state_dict(ck['optimizer']);startstep=ck['step'];before=ck['old_dev_before'];components_before=ck['components_before'];torch.set_rng_state(ck['rng'])
 else:t_save(ROOT/'initial_heads.pt',heads.state_dict())
 history=[];y=torch.tensor(loop[0]['meta']['loop_y'],dtype=torch.float32)[None]
 for step in range(startstep,32):
  budget();heads.train();optim.zero_grad();lo=oldloss(old['TRAIN'][step%16]);logit=heads(loop[0]['raw'])['loop'];bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none');ll=.5*bce[y==0].mean()+.5*bce[y==1].mean();total=lo+ll;total.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),1.);optim.step()
  history.append({'step':step+1,'old_loss':float(lo.detach()),'loop_loss':float(ll.detach()),'sum':float(total.detach())});t_save(ROOT/'checkpoint.pt',{'heads':heads.state_dict(),'optimizer':optim.state_dict(),'step':step+1,'old_dev_before':before,'components_before':components_before,'rng':torch.get_rng_state()});beat('MULTITASK',step=step+1)
 heads.eval();after=evaluate_old();save(ROOT/'LOSS_LAST_RESUME.json',history)
 from . import audit_baseline
 predictions=[];summary=[]
 for item in loop:
  with torch.no_grad():ps=torch.sigmoid(heads(item['raw'])['loop'])[0].tolist()
  meta=item['meta'];t=next(t for t in loopdata['positive_train']+loopdata['negative_holdout'] if t['trace_id']==meta['trace_id']);base=[];baseline_seconds=0
  for end in meta['char_ends']:
   start=time.perf_counter();text=t['text'][:end];signal=bool(audit_baseline.exact_loops(text,list(audit_baseline.TOK.finditer(text))) or audit_baseline.repeated_sentences(text));baseline_seconds+=time.perf_counter()-start;base.append(signal)
  for end,nt,y_,prob,b_ in zip(meta['char_ends'],meta['token_ends'],meta['loop_y'],ps,base):predictions.append({'trace_id':meta['trace_id'],'question_component':meta['question_component'],'role':meta['role'],'end_char':end,'controller_token_count':nt,'loop_y':y_,'p_loop':prob,'multitask_alarm':prob>=.5,'baseline_alarm':b_})
  sm={'trace_id':meta['trace_id'],'role':meta['role'],'positions':len(ps),'positive_positions':sum(meta['loop_y']),'baseline_cpu_seconds':baseline_seconds,'frozen_forward_seconds':item['forward_seconds']}
  for name,sigs in [('multitask',[p>=.5 for p in ps]),('baseline',base)]:
   alarm=next((j for j,x in enumerate(sigs) if x),None);sm[name+'_first_alarm_char']=meta['char_ends'][alarm] if alarm is not None else None;sm[name+'_false_positive_positions']=sum(a and not y for a,y in zip(sigs,meta['loop_y']));sm[name+'_detected_positive']=any(a and y for a,y in zip(sigs,meta['loop_y']));sm[name+'_first_alarm_delay_controller_tokens']=(meta['token_ends'][alarm]-meta['onset_token']) if alarm is not None and meta['onset_token'] is not None else None
  summary.append(sm)
 save(ROOT/'PREDICTIONS.json',predictions);save(ROOT/'RESULTS.json',{'status':'EXPLORATORY_MULTITASK_COMPLETE_NO_POSITIVE_HOLDOUT','positive_train_questions':1,'positive_holdout_questions':0,'negative_holdout_questions':len(loop)-1,'steps':32,'seed':42,'old_dev_questions':16,'old_components_before':components_before,'old_components_after':old_components(),'old_native_loss_before':before,'old_native_loss_after':after,'old_native_loss_delta':after-before,'by_trace':summary,'parameters_trained':sum(p.numel() for p in heads.parameters()),'checkpoint':str(ROOT/'checkpoint.pt'),'limitation':'No independent positive question. Positive detection is resubstitution, not generalization; negative judgments are model-based; no semantic-loop claim; no intervention or token-saving claim.'})
 save(ROOT/'TERMINAL.json',{'status':'COMPLETE_LIMITED_EXPERIMENT','time':time.time(),'elapsed':time.time()-load(clock)['start']});beat('COMPLETE_LIMITED_EXPERIMENT')
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  save(ROOT/'ERROR.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
