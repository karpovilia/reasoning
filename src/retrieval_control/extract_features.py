"""Causal saved-prefix features using frozen DetectorV1 and MiniLM. Run on allocated GPU."""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.util, json, math, os, re, sys, time
from pathlib import Path

EVENTS=['backtrack','verify','branch','subgoal_done','failed_attempt','commit']
MODES=['SETUP','DERIVING','EXPLORING','VERIFYING','REVISING','CONCLUDING']
SEGMENT_CHARS=768
def sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def chunks(text,max_chars=SEGMENT_CHARS):
    """Only completed message text is segmented; no future response is accessible."""
    out=[]
    for line in re.split(r'\n\s*\n',text):
        line=line.strip()
        while len(line)>max_chars:
            cut=line.rfind(' ',0,max_chars+1)
            if cut<max_chars//2:cut=max_chars
            out.append(line[:cut]);line=line[cut:].strip()
        if line:out.append(line)
    return out
def input_texts(case):
    reasoning=[];visible=[]
    for m in case['messages']:
        if m.get('role') in ['system','user']:continue
        r=str(m.get('reasoning_content') or m.get('reasoning') or '')
        c=m.get('content') or '';c=c if isinstance(c,str) else json.dumps(c,ensure_ascii=False)
        if m.get('role')=='assistant':
            parts=[r,c] if r and c and r!=c else [r or c]
            for x in parts:
                reasoning.extend(chunks(x));visible.extend(chunks(x))
            if m.get('tool_calls'):
                visible.extend(chunks(json.dumps(m['tool_calls'],ensure_ascii=False,sort_keys=True)))
        elif m.get('role')=='tool':visible.extend(chunks(c))
    return reasoning,visible
def causal_snapshot_features(outputs,segments):
    """Apply revisions available at this decision only. Empty values have explicit masks."""
    latest={};out_names=[];out_values=[];out_masks=[]
    def put(n,v):
        out_names.append(n);out_masks.append(int(v is not None));out_values.append(0.0 if v is None else float(v))
    for sid,x in enumerate(outputs):
        if x['sid']!=sid:raise ValueError('out-of-order detector response')
        if x.get('primary'):latest[sid]=x['primary']
        for r in x.get('revisions',[]):
            if r['as_of_sid']!=sid or r['target_sid']>sid:raise ValueError('future revision')
            latest[r['target_sid']]=r
    end=len(outputs)-1;cur=latest.get(end)
    for e in EVENTS:put('event_probability_'+e,cur['event_probabilities'][e] if cur else None)
    for m in MODES:put('mode_probability_'+m,cur['mode_probabilities'][m] if cur else None)
    modes={s:max(MODES,key=lambda m:v['mode_probabilities'][m]) for s,v in latest.items()}
    start=end
    if end in modes:
        while start-1 in modes and modes[start-1]==modes[end]:start-=1
    put('log1p_current_span_segments',math.log1p(end-start+1) if cur else None)
    put('log1p_current_span_chars',math.log1p(sum(map(len,segments[start:end+1]))) if cur else None)
    put('current_span_left_censored',int(start>0 and start-1 not in modes) if cur else None)
    put('log1p_observed_mode_changes',math.log1p(sum(s-1 in modes and modes[s]!=modes[s-1] for s in modes)))
    put('observer_current_available',int(cur is not None))
    put('observer_available_fraction',len(latest)/len(outputs) if outputs else 0)
    return {'names':out_names,'values':out_values,'mask':out_masks,'as_of_sid':end}

class MiniLMEncoder:
    """SentenceTransformer all-MiniLM-L6-v2 mean pooling + L2, without extra runtime package."""
    def __init__(self,path):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).to('cpu').eval()
    def encode(self,texts,batch_size=32,**unused):
        import torch,numpy as np
        rows=[]
        with torch.no_grad():
            for first in range(0,len(texts),batch_size):
                tokens=self.tokenizer(texts[first:first+batch_size],padding=True,truncation=True,max_length=256,return_tensors='pt')
                hidden=self.model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                vectors=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
                rows.append(torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().numpy())
        return np.concatenate(rows,axis=0)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2');ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    import torch
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('frozen_detector_api_v1',args.bundle/'api/detector_api_v1.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    cases=[json.loads(x) for x in args.cases.open() if x.strip()];cases=[x for x in cases if not args.smoke_only or x.get('smoke')]
    args.output.mkdir(parents=True,exist_ok=True)
    detector=api.DetectorV1(args.bundle/'manifest.json',device=args.device,verify_base=True)
    # Encoder stays on CPU: no simultaneous GPU placement assumption and separately measured cost.
    encoder=MiniLMEncoder(args.encoder)
    metadata={'schema':'web_prefix_features_v1','bundle_id':detector.bundle_id,'bundle_api_sha256':detector.api_sha256,'encoder':args.encoder,'encoder_device':'cpu','detector_device':args.device,'segmentation':'completed assistant message paragraphs; deterministic max768-character whitespace chunks','T':'MiniLM question, last visible chunk, mean of all visible assistant/tool chunks; each384d','C':'log1p chars, reasoning segments, tool calls, tool replies;2 rounds;4 future rounds;2048 turn budget','ES':'current6 event and6 mode probabilities; observed span age/changes and availability with explicit masks; latest revisions as of current prefix','case_file_sha256':hashlib.sha256(args.cases.read_bytes()).hexdigest(),'expected_cases':len(cases)}
    old=args.output/'RUN.json'
    if old.exists() and json.loads(old.read_text())!=metadata:raise ValueError('feature run configuration changed')
    atomic(old,metadata);complete=0
    for case in cases:
        key=hashlib.sha256(case['case_id'].encode()).hexdigest()[:24];dest=args.output/(key+'.json')
        if dest.exists():
            prior=json.loads(dest.read_text());assert prior['prefix_sha256']==case['prefix_sha256'] and prior['bundle_id']==detector.bundle_id
            complete+=1;continue
        assert sha(case['messages'])==case['prefix_sha256']
        segments,visible=input_texts(case);start=time.monotonic();cpu_start=time.process_time();torch.cuda.synchronize() if str(args.device).startswith('cuda') else None
        detector.reset(case['case_id'],case['question']);outputs=[]
        for seg in segments:outputs.append(detector.observe(seg))
        torch.cuda.synchronize() if str(args.device).startswith('cuda') else None
        detector_wall=time.monotonic()-start;detector_cpu=time.process_time()-cpu_start
        es=causal_snapshot_features(outputs,segments);t0=time.monotonic();tcpu=time.process_time()
        texts=[case['question']]+(visible or ['']);vectors=encoder.encode(texts,batch_size=32,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
        text_features=vectors[0].tolist()+vectors[-1].tolist()+vectors[1:].mean(axis=0).tolist()
        encoder_wall=time.monotonic()-t0;encoder_cpu=time.process_time()-tcpu
        chars=sum(len(x) for x in visible);tool_replies=sum(m.get('role')=='tool' for m in case['messages'])
        C={'names':['log1p_visible_chars','log1p_assistant_segments','log1p_tool_calls','log1p_tool_replies','completed_rounds','remaining_rounds','turn_token_cap'],'values':[math.log1p(chars),math.log1p(len(segments)),math.log1p(case['prefix_tool_calls']),math.log1p(tool_replies),2.,4.,2048.]}
        result={'status':'COMPLETE','case_id':case['case_id'],'benchmark':case['benchmark'],'question_id':case['question_id'],'role':case['role'],'smoke':case.get('smoke',False),'prefix_sha256':case['prefix_sha256'],'bundle_id':detector.bundle_id,'C':C,'T':text_features,'ES':es,'segments':segments,'observer_outputs':outputs,'observer_view':detector.view(end_of_trace=False),'times':{'detector_wall_s':detector_wall,'detector_process_cpu_s':detector_cpu,'encoder_wall_s':encoder_wall,'encoder_process_cpu_s':encoder_cpu},'completed_at':datetime.datetime.now().astimezone().isoformat()}
        atomic(dest,result);complete+=1;atomic(args.output/'HEARTBEAT.json',{'complete':complete,'expected':len(cases),'case_id':case['case_id'],'at':datetime.datetime.now().astimezone().isoformat()});print(json.dumps({'complete':complete,'expected':len(cases),'detector_s':round(detector_wall,3),'encoder_s':round(encoder_wall,3)}),flush=True)
    atomic(args.output/'COMPLETE.json',{'status':'COMPLETE','cases':complete,'expected':len(cases),'run_sha256':hashlib.sha256(old.read_bytes()).hexdigest()})
if __name__=='__main__':main()
