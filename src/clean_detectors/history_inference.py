"""One observed trace at a time, original +0 answers plus explicit revisions."""
from pathlib import Path
import json
import os
from copy import deepcopy
import time
import numpy as np
import torch
from .core import atomic_json,shared_input,sha,dumps,file_sha
from .model import build_model,collate
from .history_rollout import HistoryState,rollout,version_records
from .spans import predicted_spans


class HistoryOnlineDetector:
    def __init__(self,checkpoint,frozen,device='cuda'):
        from transformers import AutoTokenizer
        self.checkpoint=Path(checkpoint);self.frozen=Path(frozen);self.device=device
        self.config=json.loads((self.checkpoint/'detector_config.json').read_text())
        self.model=build_model(self.config,checkpoint=self.checkpoint).to(device).eval()
        self.tokenizers={name:AutoTokenizer.from_pretrained(self.frozen/'tokenizers'/name,local_files_only=True)
                         for name in ('encoder','lora4b')}
        self.reset('unstarted','')

    def reset(self,trace_id,question):
        self.trace_id=trace_id;self.question=question;self.segments=[]
        self.state=HistoryState(self.config['own_history']['capacity'])
        self.primary=[];self.latest={}

    @torch.no_grad()
    def push(self,segment,journal_directory=None):
        sid=len(self.segments)
        if journal_directory is not None and (Path(journal_directory)/f'step_{sid:09d}.json').exists():
            raise ValueError('Journal is ahead of session state; reconcile before advancing')
        self.segments.append(segment)
        inp=shared_input(self.question,self.segments,sid,self.tokenizers)
        if inp['status']!='eligible':
            self.primary.append({'sid':sid,'status':inp['status']})
            self.state=HistoryState(self.config['own_history']['capacity'])
            return {'status':inp['status'],'sid':sid,'versions':[]}
        row={'sample_id':f'{self.trace_id}:{sid}','trace_id':self.trace_id,'sid':sid,**inp}
        tensors=collate([row],self.model.branch,self.tokenizers[self.model.branch].pad_token_id,self.device)
        with torch.autocast(device_type=torch.device(self.device).type,dtype=torch.bfloat16,
                            enabled=torch.device(self.device).type=='cuda'):
            frames=rollout(self.model,[row],self.model.encode(**tensors),self.state)
        versions=list(version_records(frames));self.primary.append(deepcopy(versions[0]))
        for version in versions:self.latest[version['target_sid']]=deepcopy(version)
        result={'status':'eligible','sid':sid,'versions':versions}
        if journal_directory is not None:
            path=Path(journal_directory)/f'step_{sid:09d}.json'
            if path.exists():raise ValueError('Append-only inference journal already contains this step')
            atomic_json(path,result)
        return result

    def snapshot(self,end_of_trace=False):
        ids=list(range(len(self.segments)))
        modes=[int(np.argmax(self.latest[sid]['mode_probabilities'])) if sid in self.latest else None for sid in ids]
        return {'trace_id':self.trace_id,'as_of_sid':len(self.segments)-1,'primary':deepcopy(self.primary),
                'latest':[deepcopy(self.latest.get(sid)) for sid in ids],
                'spans':predicted_spans(ids,modes,end_of_trace), 'end_of_trace':end_of_trace}


    def checkpoint_binding(self):
        paths=[self.checkpoint/'heads.pt',self.checkpoint/'detector_config.json']+sorted(p for p in (self.checkpoint/'backbone').rglob('*') if p.is_file())
        return {str(p.relative_to(self.checkpoint)):file_sha(p) for p in paths}

    def save_session(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp')
        torch.save({'checkpoint_binding':self.checkpoint_binding(),
            'trace_id':self.trace_id,'question':self.question,'segments':self.segments,
            'primary':self.primary,'latest':self.latest,'history':self.state.snapshot()},temporary)
        with temporary.open('rb') as handle:os.fsync(handle.fileno())
        os.replace(temporary,path)
        atomic_json(path.with_name(path.name+'.COMPLETE.json'),{'sha256':file_sha(path)})

    def restore_session(self,path):
        path=Path(path);receipt=json.loads(path.with_name(path.name+'.COMPLETE.json').read_text())
        if file_sha(path)!=receipt['sha256']:raise ValueError('Inference state changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if self.checkpoint_binding()!=state['checkpoint_binding']:raise ValueError('Inference model binding differs')
        self.trace_id=state['trace_id'];self.question=state['question'];self.segments=state['segments']
        self.primary=state['primary'];self.latest=state['latest']
        self.state=HistoryState.restore(state['history'],self.device)


def benchmark(detector,traces,path):
    durations=[];trace_times=[];attempted=0;revisions=0
    if str(detector.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for i,trace in enumerate(traces):
        detector.reset(f'cost:{i}',trace['question']);start_trace=time.perf_counter()
        for segment in trace['segments']:
            if str(detector.device).startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter();result=detector.push(segment)
            if str(detector.device).startswith('cuda'):torch.cuda.synchronize()
            durations.append(time.perf_counter()-start);attempted+=1
            revisions+=max(0,len(result['versions'])-1)
        detector.snapshot(end_of_trace=False);trace_times.append(time.perf_counter()-start_trace)
    result={'attempted_focus':attempted,'revision_count':revisions,'trace_count':len(traces),
            'focus_p50_seconds':float(np.quantile(durations,.5)),'focus_p95_seconds':float(np.quantile(durations,.95)),
            'trace_seconds':trace_times,'throughput_focus_per_second':attempted/sum(trace_times),
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if str(detector.device).startswith('cuda') else None,
            'includes':['serialization','both_tokenizers','joint_frozen_context_fit','device_transfer','backbone',
                        'own_history','all_bounded_revisions','version_records','final_span_reconstruction'],
            'model_branch':detector.model.branch,'excludes':['network','journal disk persistence'],'batch_size':1}
    atomic_json(path,result);return result
