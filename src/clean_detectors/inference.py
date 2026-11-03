"""Prefix-only inference with events AND mode spans, including online open spans."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import MODES,shared_input,spans_from_modes,atomic_json
from .model import build_model
from .spans import predicted_spans


class OnlineDetector:
    def __init__(self, checkpoint, frozen, thresholds_path, device='cuda'):
        from transformers import AutoTokenizer
        checkpoint=Path(checkpoint);frozen=Path(frozen)
        self.config=json.loads((checkpoint/'detector_config.json').read_text())
        if self.config.get('own_history'):
            raise ValueError('Use HistoryOnlineDetector for chronological history and explicit revisions')
        self.tokenizers={name:AutoTokenizer.from_pretrained(frozen/'tokenizers'/name,local_files_only=True)
                         for name in ['encoder','lora4b']}
        self.calibration=json.loads(Path(thresholds_path).read_text())
        self.model=build_model(self.config,checkpoint=checkpoint).to(device).eval()
        self.device=device;self.branch=self.config['branch']

    def predict(self, items):
        rows=[];positions=[];results=[]
        for i,item in enumerate(items):
            # Only observed_segments is accepted; no future/full-trace fields are consumed.
            segments=item['observed_segments'];sid=len(segments)-1
            inp=shared_input(item['question'],segments,sid,self.tokenizers)
            results.append({'sample_id':item['sample_id'],'status':inp['status'],'sid':sid})
            if inp['status']=='eligible':
                rows.append({'sample_id':item['sample_id'],**inp});positions.append(i)
        if rows:
            out=self.model.predict(rows,self.tokenizers[self.branch].pad_token_id,self.device)
            ep=out['event_logits'].sigmoid().tolist();mp=out['mode_logits'].softmax(-1).tolist()
            for i,e,m,el,ml in zip(positions,ep,mp,out['event_logits'].tolist(),out['mode_logits'].tolist()):
                results[i].update(event_logits=el,mode_logits=ml,event_probabilities=e,mode_probabilities=m,
                                  event_predictions=[p>=t for p,t in zip(e,self.calibration['f1'])],
                                  mode=MODES[int(np.argmax(m))])
        return results

    def trace(self, question, observed_segments, end_of_trace=False, batch_size=1):
        """Set end_of_trace only for a confirmed natural end, never a token cutoff."""
        results=[]
        for begin in range(0,len(observed_segments),batch_size):
            items=[{'sample_id':str(i),'question':question,'observed_segments':observed_segments[:i+1]}
                   for i in range(begin,min(begin+batch_size,len(observed_segments)))]
            results.extend(self.predict(items))
        spans=predicted_spans(list(range(len(results))),[r.get('mode') for r in results],end_of_trace)
        return {'segments':results,'spans':spans,'end_of_observed_prefix':len(observed_segments),
                'end_of_trace':end_of_trace}


def benchmark(detector,traces,output,batches=(1,8),warmup=3):
    """Includes serialization, BOTH tokenizers, transfer, model and heads; no cache."""
    results={}
    for batch in batches:
        focuses=[];trace_times=[];eligible=0;start_all=time.perf_counter()
        items=[{'sample_id':f'{tid}:{i}','question':t['question'],'observed_segments':t['segments'][:i+1]}
               for tid,t in enumerate(traces) for i in range(len(t['segments']))]
        for _ in range(warmup):detector.predict(items[:batch])
        if str(detector.device).startswith('cuda'):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start_all=time.perf_counter()
        for t in traces:
            start_trace=time.perf_counter()
            for begin in range(0,len(t['segments']),batch):
                current=[{'sample_id':str(i),'question':t['question'],'observed_segments':t['segments'][:i+1]}
                         for i in range(begin,min(begin+batch,len(t['segments'])))]
                start=time.perf_counter();out=detector.predict(current)
                if str(detector.device).startswith('cuda'):torch.cuda.synchronize()
                elapsed=time.perf_counter()-start;focuses.append(elapsed/len(current))
                eligible+=sum(r['status']=='eligible' for r in out)
            trace_times.append(time.perf_counter()-start_trace)
        total=time.perf_counter()-start_all
        results[str(batch)]={'batch':batch,'focus_p50_seconds':float(np.quantile(focuses,.5)),
          'focus_p95_seconds':float(np.quantile(focuses,.95)),
          'focus_latency_note':'For batch>1, batch elapsed divided by its size (amortized latency)',
          'trace_p50_seconds':float(np.quantile(trace_times,.5)),'trace_p95_seconds':float(np.quantile(trace_times,.95)),
          'throughput_focus_per_second':len(items)/total,'eligible':eligible,'attempted':len(items),
          'peak_gpu_bytes':torch.cuda.max_memory_allocated() if str(detector.device).startswith('cuda') else None}
    atomic_json(output,{'results':results,'warmup_calls':warmup,'device':str(detector.device),'dtype':'bf16',
                       'includes':['serialization','joint_context_fit','both_tokenizers','device_transfer','backbone','heads'],
                       'trace_count':len(traces)})
    return results
