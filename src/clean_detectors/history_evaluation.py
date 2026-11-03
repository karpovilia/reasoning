"""Ordered evaluation, immutable primary outputs, append-only version streams."""
import gzip
import json
import os
from pathlib import Path
import re
import time
import torch
from .core import atomic_json,dumps,file_sha
from .history_rollout import HistoryState,chronological_indices,rollout,version_records
from .model import collate


META=['sample_id','trace_id','sid','question_key','split','event_targets','event_mask',
      'mode_target','mode_mask','benchmark','solver','protocols','origins',
      'trace_segment_count_evaluation_only','finish_reason_evaluation_only','baseline_train_overlap']


def base_record(row,branch='encoder'):
    record={k:row[k] for k in META}
    record['input_token_count']=len(row['tokens'][branch]['input_ids'])
    record['focus_characters']=row['focus_offsets'][1]-row['focus_offsets'][0]
    focus=row['text'][row['focus_offsets'][0]:row['focus_offsets'][1]]
    record['lexical_marker_present']=bool(re.search(r'\b(wait|however|check|verify|verification|alternatively|therefore|thus|mistake|wrong|backtrack)\b',focus,re.I))
    return record


def journal_path(output):
    return output.with_name(output.name.replace('.jsonl.gz','.versions.jsonl.gz'))


@torch.no_grad()
def predict(model,dataset,pad_id,batch_size,output=None,device='cuda'):
    model.eval();records=[];state=HistoryState(model.detector_config['own_history']['capacity'])
    binding=getattr(model,'prediction_binding',None)
    receipt=output.with_name(output.name+'.COMPLETE.json') if output else None
    if receipt and receipt.exists():
        accepted=json.loads(receipt.read_text())
        if binding is None or accepted['binding']!=binding:raise ValueError('Prediction binding changed')
        for name,digest in accepted['files'].items():
            if file_sha(output.parent/name)!=digest:raise ValueError('Accepted prediction file changed')
        with gzip.open(output,'rt') as stream:return [json.loads(line) for line in stream]
    streams=[];primary=None;versions=None
    if output:
        if binding is None:raise ValueError('Persistent predictions require checkpoint binding')
        # Preserve incomplete attempts; never overwrite a published accepted stream.
        for final in (output,journal_path(output)):
            if final.exists():final.rename(final.with_name(final.name+'.interrupted_'+str(time.time_ns())))
            temp=final.with_name(final.name+'.attempt_'+str(time.time_ns()))
            streams.append((gzip.open(temp,'wt',compresslevel=1),temp,final))
        primary,versions=[s[0] for s in streams]
    order=chronological_indices(dataset)
    started=time.monotonic();frames_count=0;revisions_count=0
    try:
        for begin in range(0,len(order),batch_size):
            if getattr(model,'progress_path',None) and begin%(100*batch_size)==0:
                atomic_json(model.progress_path,{'at':time.time(),'stage':'history_evaluation','split':dataset.split,'cursor':begin,'total':len(order)})
            rows=[dataset[int(i)] for i in order[begin:begin+batch_size]]
            tensors=collate(rows,model.branch,pad_id,device)
            with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=torch.device(device).type=='cuda'):
                pooled=model.encode(**tensors);frames=rollout(model,rows,pooled,state)
            metadata={r['sample_id']:base_record(r,model.branch) for r in rows}
            for version in version_records(frames):
                if versions:versions.write(dumps(version)+'\n')
                if version['lag']==0:
                    record={**metadata[version['target_sample_id']],**{k:v for k,v in version.items() if k not in ('trace_id','target_sid','target_sample_id')}}
                    records.append(record);frames_count+=1
                    if primary:primary.write(dumps(record)+'\n')
                else:revisions_count+=1
    finally:
        for stream,_,_ in streams:stream.close()
    if output:
        for _,temp,final in streams:
            with temp.open('rb') as handle:os.fsync(handle.fileno())
            os.replace(temp,final)
        atomic_json(receipt,{'binding':binding,'primary_count':frames_count,'revision_count':revisions_count,
            'seconds':time.monotonic()-started,'batch_size':batch_size,'backbone_passes_per_focus':1,
            'files':{final.name:file_sha(final) for _,_,final in streams}})
    return records


def records_at_lag(primary,output,lag):
    """Exact available lag, never a final-label substitution into initial output."""
    lookup={r['sample_id']:r for r in primary};records=[]
    with gzip.open(journal_path(output),'rt') as stream:
        for line in stream:
            version=json.loads(line)
            if version['lag']!=lag:continue
            original=lookup[version['target_sample_id']]
            record=dict(original)
            record.pop('history_used',None)
            record['history_used_reference']=version['as_of_sample_id']
            record.update({k:v for k,v in version.items() if k not in ('trace_id','target_sid','target_sample_id')})
            records.append(record)
    return records


def write_training_versions(path,versions,epoch,step,parent_checkpoint):
    """One immutable attempt/step shard; failed attempts never overwrite another."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Refusing to overwrite a training version shard')
    temp=path.with_name(path.name+'.tmp')
    with gzip.open(temp,'wt',compresslevel=1) as stream:
        for version in versions:
            stream.write(dumps({'epoch':epoch,'optimizer_step':step,'parent_checkpoint':parent_checkpoint,**version})+'\n')
    with temp.open('rb') as handle:os.fsync(handle.fileno())
    os.replace(temp,path)
    return {'path':str(path),'sha256':file_sha(path),'records':len(versions)}


def latest_records(primary,output):
    lookup={r['sample_id']:dict(r) for r in primary}
    with gzip.open(journal_path(output),'rt') as stream:
        for line in stream:
            version=json.loads(line)
            if version['lag']==0:continue
            record=lookup[version['target_sample_id']]
            record.pop('history_used',None)
            record['history_used_reference']=version['as_of_sample_id']
            record.update({k:v for k,v in version.items() if k not in ('trace_id','target_sid','target_sample_id')})
    return list(lookup.values())
