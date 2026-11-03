"""Single-GPU supervised training. Invoked only by an authorized box supervisor."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import time
import uuid
import zlib

import numpy as np
import torch

from .core import atomic_json, dumps, file_sha, spans_from_modes
from .dataset import Dataset, verify_manifest
from .metrics import evaluate, span_metrics, thresholds
from .model import build_model, collate, loss_denominators, masked_loss, parameter_report, training_loss, training_denominators, head_parameters
from .spans import predicted_spans

STOP = False


def interrupted(*_):
    global STOP
    STOP = True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(model, optimizer, scheduler, output, state, binding):
    checkpoint=output/'checkpoints'/f"step_{state['step']:09d}_{uuid.uuid4().hex[:8]}"
    if state.pop('_saving_best',False):state['best_checkpoint']=str(checkpoint)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    stage=checkpoint.with_name(checkpoint.name+'.staging')
    if stage.exists():shutil.rmtree(stage)
    model.save(stage)
    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':rng_state(),'state':state,'binding':binding,
                'history':model.training_history.snapshot() if getattr(model,'training_history',None) is not None else None},stage/'resume.pt')
    # Persist tensor bytes and nested directory entries before publishing a
    # checkpoint marker/pointer; a host outage must not publish an empty file.
    for path in stage.rglob('*'):
        if path.is_file():
            with path.open('rb') as handle:os.fsync(handle.fileno())
    for directory in sorted((p for p in stage.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    atomic_json(stage/'COMPLETE.json',{'binding':binding,'step':state['step'],
      'files':{str(p.relative_to(stage)):file_sha(p) for p in stage.rglob('*') if p.is_file()}})
    os.replace(stage,checkpoint)
    fd=os.open(checkpoint.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    atomic_json(output/'latest.json',{'checkpoint':str(checkpoint),'step':state['step']})
    return checkpoint


def select_final_checkpoint(model,optimizer,scheduler,output,state,binding,policy):
    # Keep the SAME selected checkpoint when recovery resumes final evaluation.
    if state.get('training_complete'):
        path=Path(state['best_checkpoint'])
        load_resume(path,binding)
    else:
        state['training_complete']=True;state['_saving_best']=True
        path=save_checkpoint(model,optimizer,scheduler,output,state,binding)
    atomic_json(output/'selected_final.json',{'checkpoint':str(path),'step':state['step'],
        'binding':binding,'policy':policy,'validation_used_for_checkpoint_selection':False})
    return path


def load_resume(checkpoint, binding):
    complete=json.loads((checkpoint/'COMPLETE.json').read_text())
    if complete['binding']!=binding:raise ValueError('checkpoint experiment binding differs')
    for p,digest in complete['files'].items():
        if file_sha(checkpoint/p)!=digest:raise ValueError('incomplete or corrupted checkpoint: '+p)
    saved=torch.load(checkpoint/'resume.pt',map_location='cpu',weights_only=False)
    tip=saved['state'].get('history_journal_tip')
    if tip and file_sha(tip['path'])!=tip['sha256']:raise ValueError('History journal at accepted checkpoint changed')
    return saved


@torch.no_grad()
def predict_dataset(model,dataset,pad_id,batch_size,output=None):
    if model.detector_config.get('own_history'):
        from .history_evaluation import predict
        return predict(model,dataset,pad_id,batch_size,output)
    records=[];model.eval()
    stream=gzip.open(output.with_suffix(output.suffix+'.tmp'),'wt') if output else None
    try:
        for start in range(0,len(dataset),batch_size):
            if start%max(batch_size,100*batch_size)==0 and getattr(model,'progress_path',None):
                atomic_json(model.progress_path,{'at':time.time(),'stage':'evaluation','split':dataset.split,
                            'cursor':start,'total':len(dataset)})
            batch=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
            result=model.predict(batch,pad_id,'cuda')
            ep=result['event_logits'].sigmoid().tolist();mp=result['mode_logits'].softmax(-1).tolist()
            for index,(r,e,m,el,ml) in enumerate(zip(batch,ep,mp,result['event_logits'].tolist(),result['mode_logits'].tolist())):
                record={k:r[k] for k in ['sample_id','trace_id','sid','question_key','split','event_targets',
                          'event_mask','mode_target','mode_mask','benchmark','solver','protocols','origins',
                          'trace_segment_count_evaluation_only','finish_reason_evaluation_only','baseline_train_overlap']}
                record.update(event_probabilities=e,mode_probabilities=m,event_logits=el,mode_logits=ml)
                if 'absence_logits' in result:
                    record.update(absence_probability=float(result['absence_logits'][index].sigmoid()),
                                  presence_probability=float(result['presence_logits'][index].sigmoid()))
                record['input_token_count']=len(r['tokens'][model.branch]['input_ids'])
                record['focus_characters']=r['focus_offsets'][1]-r['focus_offsets'][0]
                focus=r['text'][r['focus_offsets'][0]:r['focus_offsets'][1]]
                record['lexical_marker_present']=bool(re.search(
                    r"\b(wait|however|check|verify|verification|alternatively|therefore|thus|mistake|wrong|backtrack)\b",
                    focus,re.I))
                records.append(record)
                if stream:stream.write(dumps(record)+'\n')
    finally:
        if stream:stream.close()
    if output:os.replace(output.with_suffix(output.suffix+'.tmp'),output)
    return records


def arrays(records):
    return [np.asarray([r[k] for r in records]) for k in ['event_targets','event_mask','event_probabilities',
            'mode_target','mode_mask','mode_probabilities']]


def metric_report(records,threshold):
    result=evaluate(*arrays(records),threshold)
    result['spans']=span_metrics(records)
    count=Counter(r['question_key'] for r in records)
    result['equal_question_event_confusion']=evaluate(*arrays(records),threshold,
                            weights=[1/count[r['question_key']] for r in records])
    result['strata']={}
    keys={
        'benchmark':lambda r:r['benchmark'],
        'solver':lambda r:r['solver'],
        'protocol':lambda r:','.join(r['protocols']),
        'teacher_layers':lambda r:','.join(sorted({x['teacher'] for x in r['origins'] if x.get('teacher')})),
        'trace_length_segments':lambda r:'short_0_127' if r['trace_segment_count_evaluation_only']<128 else
            'medium_128_511' if r['trace_segment_count_evaluation_only']<512 else 'long_512_plus',
        'focus_characters':lambda r:'short_0_255' if r['focus_characters']<256 else
            'medium_256_1023' if r['focus_characters']<1024 else 'long_1024_plus',
        'trace_finish_reason':lambda r:str(r['finish_reason_evaluation_only'])}
    keys['lexical_marker_present']=lambda r:str(r['lexical_marker_present'])
    keys['focus_is_final_segment']=lambda r:str(r['sid']+1==r['trace_segment_count_evaluation_only'])
    for field,key in keys.items():
        groups={}
        for r in records:groups.setdefault(key(r),[]).append(r)
        for value,rows in groups.items():
            result['strata'][field+'|'+value]=evaluate(*arrays(rows),threshold)
    return result


def write_spans(records,path):
    groups={}
    for r in records:groups.setdefault(r['trace_id'],[]).append(r)
    with gzip.open(path,'wt') as f:
        for tid,rows in sorted(groups.items()):
            rows.sort(key=lambda r:r['sid'])
            ids=[r['sid'] for r in rows]
            reason=rows[-1]['finish_reason_evaluation_only']
            end=ids[-1]+1==rows[-1]['trace_segment_count_evaluation_only'] and reason=='stop'
            spans=predicted_spans(ids,[int(np.argmax(r['mode_probabilities'])) for r in rows],end)
            f.write(dumps({'trace_id':tid,'spans':spans,'finish_reason':reason,
                'scope':'available evaluated segments; missing segments and non-EOS tails are censored'})+'\n')


def train(config, frozen, output, runtime):
    if not torch.cuda.is_available():raise RuntimeError('CUDA device required')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable; no silent precision change')
    if runtime.get('runtime_deadline_epoch', 0) <= time.time():
        raise ValueError('runtime_deadline_epoch must be a future Unix timestamp')
    output.mkdir(parents=True,exist_ok=True)
    manifest=verify_manifest(frozen)
    binding={'config_sha256':file_sha(config),'manifest_sha256':file_sha(frozen/'manifest.json'),
             'plan_sha256':runtime.get('plan_sha256', file_sha(config))}
    cfg=json.loads(config.read_text());seed=cfg['seed']
    from .scope_policy import validation_steps, fixed_final
    fixed_final(cfg)  # Validate the frozen selection policy before model loading.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    train_ds=Dataset(frozen,'train')
    if cfg.get('selective995'):
        from .selective_calibration import RoleDataset
        validation=RoleDataset(frozen,'selection')
        from .selective import validation_role
        roles={q:validation_role(q,'validation') for (q,) in validation.db.execute("SELECT DISTINCT question_key FROM samples WHERE split='validation'")}
        atomic_json(output/'validation_roles.json', {'roles':roles,'binding':binding})
    else: validation=Dataset(frozen,'validation')
    if not len(train_ds) or not len(validation):raise ValueError('empty train or validation')
    latest=json.loads((output/'latest.json').read_text()) if (output/'latest.json').exists() else None
    resume=load_resume(Path(latest['checkpoint']),binding) if latest else None
    model=build_model(cfg,checkpoint=latest['checkpoint'] if latest else None).to('cuda')
    model.progress_path=output/'heartbeat.json'
    if cfg.get('gradient_checkpointing'):
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if hasattr(model.backbone,'enable_input_require_grads'):model.backbone.enable_input_require_grads()
    atomic_json(output/'parameters.json',parameter_report(model))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(frozen/'tokenizers'/cfg['branch'],local_files_only=True)
    tokenizer.save_pretrained(output/'tokenizer')
    backbone_params=[p for p in model.backbone.parameters() if p.requires_grad]
    groups=[{'params':backbone_params,'lr':cfg['lr_backbone']},
            {'params':[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith(('backbone.','dynamics.'))],'lr':cfg['lr_heads']}]
    if hasattr(model,'dynamics'):groups.append({'params':model.dynamics.parameters(),'lr':cfg['dynamics_prior']['lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    per_epoch=math.ceil(len(train_ds)/cfg['effective_batch']);total_steps=per_epoch*cfg['epochs']
    warmup=max(1,math.ceil(.05*total_steps))
    def multiplier(step):
        if step<warmup:return step/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    state={'step':0,'epoch':0,'cursor':0,'best_score':None,'best_checkpoint':None,'validation_checks':[],
           'active_training_seconds':0.,'seed':seed}
    if resume:
        optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
        state=resume['state'];restore_rng(resume['rng'])
    from .history_rollout import HistoryState,chronological_indices,denominators as history_denominators,rollout,sequence_loss,version_records
    from .history_evaluation import write_training_versions
    history=HistoryState.restore(resume['history'],'cuda') if resume and resume.get('history') else HistoryState(cfg['own_history']['capacity'])
    model.training_history=history
    training_attempt=uuid.uuid4().hex
    parent_checkpoint=latest['checkpoint'] if latest else None
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    micro=cfg['microbatch']
    if cfg['effective_batch']%micro:raise ValueError('microbatch must divide effective batch')
    log=(output/'training.jsonl').open('a',buffering=1)
    start=time.monotonic();base_time=state['active_training_seconds']
    best_record=json.loads((output/'best.json').read_text()) if (output/'best.json').exists() else None
    best_path=Path(state['best_checkpoint']) if state.get('best_checkpoint') else None
    if best_record and (state['best_score'] is None or best_record['validation_score']>state['best_score']):
        best_path=Path(best_record['checkpoint'])
        state['best_score']=best_record['validation_score']
        state['best_checkpoint']=str(best_path)
    for epoch in range(state['epoch'],cfg['epochs']):
        order=chronological_indices(train_ds,seed+epoch)
        cursor=state['cursor'] if epoch==state['epoch'] else 0
        if cursor==0:
            history=HistoryState(cfg['own_history']['capacity']);model.training_history=history
        checkpoints=validation_steps(cfg,per_epoch)
        while cursor<len(order):
            if STOP or time.time()>=runtime['runtime_deadline_epoch']:
                state['active_training_seconds']=base_time+time.monotonic()-start
                save_checkpoint(model,optimizer,scheduler,output,state,binding)
                atomic_json(output/'INCOMPLETE.json',{'reason':'runtime_cutoff_or_signal','state':state,'binding':binding})
                return 75
            batch=[train_ds[int(i)] for i in order[cursor:cursor+cfg['effective_batch']]]
            den=history_denominators(batch,history);optimizer.zero_grad(set_to_none=True);model.train();batch_loss=0.;versions=[]
            for begin in range(0,len(batch),micro):
                rows=batch[begin:begin+micro]
                tensors=collate(rows,cfg['branch'],tokenizer.pad_token_id,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pooled=model.encode(**tensors);frames=rollout(model,rows,pooled,history)
                    loss=sequence_loss(frames,den,cfg['own_history']['revision_loss_weight'])
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite masked loss')
                loss.backward();batch_loss+=float(loss.detach())
                versions.extend(version_records(frames))
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradients')
            optimizer.step();scheduler.step();cursor+=len(batch)
            state.update(step=state['step']+1,epoch=epoch,cursor=cursor,
                         active_training_seconds=base_time+time.monotonic()-start)
            state['history_journal_tip']=write_training_versions(output/'training_versions'/training_attempt/f"step_{state['step']:09d}.jsonl.gz",versions,epoch,state['step'],parent_checkpoint)
            log.write(dumps({'step':state['step'],'epoch':epoch,'cursor':cursor,'loss':batch_loss,
                            'gradient_norm':float(norm),'elapsed_seconds':state['active_training_seconds']})+'\n')
            atomic_json(output/'heartbeat.json',{'at':time.time(),'stage':'training',**state})
            step_in_epoch=math.ceil(cursor/cfg['effective_batch'])
            if step_in_epoch in checkpoints:
                records=predict_dataset(model,validation,tokenizer.pad_token_id,cfg['eval_batch'])
                if cfg.get('selective995'):
                    from .selective_calibration import tune
                    metrics=tune(records,cfg['selective_policy']);ap=metrics['score']
                    metrics['ap_classes']=[]
                else:
                    metrics=evaluate(*arrays(records),[.5]*6);ap=metrics['macro_ap']
                if ap is None:raise ValueError('No eligible validation AP classes')
                improved=state['best_score'] is None or ap>state['best_score']
                state['validation_checks'].append({'step':state['step'],'selection_score':ap,'classes':metrics['ap_classes']})
                if improved:
                    state['best_score']=ap
                    state['_saving_best']=True
                saved=save_checkpoint(model,optimizer,scheduler,output,state,binding)
                if improved:
                    best_path=saved;state['best_checkpoint']=str(saved)
                    atomic_json(output/'best.json',{'checkpoint':str(best_path),'validation_score':ap})
                atomic_json(output/f"validation_step_{state['step']:09d}.json",metrics)
                del records
            elif state['step']%cfg['checkpoint_steps']==0:
                save_checkpoint(model,optimizer,scheduler,output,state,binding)
            if state['step']%cfg['checkpoint_steps']==0 or step_in_epoch in checkpoints:
                saved_dirs=sorted((output/'checkpoints').glob('step_*'),key=lambda p:p.stat().st_mtime,reverse=True)
                keep={p.resolve() for p in saved_dirs[:2]}
                if best_path:keep.add(best_path.resolve())
                for old in saved_dirs:
                    if old.resolve() not in keep and (old/'COMPLETE.json').exists():shutil.rmtree(old)
        state.update(epoch=epoch+1,cursor=0)
    state['active_training_seconds']=base_time+time.monotonic()-start
    if fixed_final(cfg):
        best_path=select_final_checkpoint(model,optimizer,scheduler,output,state,binding,cfg['selection'])
    else:
        save_checkpoint(model,optimizer,scheduler,output,state,binding)
    del optimizer,scheduler,model;torch.cuda.empty_cache()
    if best_path is None:raise ValueError('No selected checkpoint')
    model=build_model(cfg,checkpoint=best_path).to('cuda')
    model.progress_path=output/'heartbeat.json'
    model.prediction_binding={'checkpoint':str(best_path),'binding':binding}
    if cfg.get('selective995'):
        from .selective_finish import finish
        return finish(model,cfg,frozen,output,state,binding,best_path,tokenizer.pad_token_id,len(train_ds))
    if cfg.get('own_history'):
        from .history_finish import finish
        return finish(model,cfg,frozen,output,state,binding,best_path,tokenizer.pad_token_id,len(train_ds))
    records=predict_dataset(model,validation,tokenizer.pad_token_id,cfg['eval_batch'],output/'validation_predictions.jsonl.gz')
    a=arrays(records);calibration=thresholds(a[0],a[1],a[2])
    calibration.update(selection_split='validation',checkpoint=str(best_path),binding=binding)
    atomic_json(output/'thresholds.json',calibration)
    atomic_json(output/'validation_metrics.json',metric_report(records,calibration['f1']))
    del records
    # Test first read for model evaluation occurs after checkpoint and thresholds are frozen.
    test=Dataset(frozen,'test')
    records=predict_dataset(model,test,tokenizer.pad_token_id,cfg['eval_batch'],output/'test_predictions.jsonl.gz')
    test_report=metric_report(records,calibration['f1'])
    atomic_json(output/'test_metrics.json',test_report)
    from .plots import calibration_plot
    calibration_plot(test_report,output/'test_calibration.png')
    operational=[1.000001 if t is None else t for t in calibration['recall70']]
    atomic_json(output/'test_recall70_metrics.json',metric_report(records,operational))
    write_spans(records,output/'test_mode_spans.jsonl.gz')
    training_peak=torch.cuda.max_memory_allocated()
    del records,model;torch.cuda.empty_cache()
    from .inference import OnlineDetector,benchmark
    detector=OnlineDetector(best_path,frozen,output/'thresholds.json')
    selected=[]
    for (tid,) in validation.db.execute("SELECT DISTINCT trace_id FROM samples WHERE split='validation' ORDER BY trace_id"):
        trace=json.loads(zlib.decompress(validation.db.execute('SELECT data FROM traces WHERE trace_id=?',(tid,)).fetchone()[0]))
        if 20<=len(trace['segments'])<=256:selected.append(trace)
        if len(selected)==3:break
    if not selected:raise ValueError('No bounded whole validation traces for cost measurement')
    benchmark(detector,selected,output/'cost.json')
    from .report import write_report
    write_report(output,cfg,json.loads((frozen/'coverage.json').read_text()),state,binding)
    required=['thresholds.json','validation_predictions.jsonl.gz','validation_metrics.json',
        'test_predictions.jsonl.gz','test_mode_spans.jsonl.gz','test_metrics.json',
        'test_recall70_metrics.json','test_calibration.png','cost.json','REPORT_RU.txt']
    atomic_json(output/'COMPLETE.json',{'status':'complete','binding':binding,'seed':seed,'state':state,
      'best_checkpoint':str(best_path),'test_samples':len(test),'train_samples':len(train_ds),
      'training_gpu_hours':state['active_training_seconds']/3600,
      'peak_gpu_memory_bytes':training_peak,
      'output_hashes':{name:file_sha(output/name) for name in required},
      'required_outputs':required})
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    return train(a.config,a.frozen,a.output,json.loads(a.runtime.read_text()))


if __name__=='__main__':
    from .failures import worker_entry
    raise SystemExit(worker_entry(main))
