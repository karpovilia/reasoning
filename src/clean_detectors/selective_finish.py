"""Independent primary/lag policies, audit, and immutable final history outputs."""
import gzip
import json
import time
import zlib
import numpy as np
import torch
from .core import atomic_json,dumps,file_sha
from .dataset import Dataset
from .selective_calibration import RoleDataset,tune,report,audit_policy,tune_full,full_report,full_candidates
from .history_evaluation import journal_path,records_at_lag,latest_records
from .train import predict_dataset,metric_report,write_spans


def freeze_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen policy changed on recovery')
    else:atomic_json(path,value)


def views(primary,path,lags):
    yield 0,primary
    for lag in lags:yield lag,records_at_lag(primary,path,lag)


def candidate(records,cfg):
    binary=tune(records,cfg)['policy'] if records else {'absence_max':None,'presence_min':None}
    return {'binary':binary,'full_thresholds':tune_full(records,binary,cfg)}


def certify(records,selected,cfg):
    audited=audit_policy(records,selected['binary'],cfg)
    full=full_report(records,selected['binary'],selected['full_thresholds'],cfg)
    required=[k for k,v in selected['full_thresholds'].items() if v is not None]
    passed=audited['passed'] and bool(required) and all(full['branches'][k]['pass'] for k in required)
    return {'binary':audited['enabled'], 'full_thresholds':selected['full_thresholds'] if passed else {'absent':None,'present':None},
            'binary_audit':audited,'full_audit':full,'full_pass':passed}


def finish(model,cfg,frozen,output,state,binding,best_path,pad_id,train_count):
    training_peak=torch.cuda.max_memory_allocated()
    policy_cfg=cfg['selective_policy'];lags=cfg['own_history']['revision_lags_reported']
    tuning=RoleDataset(frozen,'tuning');tuning_path=output/'tuning_predictions.jsonl.gz'
    records=predict_dataset(model,tuning,pad_id,cfg['eval_batch'],tuning_path)
    candidates={str(lag):candidate(rows,policy_cfg) for lag,rows in views(records,tuning_path,lags)}
    freeze_json(output/'candidate_policy.json',{'by_exact_lag':candidates,'binding':binding,'checkpoint':str(best_path),
        'selection':'tuning role only; lag0 primary independent from revised policies'})
    del records
    audit=RoleDataset(frozen,'audit');audit_path=output/'audit_predictions.jsonl.gz'
    records=predict_dataset(model,audit,pad_id,cfg['eval_batch'],audit_path)
    policies={str(lag):certify(rows,candidates[str(lag)],policy_cfg) for lag,rows in views(records,audit_path,lags)}
    freeze_json(output/'policy.json',{'by_exact_lag':policies,'checkpoint':str(best_path),'binding':binding,
        'test_used_for_selection':False,'other_lags':'uncertified; abstain for selective routing',
        'mode':'initial policy at lag0; delayed policies do not change earlier acceptance claims'})
    del records
    test=Dataset(frozen,'test');test_path=output/'test_predictions.jsonl.gz'
    records=predict_dataset(model,test,pad_id,cfg['eval_batch'],test_path)
    summaries={};extra=[]
    for lag,rows in views(records,test_path,lags):
        prefix=f'test_lag{lag}';policy=policies[str(lag)]
        event_report=metric_report(rows,[.5]*6) if rows else {'status':'no_eligible_foci'}
        binary=report(rows,policy['binary'],policy_cfg)
        routed=full_report(rows,policy['binary'],policy['full_thresholds'],policy_cfg)
        routed.update(eligible_at_exact_lag=len(rows),primary_focus_count=len(records),
                      availability_fraction=len(rows)/len(records),
                      full_bypass_fraction_of_original_stream=routed['full_bypass_count']/len(records),
                      unavailable_tail_or_gap_count=len(records)-len(rows),lag=lag)
        atomic_json(output/(prefix+'_metrics.json'),{'events_modes_spans':event_report,'binary':binary,'routing':routed})
        write_spans(rows,output/(prefix+'_mode_spans.jsonl.gz'))
        with gzip.open(output/(prefix+'_routes.jsonl.gz'),'wt',compresslevel=1) as stream:
            for row,decision in zip(rows,full_candidates(rows,policy['binary']),strict=True):
                threshold=policy['full_thresholds'].get(decision['route'])
                bypass=bool(decision['valid'] and threshold is not None and decision['score']>=threshold)
                stream.write(dumps({'sample_id':row['sample_id'],'sid':row['sid'],'as_of_sid':row['as_of_sid'],
                    'lag':lag,'binary_route':decision['route'],'full_bypass':bypass,'needs_lora':not bypass,
                    'events':decision['events'] if bypass else None,'mode':decision['mode'] if bypass else None,
                    'fallback_context':'original available prefix, never filtered history'})+'\n')
        summaries[str(lag)]=routed
        extra.extend(prefix+suffix for suffix in ('_metrics.json','_mode_spans.jsonl.gz','_routes.jsonl.gz'))
    latest=latest_records(records,test_path)
    atomic_json(output/'test_latest_metrics.json',metric_report(latest,[.5]*6))
    write_spans(latest,output/'test_latest_mode_spans.jsonl.gz')
    atomic_json(output/'test_revision_summary.json',{'exact_lags':summaries,
        'latest_scope':'latest bounded revision, age0..16; no primary precision claim for this mixed-lag output',
        'training_order':'trace shuffle; chronological sid; no teacher-forced labels',
        'primary_and_revised_outputs_preserved':True})
    del records,latest
    # Measure actual batch1 tokenization + own history + all revisions on bounded
    # whole tuning traces. The accepted LoRA remains a separate evaluation task.
    selected=[]
    from .selective import validation_role
    for tid,q in tuning.db.execute("SELECT DISTINCT trace_id,question_key FROM samples WHERE split='validation' ORDER BY trace_id"):
        if validation_role(q,'validation')!='tuning':continue
        trace=json.loads(zlib.decompress(tuning.db.execute('SELECT data FROM traces WHERE trace_id=?',(tid,)).fetchone()[0]))
        if 20<=len(trace['segments'])<=128:selected.append(trace)
        if len(selected)==3:break
    if not selected:raise ValueError('No bounded tuning traces for cost')
    from .history_inference import HistoryOnlineDetector,benchmark
    # Share the already loaded encoder; avoid two production backbones in memory.
    detector=HistoryOnlineDetector.__new__(HistoryOnlineDetector)
    detector.config=cfg;detector.model=model;detector.device='cuda'
    from transformers import AutoTokenizer
    detector.tokenizers={name:AutoTokenizer.from_pretrained(frozen/'tokenizers'/name,local_files_only=True) for name in ('encoder','lora4b')}
    peak_before_benchmark=torch.cuda.max_memory_allocated()
    benchmark(detector,selected,output/'cost.json')
    atomic_json(output/'cost_limits.json',{'actual_lora_chain_cost':None,'actual_lora_chain_quality':None,
        'note':'Actual LoRA chain requires accepted LoRA checkpoint and separate evaluation; routing replay is not measured total savings',
        'max_revisions_per_focus':cfg['own_history']['revision_horizon'],'backbone_passes_per_focus':1,'extra_revision_loss_weight':.5,
        'training_gpu_hours':state['active_training_seconds']/3600})
    (output/'REPORT_RU.txt').write_text('SELECTIVE995 С СОБСТВЕННОЙ ИСТОРИЕЙ И ПРАВКАМИ\n'
        'Первичный ответ +0 сохранён; правки age1..16 используют только доступный префикс.\n'
        'История из своих вероятностей, targets только Flash/Pro; masks/split неизменны.\n'
        'Порог99,5% оценён отдельно для lag0/1/2/4/8/16 на известных метках.\n'
        'Unknown outcomes и availability знаменатели сохранены; универсальная гарантия не заявляется.\n'
        'Реальная цепочка с LoRA ещё требует отдельной оценки.\n'+dumps(summaries)+'\n')
    required=['validation_roles.json','candidate_policy.json','policy.json','test_latest_metrics.json',
              'test_latest_mode_spans.jsonl.gz','test_revision_summary.json','cost.json','cost_limits.json','REPORT_RU.txt']+extra
    for name in ('tuning_predictions.jsonl.gz','audit_predictions.jsonl.gz','test_predictions.jsonl.gz'):
        required += [name,journal_path(output/name).name,name+'.COMPLETE.json']
    atomic_json(output/'COMPLETE.json',{'status':'complete','scientific_acceptance':'pending_review','binding':binding,
        'seed':cfg['seed'],'state':state,'best_checkpoint':str(best_path),'test_samples':len(test),'train_samples':train_count,
        'training_gpu_hours':state['active_training_seconds']/3600,'peak_gpu_memory_bytes':max(training_peak,peak_before_benchmark,torch.cuda.max_memory_allocated()),
        'required_outputs':required,'output_hashes':{name:file_sha(output/name) for name in required}})
    return 0
