"""Full-quality LoRA history evaluation; independent validation thresholds per lag."""
import json,zlib
import torch
from .core import atomic_json,file_sha,dumps
from .dataset import Dataset
from .history_evaluation import journal_path,records_at_lag,latest_records
from .train import arrays,predict_dataset,metric_report,write_spans
from .metrics import thresholds
from .selective_finish import freeze_json


def calibration(records):
    if not records:return {'f1':[None]*6,'recall70':[None]*6,'status':'no_available_foci'}
    a=arrays(records);return thresholds(a[0],a[1],a[2])


def safe_thresholds(values):
    return [1.000001 if value is None else value for value in values]


def finish(model,cfg,frozen,output,state,binding,best_path,pad_id,train_count):
    peak=torch.cuda.max_memory_allocated();lags=cfg['own_history']['revision_lags_reported']
    selection=cfg.get('selection','primary validation event macro AP')
    validation=Dataset(frozen,'validation');vp=output/'validation_predictions.jsonl.gz'
    val=predict_dataset(model,validation,pad_id,cfg['eval_batch'],vp)
    initial=calibration(val)
    freeze_json(output/'thresholds.json',{**initial,'binding':binding,'checkpoint':str(best_path),
        'selection_split':'validation','answer_version':'immutable primary +0'})
    revisions={str(lag):calibration(records_at_lag(val,vp,lag)) for lag in lags}
    revisions['latest']=calibration(latest_records(val,vp))
    freeze_json(output/'revision_thresholds.json',{'by_exact_lag':revisions,'binding':binding,
        'checkpoint':str(best_path),'selection_split':'validation','test_used_for_selection':False,
        'latest_scope':'mixed available ages0..16; separate from the original +0 answer'})
    atomic_json(output/'validation_metrics.json',metric_report(val,safe_thresholds(initial['f1'])))
    dynamics_outputs=[]
    if cfg.get('dynamics_prior'):
        from .dynamics_report import forecast_records,binary_report
        prior_calibration=calibration(forecast_records(val))
        freeze_json(output/'dynamics_thresholds.json',{**prior_calibration,'binding':binding,
            'selection_split':'validation','scope':'next-focus forecast before reading current text'})
        atomic_json(output/'validation_dynamics_metrics.json',{'events_modes_spans':metric_report(forecast_records(val),safe_thresholds(prior_calibration['f1'])),
            'any_event':binary_report(val)})
        dynamics_outputs=['dynamics_thresholds.json','validation_dynamics_metrics.json','test_dynamics_metrics.json']
    del val
    test=Dataset(frozen,'test');tp=output/'test_predictions.jsonl.gz'
    records=predict_dataset(model,test,pad_id,cfg['eval_batch'],tp)
    if cfg.get('dynamics_prior'):
        atomic_json(output/'test_dynamics_metrics.json',{'events_modes_spans':metric_report(forecast_records(records),safe_thresholds(prior_calibration['f1'])),
            'any_event':binary_report(records)})
    primary=metric_report(records,safe_thresholds(initial['f1']))
    atomic_json(output/'test_metrics.json',primary)
    atomic_json(output/'test_recall70_metrics.json',metric_report(records,safe_thresholds(initial['recall70'])))
    write_spans(records,output/'test_mode_spans.jsonl.gz')
    from .plots import calibration_plot
    calibration_plot(primary,output/'test_calibration.png')
    availability={};extra=[]
    for lag in lags:
        rows=records_at_lag(records,tp,lag);prefix=f'test_lag{lag}'
        metrics=metric_report(rows,safe_thresholds(revisions[str(lag)]['f1'])) if rows else {'status':'no_available_foci'}
        support={'lag':lag,'available':len(rows),'original_stream':len(records),
            'availability_fraction':len(rows)/len(records),'unavailable_tail_or_gap':len(records)-len(rows)}
        atomic_json(output/(prefix+'_metrics.json'),{'events_modes_spans':metrics,'availability':support})
        write_spans(rows,output/(prefix+'_mode_spans.jsonl.gz'))
        availability[str(lag)]=support;extra.extend([prefix+'_metrics.json',prefix+'_mode_spans.jsonl.gz'])
    latest=latest_records(records,tp)
    atomic_json(output/'test_latest_metrics.json',metric_report(latest,safe_thresholds(revisions['latest']['f1'])))
    write_spans(latest,output/'test_latest_mode_spans.jsonl.gz')
    atomic_json(output/'test_revision_summary.json',{'exact_lags':availability,'primary_and_revised_outputs_preserved':True,
        'latest_scope':'mixed available lag0..16; diagnostic delayed result, no primary quality substitution',
        'history':'own detached probabilities; no teacher forcing','model_selection':selection})
    del latest,records
    selected=[]
    for (tid,) in validation.db.execute("SELECT DISTINCT trace_id FROM samples WHERE split='validation' ORDER BY trace_id"):
        trace=json.loads(zlib.decompress(validation.db.execute('SELECT data FROM traces WHERE trace_id=?',(tid,)).fetchone()[0]))
        if 20<=len(trace['segments'])<=128:selected.append(trace)
        if len(selected)==3:break
    if not selected:raise ValueError('No bounded validation traces for online cost')
    from .history_inference import HistoryOnlineDetector,benchmark
    from transformers import AutoTokenizer
    detector=HistoryOnlineDetector.__new__(HistoryOnlineDetector)
    detector.model=model;detector.config=cfg;detector.device='cuda'
    detector.tokenizers={name:AutoTokenizer.from_pretrained(frozen/'tokenizers'/name,local_files_only=True) for name in ('encoder','lora4b')}
    peak=max(peak,torch.cuda.max_memory_allocated());benchmark(detector,selected,output/'cost.json')
    atomic_json(output/'cascade_scope.json',{'encoder_filter_plus_lora_measured':False,
        'scope':'this is the complete standalone LoRA stream; cascade replay requires an independently accepted encoder policy',
        'input':'original observed prefix including all prior text; no filtered-history or gold-history substitution'})
    (output/'REPORT_RU.txt').write_text('LoRA с собственной историей и Transformer-прогнозом: события И режимы/спаны\n'
        'Transformer обучен с нуля совместно на train; его прогноз видит только прошлые 16 сегментов.\n'
        'Первичные +0 ответы неизменяемы; отдельные пересмотры age1..16 используют наблюдаемый префикс.\n'
        'Прямые Flash/Pro targets, исходные masks/split; один seed, human gold отсутствует.\n' +
        ('Выбор checkpoint: '+selection+'. Пороги первичных, exact-lag и latest ответов раздельны.\n') +
        'Тест не используется для выбора. Latest — смешанная задержка, не замена первичной оценки.\n'
        'Совместная цепочка с энкодером ещё не измерена. Сравнение с прежней LoRA требует paired question bootstrap.\n'
        +dumps({'primary_test':{'event_macro_f1':primary['macro']['f1'],'event_macro_ap':primary['macro_ap'],
        'mode_macro_f1':primary['modes']['macro_f1'],'spans':primary['spans']},'availability':availability})+'\n')
    required=['thresholds.json','revision_thresholds.json','validation_metrics.json','test_metrics.json',
        'test_recall70_metrics.json','test_mode_spans.jsonl.gz','test_calibration.png','test_latest_metrics.json',
        'test_latest_mode_spans.jsonl.gz','test_revision_summary.json','cost.json','cascade_scope.json','REPORT_RU.txt']+extra+dynamics_outputs
    for name in ('validation_predictions.jsonl.gz','test_predictions.jsonl.gz'):
        required.extend([name,journal_path(output/name).name,name+'.COMPLETE.json'])
    atomic_json(output/'COMPLETE.json',{'status':'complete','scientific_acceptance':'pending_review','binding':binding,
        'seed':cfg['seed'],'state':state,'best_checkpoint':str(best_path),'test_samples':len(test),'train_samples':train_count,
        'training_gpu_hours':state['active_training_seconds']/3600,
        'peak_gpu_memory_bytes':max(peak,torch.cuda.max_memory_allocated()),
        'required_outputs':required,'output_hashes':{name:file_sha(output/name) for name in required}})
    return 0
