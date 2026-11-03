"""Write the branch report from completed measurements, never from expectations."""
import json
from pathlib import Path
from .core import EVENTS,MODES


def number(value):return 'нет оценки' if value is None else f'{value:.4f}'


def write_report(output,config,coverage,state,binding):
    output=Path(output)
    test=json.loads((output/'test_metrics.json').read_text())
    cost=json.loads((output/'cost.json').read_text())
    calibration=json.loads((output/'thresholds.json').read_text())
    lines=[f"Детектор {config['branch']}: события и спаны режимов",
        f"База: {config['model']['model_id']} @ {config['model']['revision']}",
        f"Seed: {config['seed']}; завершено эпох: {state['epoch']}; optimizer steps: {state['step']}.",
        'Это один seed: разброс обучения между seeds не оценён.',
        'Только прямые подтверждённые Flash/Pro; unknown исключены из loss и оценки.',
        'Reference — автоматическая разметка Flash/Pro; human gold для этого опыта отсутствует.',
        'Вход +0: вопрос, до 16 прошлых целых сегментов, весь текущий завершённый сегмент.',
        'Общий текст помещается в 2048 токенов обоих tokenizer; будущего текста во входе нет.',
        'Teacher мог видеть будущий текст: reference ретроспективный, а детектор причинный.',
        'Checkpoint выбран на validation по macro-AP событий, пороги — по validation F1.',
        'Выбор checkpoint по событиям не гарантирует максимум качества режимов.',
        '',f"Тест, события: micro-F1={number(test['micro']['f1'])}, macro-F1={number(test['macro']['f1'])}, macro-AP={number(test['macro_ap'])}.",
        'Тип | известные | положительные | precision | recall | F1 | AP | Brier | порог']
    for index,event in enumerate(EVENTS):
        m=test['events'][event]
        lines.append(f"{event} | {m['known']} | {m['positive']} | "+' | '.join(number(m[k]) for k in ['precision','recall','f1','ap','brier'])+f" | {number(calibration['f1'][index])}")
    mode=test['modes'];span=test['spans']
    lines += ['',f"Режимы: {', '.join(MODES)}.",
        'GROUNDING не обучался; интерфейс отличается от production с семью режимами.',
        f"Известно режимов: {mode['known']}; покрытие={number(mode['coverage'])}; accuracy={number(mode['accuracy'])}; macro-F1={number(mode['macro_f1'])}.",
        f"Спаны: exact internal F1={number(span['exact_internal_spans']['f1'])}; typed boundary F1={number(span['typed_boundaries']['f1'])}.",
        'Спаны оцениваются внутри непрерывных участков с известной разметкой; спорные и непокрытые края цензурированы.',
        'Онлайн последний спан остаётся открытым до наблюдаемого перехода/EOS; границы выдаются по индексам сегментов.',
        '',f"Накопленное время сохранённого training state: {state['active_training_seconds']/3600:.3f} часов.",
        'Полная стоимость, включая рестарты и интервалы сбоев, дополнительно находится в supervisor COMPLETE.json.',
        'Полный inference pipeline: сериализация, общий подбор контекста, оба tokenizer, перенос, backbone и головы.']
    for batch,m in cost['results'].items():
        lines.append(f"batch={batch}: p50/p95 фокуса {number(m['focus_p50_seconds'])}/{number(m['focus_p95_seconds'])} с; "
            f"p50/p95 трассы {number(m['trace_p50_seconds'])}/{number(m['trace_p95_seconds'])} с; throughput={number(m['throughput_focus_per_second'])} фокусов/с; peak={m['peak_gpu_bytes']} bytes.")
    lines += ['', 'Старые Hashing/SGD, MiniLM+GRU и LoRA ck22000: не переоценены; прежние числа не перенесены.',
        'Преимущество над второй новой моделью пока не заявляется. После обеих веток нужен paired bootstrap по вопросам.',
        'Исторические роли вопросов и baseline-overlap сохранены в общем корпусе; test не объявлен новым независимым holdout целиком.',
        'Срезы по учителю/протоколу/решателю/длине/концу трассы/маркерам: test_metrics.json.',
        'Рабочая точка validation recall>=70%: test_recall70_metrics.json; фактический test recall может быть ниже.',
        'Покрытие корпуса: '+json.dumps({key:coverage.get(key) for key in ['train','validation','test']},ensure_ascii=False),
        'Binding: '+json.dumps(binding,ensure_ascii=False),
        'Этот отчёт не меняет production-контроллер или тексты статьи.']
    (output/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n')
