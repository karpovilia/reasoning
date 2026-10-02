# Инференс детектора и гейта

Дата: 2026-10-02

Команды запускаются из корня репозитория после установки по [README](../../README.md). Все выходные файлы должны быть новыми. Примеры содержат вымышленные арифметические рассуждения.

## Детектор отдельно

```bash
mkdir -p outputs
reasoning-detector predict \
  --checkpoint models/detector/lexical-event-span-v1.pkl \
  --input examples/detector_segments.jsonl \
  --output outputs/segments.jsonl
```

Каждая строка входа содержит `segments`: список завершённых текстовых сегментов в исходном порядке. На выходе каждому сегменту соответствуют словарь шести вероятностей `events`, название `span` и его `span_index`. При обработке последовательными запросами сохраняйте предыдущий сегмент: модель использует его текст как контекст. Для обычного текста доступны `segment_records(text)` и `Detector.predict_segments(...)`.

## Гейт по готовой структуре

```bash
reasoning-gate --config configs/gate/counts_spans.json \
  --input examples/gate_observation.jsonl \
  --output outputs/gate.jsonl
```

Вход одной строки:

- `now`: уже сгенерированные токены рассуждения, без начального промпта;
- `prompt_tokens`: длина начального промпта в токенах решателя;
- `segments`: только завершённые и уже размеченные сегменты;
- у сегмента `token_start`, `token_end`: полуинтервал в нативных токенах генератора; `available_at`: позиция, к которой метка стала доступна;
- `events`: шесть вероятностей в порядке `backtrack`, `verify`, `branch`, `subgoal_done`, `failed_attempt`, `commit`;
- `span`: индекс в порядке `SETUP`, `DERIVING`, `EXPLORING`, `VERIFYING`, `REVISING`, `CONCLUDING`, `GROUNDING`; `-1` означает неизвестный тип;
- `confidence`: текущая самооценка вероятности правильного ответа от решателя, от 0 до 1; `confidence_available_at`: позиция её получения, по умолчанию `now`;
- необязательный `key`: ключ для воспроизводимого перемешивания в соответствующей абляции. Он не входит в признаки.

Все координаты токенов должны происходить из исходной генерации. Сумма длин отдельно токенизированных предложений может отличаться от длины исходной последовательности. Её нельзя использовать как замену нативным координатам. Для сегмента требуется `token_end <= available_at <= now`.

`confidence` должна быть получена отдельным запросом к решателю о текущем ответе. В эксперименте использовался промпт: `Estimate the probability that the answer you would give NOW is correct. Use the supplied question and your partial reasoning, without continuing the solution. Return only a JSON object {"confidence": <number between 0 and 1>}.` Расход этого запроса входит в стоимость наблюдения. Уверенность разметчика и правильность из эталона не заменяют самооценку решателя.

## Детектор и гейт вместе

```bash
reasoning-gate --config configs/gate/counts_spans.json \
  --detector models/detector/lexical-event-span-v1.pkl \
  --input examples/pipeline.jsonl --output outputs/pipeline.jsonl
```

Здесь у сегментов передаются `text` и токенные координаты. Обёртка получает события и спаны от детектора и подаёт их в гейт. Координаты в примере иллюстративные; они не являются результатом токенизации какой-либо модели.

Для вызова из Python:

```python
from reasoning_control.detector import Detector
from reasoning_control.gate import detect_and_decide, load

detector = Detector("models/detector/lexical-event-span-v1.pkl")
gate = load("configs/gate/counts_spans.json")
result = detect_and_decide(detector, gate, observation)
action = result["decision"]["action"]
```

`observation` имеет формат строки из `examples/pipeline.jsonl`. Для конфигурации вне репозитория передайте `root` в `load` или `--root` в CLI — корень каталога с `models/gate`.

## Выполнение выбранного действия

Обёртка вычисляет решение гейта. Вызывающий генератор выполняет выбранное действие из того же префикса:

- `stop`: завершить рассуждение и получить итоговый ответ; это не возврат пустого ответа;
- `continue_512` и `continue_2048`: продолжить рассуждение в пределах указанного дополнительного бюджета, затем получить ответ;
- `verify_512`: вставить подсказку `I should check the last conclusion for an error before continuing.` и продолжить;
- `neutral_512`: вставить `I will continue the reasoning from the current point.` и продолжить.

Обе подсказки окружались двумя переводами строки и дополнялись переводами строки до одинаковой токенной длины. Их токены вычитались из бюджета 512. На финальный ответ отдельно отводилось до 512 токенов. Это часть протокола действий, на котором обучались головы гейта; изменение подсказки или лимитов меняет смысл прогноза. В опубликованном эксперименте гейт выбирал действие в одной контрольной точке, после 1 024 токенов. Многократное применение требует отдельной проверки.

Выход содержит `actions`, `p_success`, `predicted_additional_tokens`, `cost`, `cost_unit`, `action`, `utility`, `fallback` и `reason`. У штатного решения `cost_unit="seconds"`; при возврате резервного действия числовые прогнозы равны `null` и единицу стоимости следует читать из записи. `fallback=true` нужно сохранять при подсчёте качества и расхода.

## Большой разметчик

[Адаптер](../../models/annotator/qwen3.6-35b-a3b-ck22000) совместим с базой из [рецепта](../../configs/annotation/training.json). Для исторического способа инференса сначала слейте адаптер с базой:

```bash
python tools/release/merge_annotator.py \
  --base BASE_SNAPSHOT_DIRECTORY \
  --adapter models/annotator/qwen3.6-35b-a3b-ck22000 \
  --out MERGED_MODEL_DIRECTORY
```

`BASE_SNAPSHOT_DIRECTORY` и `MERGED_MODEL_DIRECTORY` заменяются локальными каталогами. Затем загрузите объединённую модель в совместимый сервер генерации, передайте системный промпт [event_span_math.txt](../../configs/annotation/event_span_math.txt) и нумерованные сегменты. Параметры разметки: температура 0, `top_p=1`, `enable_thinking=false`, до 4 096 выходных токенов, контекст до 24 576 токенов. Нужный правый контекст и границы целевого окна записываются в конфигурации разметки. Большой разметчик не запускался заново при подготовке этого выпуска.
