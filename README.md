# Covenant agent

Агент проверки финансовых ковенантов с изолированным ядром `model/` и CLI-адаптером `cli/`. Ядро возвращает типизированные объекты и не читает переменные окружения, не печатает в терминал и не пишет итоговый файл.

## Быстрый запуск

Требования: Python 3.9+ и `uv`; для редких PDF-страниц без текстового слоя нужен Tesseract с языками `rus` и `eng`.

```bash
uv run --extra cli python -m cli run
```

Если виртуальное окружение уже активировано, достаточно `python -m cli run`.

При первом интерактивном запуске CLI:

- предложит установить Ollama, если она отсутствует (macOS, Linux или Windows);
- предложит скачать `gemma4:e2b`, выбрать другую модель Ollama или продолжить offline;
- запустит локальный сервер и сохранит модель в системном кэше, а не в Git-репозитории.

Для своей Ollama-совместимой модели:

```bash
python -m cli run --gemma-model gemma3:4b
python -m cli run --gemma-model my-model --gemma-endpoint http://server:11434
```

В CI и других неинтерактивных окружениях установка не запускается автоматически: заранее
подготовьте endpoint либо используйте `--offline`.

Название команды и контактная почта заданы константами `TEAM_NAME` и `CONTACT_EMAIL` в `cli/settings.py`; при каждом запуске они автоматически записываются в `submission.json`. По умолчанию Gemma вызывается через Ollama-совместимый endpoint. Модель и URL меняются через `GEMMA_MODEL_ID` и `GEMMA_ENDPOINT` либо аргументами CLI.

## Команды

```bash
# Основной запуск: сам проверит Ollama, возьмёт данные из текущей папки
# и запишет результат в submission.json
python -m cli run

# Запуск и вывод процента сходства результата с ground_truth.json
python -m cli run --score

# Детерминированный прогон без нейросети и без Ollama
python -m cli run --offline --no-cache

python -m cli validate
python -m cli score
python -m cli inspect --scenario P1 --covenant 6.3 --offline
```

Offline-режим работает без Ollama и без скачивания модели: числовой участник ансамбля остаётся,
а Gemma заменяется детерминированной заглушкой. Текущий регрессионный результат:
`35.00 / 36 = 97.2%`. `ground_truth.json` используется только скорером и никогда не читается генератором обучающих данных.

## LoRA-обучение на Apple Silicon

```bash
UV_CACHE_DIR=/tmp/covenant-uv uv sync --extra cli --extra dev --extra train
.venv/bin/python -m cli train \
  --input . \
  --base-model mlx-community/gemma-3-1b-it-4bit \
  --data artifacts/training-data \
  --adapter artifacts/adapters/gemma-covenants \
  --iters 50 \
  --offline --workers 1 --no-cache
```

Корпус делится по заёмщикам (`P1-P8` train, `P9-P10` validation, `B1/B4` test) и содержит задачи классификации документов, извлечения KYC, определения ковенантов и категорий операций. Ответы из `ground_truth.json` в корпус не попадают.

## Публичный API

```python
from model import DatasetRef, Settings, build_pipeline

pipeline = build_pipeline(Settings())
submission = pipeline.run(DatasetRef("."))
payload = submission.to_submission_dict()
```

`pipeline.analyze_one(task, context)` принимает готовый `BorrowerContext`, поэтому будущий HTTP/backend-адаптер сможет передать контекст без чтения PDF внутри endpoint-а.

## Проверки

```bash
.venv/bin/pytest
```

Тесты контролируют архитектурную границу, самостоятельный импорт `model/`, отсутствие терминального вывода в ядре, точную сериализацию шаблона, Decimal-агрегаты, KYC matching и JSON-разбор Gemma.
