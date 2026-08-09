<div align="center">

<img src="img/banner.svg" alt="Astrea — Covenant Compliance Agent" width="100%">

<br>

[![Python](https://img.shields.io/badge/python-3.10%2B-000000?logo=python&logoColor=00C853)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-gemma-000000?logo=ollama&logoColor=00C853)](https://ollama.com)
[![Docker](https://img.shields.io/badge/Docker-compose-000000?logo=docker&logoColor=00C853)](https://docs.docker.com/compose/)
[![uv](https://img.shields.io/badge/uv-managed-000000?logo=uv&logoColor=00C853)](https://docs.astral.sh/uv/)
[![Platforms](https://img.shields.io/badge/macOS%20·%20Linux%20·%20Windows-000000)](#быстрый-старт)

**Команда Astrea** · Halyk AI Challenge

</div>

---

## О проекте

Агент читает «грязные» кредитные документы и определяет, нарушены ли финансовые ковенанты
корпоративных займов. На каждую ячейку (заёмщик × ковенант) он выдаёт вердикт
`COMPLIANT`/`BREACH`, фактическое значение показателя и транзакцию-доказательство.

Агент рассчитан на ловушки реального документооборота и уверенно их проходит: узнаёт заёмщика
по обезличенному файлу, определяет категорию операции из назначения платежа, отличает
действующий договор от прошлогодней редакции и опирается только на итоговое заключение
аудитора, игнорируя черновики.

---

## Быстрый старт

### Шаг 1. Конфигурация

Скопируйте шаблон настроек и при необходимости поправьте значения:

**macOS и Linux**

```bash
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

### Шаг 2. Зависимости

<details open>
<summary><b>Через uv</b> — рекомендуется</summary>

**macOS и Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra cli --extra dev
```

**Windows (PowerShell)**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --extra cli --extra dev
```
</details>

<details>
<summary><b>Через pip</b></summary>

**macOS и Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
</details>

### Шаг 3. Модель Gemma

<details open>
<summary><b>Вариант A — Docker Compose</b>, одинаково на любой ОС</summary>

```bash
docker compose up -d
docker compose ps
```

Compose поднимает Ollama и сам скачивает `gemma4:e2b`. Дождитесь статуса `healthy`, прогресс
первой загрузки виден в `docker compose logs -f gemma`.

Веса лежат в volume `covenant-model_gemma-models`, поэтому повторно 7,2 ГБ не скачиваются.
API доступно на `http://localhost:11434`.

Остановить: `docker compose down`. Удалить вместе с весами: `docker compose down --volumes`.
</details>

<details>
<summary><b>Вариант B — нативная Ollama</b></summary>

**macOS**

```bash
brew install ollama
ollama serve &
ollama pull gemma4:e2b
```

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma4:e2b
```

**Windows (PowerShell)**

```powershell
winget install Ollama.Ollama
ollama pull gemma4:e2b
```

Сервер поднимается автоматически вместе с приложением.
</details>

### Шаг 4. Данные

В репозитории используются два каталога данных:

```
public/                       открытый тестовый набор с ground_truth.json
private/                      основной набор без ground_truth.json
```

В каждом наборе находятся `master_ledger_2025.csv`, `documents/` и
`submission_template.json`. Для произвольного набора можно передать `--input PATH`.

Состав набора агент определяет сам: список сценариев и номера пунктов читаются из
`submission_template.json`, а соответствие сценария банковскому счёту выводится из леджера.
Поэтому наборы с любым числом заёмщиков и любой нумерацией ковенантов — включая пункты
`5.x` и англоязычные договоры — обрабатываются без правки кода.

---

## Команды

| Команда | Что делает |
|---|---|
| `run` | Полный прогон датасета → `submission.json` |
| `score` | Оценка готового сабмишна против ключа по официальной формуле |
| `validate` | Сверка структуры сабмишна с шаблоном организаторов |
| `inspect` | Разбор одной ячейки: оценки участников, цитата, документ |
| `train` | Подготовка SFT-корпуса и запуск MLX LoRA |

```bash
python -m cli run                                          # основной private-набор
python -m cli run --private                                # то же самое явно
python -m cli run --public                                 # public-набор + тестовый скоринг
python -m cli run --input data/custom --output final.json  # произвольный набор
python -m cli validate                                     # проверка по private-шаблону
python -m cli validate --public                            # проверка по public-шаблону
python -m cli score --public                               # скоринг против public-ключа
python -m cli inspect --scenario S1 --covenant 6.1         # почему такой ответ
python -m cli train --input private                        # корпус и запуск LoRA
```

`--public` и `--private` работают одинаково в `run`, `score` и `validate`, поэтому набор
выбирается одним и тем же флагом на всех шагах.

## Флаги

В столбце «Ограничения» прочерк означает, что флаг доступен во всех командах, которые его
принимают: `run`, `inspect` и `train`.

| Флаг | По умолчанию | Ограничения | Назначение |
|---|---|---|---|
| `--env-file` | `.env` | — | Откуда читать настройки |
| `--llm-mode` | `gaps-only` | — | `gaps-only` — Gemma только там, где числовая не дала значения; `always` — на каждой ячейке |
| `--offline` | выкл | — | Заменяет Gemma детерминированной заглушкой |
| `--no-cache` | выкл | — | Отключает дисковый кэш ответов модели |
| `--gemma-model` | `gemma4:e2b` | — | Имя модели в Ollama |
| `--gemma-endpoint` | `http://localhost:11434` | — | URL Ollama-совместимого сервера |
| `--workers` | `1` | — | Потоков обработки |
| `--fx-eur-usd` | `1.00` | — | Курс пересчёта EUR-строк леджера |
| `--team` | `Astrea` | — | Поле `team` в сабмишне |
| `--contact-email` | из `cli/settings.py` | — | Поле `contact_email` в сабмишне |
| `--public` | выкл | `run`, `score`, `validate` | Открытый набор `public/`; в `run` включает локальный скоринг |
| `--private` | вкл | `run`, `score`, `validate` | Основной набор `private/`, ключ не требуется |
| `--input` | `private/` | — | Каталог с `documents/`, леджером и шаблоном; в `run` несовместим с `--public`/`--private` |
| `--output` | `submission.json` | Только для `run` | Куда записать результат |
| `--key` | из выбранного набора | Только для `score` | Явный путь к ключу вместо `<набор>/ground_truth.json` |
| `--template` | из выбранного набора | Только для `validate` | Явный путь к шаблону вместо `<набор>/submission_template.json` |
| `--scenario` | обязателен | Только для `inspect` | Идентификатор сценария из выбранного шаблона |
| `--covenant` | обязателен | Только для `inspect` | Номер пункта из выбранного шаблона |
| `--data` | `artifacts/training-data` | Только для `train` | Куда сложить корпус |
| `--adapter` | `artifacts/adapters/gemma-covenants` | Только для `train` | Куда сохранить веса |
| `--base-model` | `mlx-community/gemma-3-1b-it-4bit` | Только для `train` | Базовая модель MLX |
| `--iters` | `50` | Только для `train` | Шагов обучения |

`score` и `validate` принимают путь к сабмишну первым позиционным аргументом; по умолчанию —
`submission.json`.

---

## Стек технологий

| Слой | Технология | Зачем |
|---|---|---|
| Язык | **Python 3.10+** | `Decimal` для денег, `Protocol` для портов |
| Извлечение текста | [PyMuPDF](https://pymupdf.readthedocs.io/) | Текстовый слой напрямую |
| LLM | [Ollama](https://ollama.com) + Gemma | Локальный инференс, ключей и внешних API не требует |
| Развёртывание модели | [Docker Compose](https://docs.docker.com/compose/) | Одинаковый запуск на любой машине |
| Дообучение | [MLX](https://ml-explore.github.io/mlx/) + `mlx-lm` | LoRA на Apple Silicon |
| Числовая модель | Своя, детерминированная | Арифметика должна быть точной, а не правдоподобной |
| CLI | `argparse` + [rich](https://rich.readthedocs.io/) | Прогресс и таблицы только в адаптере, ядро о них не знает |
| Тесты | `pytest` | Границы, юниты, end-to-end регрессия |
| Пакеты | [uv](https://docs.astral.sh/uv/) | Лок-файл и быстрая установка |

Внешних API, облачных ключей и векторных БД нет: пакет документов заёмщика целиком помещается
в контекст модели, поэтому retrieval не нужен.

---

## Архитектура

Две директории верхнего уровня и строго односторонняя зависимость между ними.

```
                        ┌──────────────────────────────────────┐
   cli/       ─────────►│               model/                 │
   (сегодня)            │                                      │
                        │   Ансамбль: gemma + числовая         │
   api/       ─────────►│                                      │
   (будущий сервис)     │   Не знает, кто его вызвал,          │
                        │   куда пишется результат             │
   worker/    ─────────►│   и существует ли терминал           │
   (будущая очередь)    └──────────────────────────────────────┘
```

`model/` — самодостаточный пакет: копируется в другой репозиторий и подключается к любому
сервису без единой правки.

```
model/
├── domain/           типы и правила. Нулевые зависимости
├── ports/            Protocol-интерфейсы. Ядро зависит только от них
├── adapters/         PyMuPDF, CSV, Ollama, кэш, заглушка — заменяются поштучно
├── ensemble/         участники, политика разрешения, оркестрация
├── services/         контекст, метрики, matching, пайплайн
├── config.py         Settings — frozen dataclass, без чтения env
└── composition.py    сборка графа объектов: Settings → CovenantPipeline

cli/
├── commands/         run, score, validate, inspect, train
├── presenters/       rich-прогресс, отчёт, запись файла
├── runtime.py        интерактивная установка Ollama и выбор модели
└── settings.py       .env + argv → model.Settings
```

---

## Как это работает

### Путь одной ячейки

1. **Разбор документов.** Все PDF читаются один раз. Документ привязывается к заёмщику по номеру
   счёта `ACC-*` в тексте — признак, который однозначно указывает на владельца даже при
   обезличенных именах файлов и отсеивает корпоративный шум вроде пресс-релизов.
2. **Сборка контекста.** Для каждого заёмщика: действующий договор, аудиторское дело, KYC-досье,
   строки леджера, связанные стороны, транзакции с пустой суммой. Здесь же применяются
   корректировки из окончательного заключения: переклассификации статей, исключения по
   отсечению периода, раскрытые курсы валют и суммы, которых нет в леджере отдельной
   операцией, — например поручительства и одобренные овердрафты.
3. **Оценка участниками.** Числовая модель и Gemma дают оценку с уверенностью по полям.
   Банк формул подбирает расчёт по тексту пункта: коэффициенты долговой нагрузки и покрытия,
   лимиты расходов по названной статье, квартальные ограничения, отношения к выручке.
   Отдельно поддержаны springing- и двойные тесты, где условие срабатывания и отчётный
   показатель — разные величины: в ответ идёт фактическое значение показателя, как того
   требуют правила организаторов.
4. **Разрешение.** Политика собирает ответ: числа от одного участника, семантика от другого.
5. **Сериализация.** `actual` приводится к модулю и округляется до двух знаков — расходы в
   леджере отрицательные, а по правилам значение всегда положительное.

### Кто за что отвечает

Участники решают **разные** вопросы, поэтому не конкурируют, а закрывают слабости друг друга.

| Что решается | Кто | Почему он |
|---|---|---|
| Действующий документ или устаревшая редакция | **gemma** | Признак текстовый, в шапке страницы |
| Определение ковенанта и порог | **gemma** | Формулировка уникальна у каждого заёмщика |
| Связанные стороны из KYC | **gemma** читает, **числовая** матчит | Таблица — текст, сопоставление — строгое |
| Категория операции по `description` | **gemma** | Колонки категории в леджере нет |
| `actual` | **числовая** | `Decimal`, без арифметики нейросети |
| `status` | **числовая** | Сравнение с порогом — вычисление, не суждение |
| `evidence_txn_id` | **числовая** | Перебор: какую строку убрать, чтобы вердикт перевернулся |

`NumericAuthoritativePolicy` разрешает расхождения по полям: числа берутся у детерминированного
участника, семантика — у языковой модели. Несовпадение вердиктов попадает в `has_disagreement`
и печатается в отчёте — это самый дешёвый признак того, что модель прочитала не тот документ.

Добавить третьего участника — значит написать класс с `Estimator`. Ни `ensemble.py`, ни политика
при этом не меняются.

### Принципы, зашитые в тесты

| Принцип | Где | Тест |
|---|---|---|
| `model/` не знает о `cli/` | запрет импортов, `argparse`, `print` | `test_boundaries.py` |
| `model/` импортируется автономно | копия пакета в чистом каталоге | `test_boundaries.py` |
| `services/` и `ensemble/` не выбирают адаптеры | зависимость только от `ports/` | `test_boundaries.py` |
| Сбой участника не топит ячейку | `try/except` вокруг каждого | `test_ensemble.py` |
| Валидный ответ есть всегда | дефолты кладутся до обработки | `test_ensemble.py` |
| Числа берёт числовая модель | политика разрешения | `test_ensemble.py` |
| Качество держится при правках | прогон против ключа | `test_regression.py` |

---

## Тесты

```bash
pytest -q                          # всё
pytest tests/test_boundaries.py    # только архитектурные границы
pytest tests/test_regression.py    # только end-to-end регрессия
```

Регрессионный тест прогоняет весь пайплайн и стережёт качество ответов при каждой правке.
Сети не требует.

---

<div align="center">

**Astrea** · Halyk AI Challenge

</div>
