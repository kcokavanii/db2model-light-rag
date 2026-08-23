# Скрипты исследовательского pipeline

Все команды запускаются из корня репозитория через окружение проекта:

```powershell
uv run --env-file .env python <путь-к-скрипту> <аргументы>
```

Полный воспроизводимый маршрут и список benchmark-артефактов приведены в
[`../README.md`](../README.md). Здесь собрана краткая справка по каждому
Python-скрипту в каталоге `scripts/`.

## Последовательность pipeline

```text
PostgreSQL
  -> explore_database.py
  -> *_knowledge.json
  -> generate_m_schema.py / run_week2_exploration.py
  -> *_m_schema.txt + *_semantic.txt
  -> build_db_graph.py
  -> *_graph.pkl + *.graphml + *_graph_stats.json
  -> load_to_lightrag.py
  -> LightRAG storage
  -> query_lightrag.py или run_week3_lightrag.py
```

## `explore_database.py`

Извлекает из одной PostgreSQL-базы таблицы, колонки, типы, PK/FK, комментарии,
`pg_stats` и примеры строк.

```powershell
uv run --env-file .env python scripts/explore_database.py --db financial
```

Параметры:

| Параметр | Назначение |
|---|---|
| `--db NAME` | Имя целевой PostgreSQL-базы; для рабочего запуска указывать обязательно |
| `--output-dir PATH` | Папка результата; default: `artifacts/db_knowledge` |

Основной результат: `artifacts/db_knowledge/financial_knowledge.json`.
Скрипт читает строки для статистик и sample rows, но не вызывает LLM.

## `generate_m_schema.py`

Преобразует один `*_knowledge.json` или все такие JSON из указанного каталога в
две текстовые схемы:

```powershell
uv run --env-file .env python scripts/generate_m_schema.py `
  --input artifacts/db_knowledge/financial_knowledge.json `
  --output artifacts/m_schemas
```

| Параметр | Назначение |
|---|---|
| `-i`, `--input` | JSON-файл или каталог с `*_knowledge.json` |
| `-o`, `--output` | Выходной каталог |

Результаты: `<db>_m_schema.txt`, `<db>_semantic.txt` и общий лог
`artifacts/generate_m_schema.log`. Семантические описания генерирует
Teacher-LLM, поэтому шаг требует `LLM_*` и расходует токены.

## `run_week2_exploration.py`

Оркестратор первых двух шагов: запускает `explore_database.py`, проверяет
созданный JSON и затем запускает `generate_m_schema.py`.

```powershell
uv run --env-file .env python scripts/run_week2_exploration.py --db financial
```

Единственный параметр `--db` обязателен. Результаты сохраняются в
`artifacts/db_knowledge/` и `artifacts/m_schemas/`.

## `build_db_graph.py`

Строит NetworkX-граф из structured knowledge и обеих M-Schema.

```powershell
uv run --env-file .env python scripts/build_db_graph.py --db financial --inspect
```

| Параметр | Назначение |
|---|---|
| `--db NAME` | Целевая БД |
| `--no-llm` | Не проверять semantic edges через Teacher-LLM |
| `--inspect` | Напечатать примеры узлов построенного графа |
| `--verify-all` | Проверить Teacher-LLM все semantic edge candidates |

Результаты в `artifacts/graphs/`: `<db>_graph.pkl`, `<db>_graph.graphml` и
`<db>_graph_stats.json`.

## `load_to_lightrag.py`

Конвертирует `<db>_graph.pkl` в LightRAG custom KG и создаёт постоянный storage.

```powershell
uv run --env-file .env python scripts/load_to_lightrag.py --db financial
```

Единственный параметр `--db` обязателен. Источник:
`artifacts/graphs/<db>_graph.pkl`. Результат:
`artifacts/lightrag/<db>/`.

## `query_lightrag.py`

Выполняет один retrieval по готовому storage и печатает найденные entities,
relations и форматы schema context.

```powershell
uv run --env-file .env python scripts/query_lightrag.py `
  --db financial `
  --query "Какие счета относятся к восточной Богемии?"
```

Параметры `--db` и `--query` обязательны. Скрипт не записывает новый артефакт.
LightRAG keyword extraction может использовать LLM.

## `run_week3_lightrag.py`

Запускает BIRD Text-to-SQL benchmark с Compact LightRAG-контекстом. Target
prompt и evaluator совпадают с baseline runner; вместо полной схемы передаются
только найденные table, column и type.

```powershell
uv run --env-file .env python scripts/run_week3_lightrag.py `
  --db financial `
  --question-ids 119 135
```

| Параметр | Назначение |
|---|---|
| `--db NAME` | Целевая БД |
| `--dataset PATH` | Dataset; default: `data/bird_large_filtered.json` |
| `--question-ids ID ...` | Только перечисленные BIRD question IDs |
| `--db-url HOST:PORT` | PostgreSQL; default: `localhost:5444` |
| `--storage-dir PATH` | Исходный storage; default: `artifacts/lightrag/<db>` |
| `--output-dir PATH` | Новая или пустая папка результата |
| `--reuse-keyword-cache` | Использовать исходный storage и накопленный cache |
| `--verbose-retrieval` | Печатать все найденные entities и relations |

Без `--output-dir` создаётся
`artifacts/benchmarks/lightrag/<db>/<timestamp>/`. Полный список файлов этой
папки описан в корневом README.

## `profile_bird_schemas.py`

Считает размер baseline-схем, количество таблиц/колонок/FK и
`context_tokens_cl100k_per_query`. Работает в read-only транзакции, не читает
строки и не вызывает LLM.

```powershell
uv run --env-file .env python scripts/profile_bird_schemas.py `
  --dataset data/bird_large_filtered.json `
  --db toxicology financial codebase_community `
  --output artifacts/schema_profiles/bird_large_filtered_schema_profile.csv
```

| Параметр | Назначение |
|---|---|
| `--dataset PATH` | Dataset для списка БД и числа вопросов |
| `--db NAME ...` | Необязательное подмножество БД |
| `--db-url HOST:PORT` | PostgreSQL endpoint |
| `--connect-timeout SEC` | Timeout подключения; default: 5 секунд |
| `--output PATH` | Выходной CSV |

## `extract_subgraph.py`

Ранний prototype локального embedding + BFS retrieval. Текущая точка входа
жёстко использует `artifacts/graphs/financial_graph.pkl`, выполняет четыре
демонстрационных запроса и печатает результат:

```powershell
uv run --env-file .env python scripts/extract_subgraph.py
```

Скрипт не создаёт файлов и не входит в текущий benchmark/MCP pipeline. Для
актуальной ручной проверки используйте `query_lightrag.py`.

## Карта артефактов

| Этап | Артефакт |
|---|---|
| PostgreSQL exploration | `artifacts/db_knowledge/<db>_knowledge.json` |
| M-Schema | `artifacts/m_schemas/<db>_m_schema.txt` |
| Semantic descriptions | `artifacts/m_schemas/<db>_semantic.txt` |
| NetworkX graph | `artifacts/graphs/<db>_graph.pkl` |
| Просмотр графа | `artifacts/graphs/<db>_graph.graphml` |
| Статистика графа | `artifacts/graphs/<db>_graph_stats.json` |
| LightRAG storage | `artifacts/lightrag/<db>/` |
| Профиль схем | `artifacts/schema_profiles/*.csv` |
| LightRAG benchmark | `artifacts/benchmarks/lightrag/<db>/<timestamp>/` |

Проверить созданные файлы:

```powershell
Get-ChildItem artifacts -Recurse -File
```
