# AdvText2SQL

AdvText2SQL — исследовательский проект по преобразованию вопросов на естественном
языке в PostgreSQL-запросы. В репозитории сравниваются два основных подхода:

- **baseline** — target-LLM получает полную схему базы данных;
- **LightRAG** — перед генерацией SQL из графа знаний извлекается только
  релевантная часть схемы.

Выбранные для основного сравнения базы BIRD: `toxicology`, `financial` и
`codebase_community`. Зафиксированный evaluation-набор находится в
`data/bird_large_filtered.json` и содержит 91 вопрос. История экспериментов,
точные конфигурации и полученные EX/VES записаны в
[`research_log.md`](research_log.md).

В конце pipeline выбранное решение упаковано в MCP-сервер. Он принимает вопрос,
подбирает schema context и возвращает безопасный read-only SQL, который можно
передать вызывающему приложению.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `src/adv_text2sql/` | Text-to-SQL и MCP-код |
| `scripts/` | Извлечение знаний, M-Schema, граф, LightRAG и исследовательские запуски |
| `benchmarks/` | Генерация ответов и расчёт BIRD/Ambrosia метрик |
| `data/` | Локальные benchmark-наборы |
| `artifacts/` | Сгенерированные знания, графы, storage и результаты запусков |
| `tests/mcp/` | Локальные тесты MCP-маршрутизации и SQL safety |
| `research_log.md` | Журнал гипотез, конфигураций и результатов |

`artifacts/` исключён из Git. Артефакты эксперимента нужно хранить отдельно или
добавлять в коммит только явно выбранные компактные сводки.

## Установка

Требования:

- Python 3.11 или новее;
- `uv`;
- доступ к PostgreSQL с read-only пользователем;
- OpenAI-compatible endpoint для LLM;
- интернет при первой загрузке embedding-модели `BAAI/bge-m3`.

Установите `uv`, клонируйте репозиторий и установите зависимости:

```powershell
python -m pip install uv
git clone https://github.com/deeppavlov/AdvText2SQL
Set-Location AdvText2SQL
uv sync --all-groups
```

Все дальнейшие команды выполняются из корня репозитория.

Создайте локальный `.env`:

```powershell
Copy-Item .env.example .env
```

Заполните как минимум:

```env
DB_USER=benchmark
DB_PASS=replace_me
DB_HOST=localhost
DB_PORT=5444

LLM_MODEL_NAME=Qwen2.5-Coder-7B-Instruct
LLM_BASE_URL=https://example.invalid/v1
LLM_API_KEY=replace_me
```

Для честного сравнения можно отдельно задать `TARGET_LLM_*` и
`LIGHTRAG_LLM_*`. Если они не заданы, код использует соответствующие `LLM_*`
значения. Не добавляйте `.env` и ключи в Git.

## Подключение к PostgreSQL

Репозиторий не содержит публичных реквизитов или дампа готового PostgreSQL-
сервера. Доступ к `lnsigo.mipt.ru` есть только у участников учебного проекта,
которым выданы личная учётная запись и пароль.

### Если есть доступ к учебному серверу

Откройте SSH-туннель в отдельном терминале, подставив имя своей личной учётной
записи:

```powershell
ssh -N -L 5444:10.11.1.6:5444 user_name@lnsigo.mipt.ru -p 2278
```

После запуска `ssh` запросит пароль от личной учётной записи на
`lnsigo.mipt.ru`. При вводе пароль не отображается в терминале — это нормальное
поведение. SSH-пароль не является паролем PostgreSQL (`DB_PASS`), и его не нужно
записывать в `.env`, команду или документацию.

Проверьте подключение:

```powershell
psql -d postgres -U benchmark --host localhost --port 5444
```

В консоли PostgreSQL команда `\l` должна показать список баз. Выйти можно
командой `\q`. Для pipeline и evaluation используйте read-only роль.

### Если доступа к учебному серверу нет

Необходимо самостоятельно развернуть PostgreSQL и импортировать в него базы
BIRD, с которыми будет запускаться проект. Для воспроизведения основного
эксперимента нужны базы `toxicology`, `financial` и `codebase_community`.

Общий порядок:

1. Установить или запустить PostgreSQL, например в Docker.
2. Скачать BIRD dev с официального сайта и импортировать выбранные SQLite-базы
   в PostgreSQL.
3. Создать отдельного пользователя только для чтения и включить для него
   `default_transaction_read_only`.
4. Указать адрес собственного сервера и реквизиты read-only пользователя в
   `DB_HOST`, `DB_PORT`, `DB_USER` и `DB_PASS` файла `.env`.
5. Проверить, что имена баз в PostgreSQL совпадают с `db_id` в dataset.

Черновые вспомогательные файлы миграции и служебная инструкция находятся в
[`data/launch_db/`](data/launch_db/README.md). Это не готовый deployment:
`docker-compose.yml`, данные BIRD и реквизиты доступа в репозиторий не входят.
Пользователь получает dataset из официального источника, подготавливает
конфигурацию PostgreSQL и разворачивает сервер самостоятельно.

## Быстрый воспроизводимый pipeline для одной БД

Ниже показан полный путь для `financial`. Часть шагов вызывает LLM и может быть
платной; такие места отмечены явно.

### 1. Извлечь структурированные знания из PostgreSQL

```powershell
uv run --env-file .env python scripts/explore_database.py --db financial
```

Результат:

```text
artifacts/db_knowledge/financial_knowledge.json
```

JSON содержит таблицы, колонки, типы, PK/FK, комментарии, статистики и до пяти
примеров строк. Этот шаг читает схему и данные PostgreSQL, но LLM не вызывает.

### 2. Построить M-Schema и семантические описания

Рекомендуемая команда объединяет извлечение знаний и генерацию обеих схем:

```powershell
uv run --env-file .env python scripts/run_week2_exploration.py --db financial
```

Результаты:

```text
artifacts/db_knowledge/financial_knowledge.json
artifacts/m_schemas/financial_m_schema.txt
artifacts/m_schemas/financial_semantic.txt
artifacts/generate_m_schema.log
```

`financial_m_schema.txt` создаётся детерминированно из JSON.
`financial_semantic.txt` создаётся Teacher-LLM и расходует токены.

Если JSON уже существует, генератор можно запустить отдельно:

```powershell
uv run --env-file .env python scripts/generate_m_schema.py `
  --input artifacts/db_knowledge/financial_knowledge.json `
  --output artifacts/m_schemas
```

### 3. Построить граф знаний

Полный вариант с LLM-проверкой семантических связей:

```powershell
uv run --env-file .env python scripts/build_db_graph.py --db financial --inspect
```

Быстрый отладочный вариант без дополнительной LLM-проверки:

```powershell
uv run --env-file .env python scripts/build_db_graph.py --db financial --no-llm --inspect
```

Результаты:

```text
artifacts/graphs/financial_graph.pkl
artifacts/graphs/financial_graph.graphml
artifacts/graphs/financial_graph_stats.json
```

Флаг `--verify-all` проверяет через Teacher-LLM все найденные семантические
кандидаты и стоит дороже. Для воспроизводимого сравнения обязательно фиксируйте,
какой из трёх вариантов использован: default, `--no-llm` или `--verify-all`.

### 4. Загрузить граф в LightRAG storage

```powershell
uv run --env-file .env python scripts/load_to_lightrag.py --db financial
```

Результаты появляются в `artifacts/lightrag/financial/`:

```text
graph_chunk_entity_relation.graphml
kv_store_text_chunks.json
vdb_chunks.json
vdb_entities.json
vdb_relationships.json
```

### 5. Вручную проверить retrieval

```powershell
uv run --env-file .env python scripts/query_lightrag.py `
  --db financial `
  --query "Какие счета относятся к восточной Богемии?"
```

Скрипт печатает извлечённый подграф и сформированные контексты в терминал. Он не
создаёт новый артефакт. Keyword extraction может вызывать LLM.

## Получение артефактов для всех выбранных БД

Следующий PowerShell-фрагмент последовательно создаёт знания, M-Schema, графы и
LightRAG storage для трёх основных БД:

```powershell
$databases = "toxicology", "financial", "codebase_community"

foreach ($database in $databases) {
    uv run --env-file .env python scripts/run_week2_exploration.py --db $database
    uv run --env-file .env python scripts/build_db_graph.py --db $database
    uv run --env-file .env python scripts/load_to_lightrag.py --db $database
}
```

Этот запуск вызывает Teacher-LLM и загружает embedding-модель. Перед полным
запуском разумно пройти все шаги на одной БД и проверить файлы:

```powershell
Get-ChildItem artifacts -Recurse -File
```

## Профилирование размера схем

Скрипт отражает baseline-схему точно в том формате, который получает модель,
считает токены `cl100k_base` и не читает строки таблиц и не вызывает LLM:

```powershell
uv run --env-file .env python scripts/profile_bird_schemas.py `
  --dataset data/bird_large_filtered.json `
  --db toxicology financial codebase_community `
  --output artifacts/schema_profiles/bird_large_filtered_schema_profile.csv
```

Результат:

```text
artifacts/schema_profiles/bird_large_filtered_schema_profile.csv
```

## BIRD baseline

Дешёвый ручной запуск: два вопроса `financial` из малого набора.

```powershell
$run = Get-Date -Format "yyyyMMdd_HHmmss"
uv run --env-file .env python bird_benchmark.py `
  --dataset data/bird_small.json `
  --db financial `
  --output-dir "artifacts/benchmarks/baseline/financial/$run"
```

Полный зафиксированный набор из 91 вопроса:

```powershell
$run = Get-Date -Format "yyyyMMdd_HHmmss"
uv run --env-file .env python bird_benchmark.py `
  --dataset data/bird_large_filtered.json `
  --output-dir "artifacts/benchmarks/baseline/all/$run"
```

Полный запуск платный и исполняет predicted и gold SQL для расчёта EX/VES.
Папка, переданная через `--output-dir`, должна быть новой или пустой.

Baseline создаёт:

| Файл | Содержимое |
|---|---|
| `db_schemas.json` | Полные schema context, использованные генератором |
| `query_results.json` | Сгенерированный SQL по `question_id` |
| `all_gold_results.json` | Результаты исполнения gold SQL |
| `all_predicted_results.json` | Результаты исполнения сгенерированного SQL |
| `manual_check.json` | Построчное сравнение для ручной проверки |
| `baseline.csv` | EX, VES, target prompt tokens и schema tokens |

## BIRD + LightRAG Compact

Smoke-прогон на двух вопросах:

```powershell
uv run --env-file .env python scripts/run_week3_lightrag.py `
  --db financial `
  --question-ids 119 135
```

Полный прогон одной БД запускается без `--question-ids`:

```powershell
uv run --env-file .env python scripts/run_week3_lightrag.py --db financial
```

Для всех трёх БД:

```powershell
$databases = "toxicology", "financial", "codebase_community"

foreach ($database in $databases) {
    uv run --env-file .env python scripts/run_week3_lightrag.py --db $database
}
```

По умолчанию каждый прогон получает новую timestamp-папку:

```text
artifacts/benchmarks/lightrag/<db>/<YYYYMMDD_HHMMSS>/
```

В ней сохраняются:

| Файл или папка | Содержимое |
|---|---|
| `bird_selected.json` | Точный snapshot выбранных вопросов |
| `run_config.json` | Модели, commit, dirty-state и retrieval-конфигурация |
| `lightrag_storage/` | Изолированная копия storage без общего keyword cache |
| `db_schemas.json` | Контекст, переданный target-модели |
| `contexts.json` | Retrieval diagnostics и размеры контекста по вопросам |
| `query_results.json` | Сгенерированный SQL |
| `all_gold_results.json` | Результаты gold SQL |
| `all_predicted_results.json` | Результаты predicted SQL |
| `manual_check.json` | Данные для ручной сверки EX |
| `token_usage.json` | Target, retrieval и combined token usage |
| `lightrag.csv` | Итоговые EX, VES и token budget |

Текущий `run_week3_lightrag.py` измеряет **Compact**: найденные table, column и
type без длинных семантических описаний. Он переиспользует baseline target prompt
и evaluator; исследуемая переменная — schema context и дополнительный retrieval.

Флаг `--reuse-keyword-cache` ускоряет повторный прогон, но делает его зависимым
от ранее накопленного LLM cache. Для финального честного сравнения оставляйте
default: скрипт создаст локальную копию storage без keyword cache.

## MCP-сервер

MCP-сервер находится в
`src/adv_text2sql/mcp_servers/text2sql_tool`. Полная инструкция по переменным
окружения, режимам `baseline`, `compact`, `structural`, `semantic_v3`, политике
`auto`, формату ответа и ручному smoke-вызову находится в
[`src/adv_text2sql/mcp_servers/text2sql_tool/README.md`](src/adv_text2sql/mcp_servers/text2sql_tool/README.md).

Минимальный локальный запуск через `stdio`:

```powershell
uv run --env-file .env python -m src.adv_text2sql.mcp_servers.text2sql_tool.main --transport stdio
```

Ручной end-to-end вызов одного MCP tool:

```powershell
uv run --env-file .env python -m src.adv_text2sql.mcp_servers.text2sql_tool.smoke_client `
  --question "Сколько счетов есть в базе?" `
  --mode auto
```

MCP генерирует и статически проверяет SQL, а при включённой настройке выполняет
только `EXPLAIN` без `ANALYZE`. Сам SQL он не исполняет и EX/VES не считает.
Исполнение predicted и gold SQL и расчёт метрик выполняют benchmark-скрипты.
Поэтому результаты baseline или LightRAG benchmark нельзя автоматически
приписывать MCP: `auto` меняет формат контекста, а MCP использует отдельную
версию prompt `mcp_v1`.

## Остальные и экспериментальные точки входа

- `scripts/extract_subgraph.py` — ранний локальный prototype retrieval. Он
  использует жёстко заданный `artifacts/graphs/financial_graph.pkl`, печатает
  четыре демонстрационных контекста и не создаёт файлы. В основном pipeline
  вместо него используется `query_lightrag.py`.
- `bird_evaluate_only.py` — повторно оценивает уже существующий
  `query_results.json` в корне репозитория без нового LLM-вызова. Требует
  `BENCHMARK_DB_URL=localhost:5444` и перезаписывает корневые
  `all_gold_results.json`/`all_predicted_results.json`; для новых экспериментов
  безопаснее использовать изолированные artifact-папки основных runner-ов.
- `ambrosia_benchmark.py` — legacy smoke-run на `data/ambrosia_small.json`.
  Он пишет `db_schemas.json` и `query_results.json` в текущую директорию и не
  относится к зафиксированному сравнению LightRAG на BIRD.
- `data/test_permissions.py` — проверка read-only роли PostgreSQL. Скрипт
  намеренно пытается выполнить write-команды и безопасен только при корректно
  настроенной read-only роли; требуется `BENCHMARK_DB_URL`.

Подробное назначение каждого файла в `scripts/` и его параметры приведены в
[`scripts/README.md`](scripts/README.md).

## Проверки

Локальные тесты MCP не вызывают LLM и не требуют PostgreSQL:

```powershell
uv run --group test python -m unittest discover -s tests/mcp -v
uv run ruff check src/adv_text2sql/mcp_servers tests/mcp
uv run pyrefly check src/adv_text2sql/mcp_servers
```

После изменения evaluation logic сначала проведите smoke-run на 1–2 вопросах и
вручную сверьте `manual_check.json`. Не сравнивайте запуски с разными split,
target-моделями, prompt-версиями или правилами подсчёта токенов как один
эксперимент.

## Данные и источники

- BIRD dev: [bird-bench.github.io](https://bird-bench.github.io/)
- Ambrosia: [ambrosia-benchmark.github.io](https://ambrosia-benchmark.github.io/)
- `data/bird_small.json` — дешёвый smoke-набор;
- `data/bird_large_filtered.json` — основной зафиксированный набор проекта;
- `data/bird_large.json` — более широкий локальный набор;
- `data/train_queries.json` и `data/ambrosia_train.json` — обучающие данные,
  которые нельзя подмешивать в evaluation.
