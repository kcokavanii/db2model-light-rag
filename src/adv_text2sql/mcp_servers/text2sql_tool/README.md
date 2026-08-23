# AdvText2SQL MCP server

MCP-сервер превращает Text-to-SQL pipeline проекта в стандартный инструмент,
который может вызвать MCP-клиент: IDE, агент или другое AI-приложение. Сервер
принимает вопрос на естественном языке, подбирает контекст схемы для одной
настроенной PostgreSQL-базы и возвращает сгенерированный SQL.

Сервер **не исполняет сгенерированный SQL** и не возвращает данные из БД. Он
проверяет, что ответ содержит ровно один read-only PostgreSQL-запрос, и может
дополнительно проверить построение плана через `EXPLAIN` без `ANALYZE`.

Поток одного вызова:

```text
MCP-клиент
  -> generate_sql(question, mode, evidence)
  -> выбор контекста схемы
  -> target LLM
  -> parse + read-only validation
  -> опциональный EXPLAIN в read-only транзакции
  -> SQL и диагностические metadata
```

## Предварительные условия

1. Запускайте сервер из корня репозитория.
2. Установите зависимости проекта командой `uv sync`. Она создаст `.venv`.
3. Настройте доступ к PostgreSQL. Для удалённой учебной БД должен быть активен
   SSH-туннель, описанный в корневом `README.md`.
4. Создайте `.env` в корне репозитория и не добавляйте его в Git.
5. Для retrieval-режимов заранее подготовьте LightRAG storage:
   `artifacts/lightrag/<db_name>`.
6. Для `structural` также нужен артефакт
   `artifacts/db_knowledge/<db_name>_knowledge.json`.

`baseline` не требует LightRAG-артефактов. Они также не нужны для первой
попытки `auto`, если полная схема не превышает token threshold, однако без них
сервер не сможет сделать structural fallback.

## Переменные окружения

Минимальная конфигурация через отдельные параметры подключения:

```env
MCP_DB_NAME=financial
DB_USER=benchmark
DB_PASS=replace_me
DB_HOST=localhost
DB_PORT=5444

LLM_MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
LLM_BASE_URL=https://example.invalid/v1
LLM_API_KEY=replace_me
```

Вместо `DB_USER`, `DB_PASS`, `DB_HOST` и `DB_PORT` можно передать полный
SQLAlchemy URI:

```env
MCP_DB_URI=postgresql+psycopg://user:password@localhost:5444/financial
```

Если в `MCP_DB_URI` есть имя БД, `MCP_DB_NAME` можно не задавать. Сервер всегда
специализирован на одной БД на процесс; MCP-клиент не может подменить URI или
имя БД аргументом tool-вызова. Если заданы обе переменные, имена БД должны
совпадать; URI другого SQL-диалекта сервер отклонит.

### Настройки моделей

| Переменная | Назначение | Fallback |
|---|---|---|
| `TARGET_LLM_MODEL_NAME` | target-модель для ambiguity check и SQL generation | `LLM_MODEL_NAME` |
| `TARGET_LLM_BASE_URL` | OpenAI-compatible endpoint target-модели | `LLM_BASE_URL` |
| `TARGET_LLM_API_KEY` | API key target-модели | `LLM_API_KEY` |
| `LIGHTRAG_LLM_MODEL_NAME` | модель keyword extraction в LightRAG | target-модель |
| `LIGHTRAG_LLM_BASE_URL` | endpoint retrieval-модели | target endpoint |
| `LIGHTRAG_LLM_API_KEY` | API key retrieval-модели | target API key |

Чтобы использовать одну и ту же модель для retrieval и генерации SQL, достаточно
задать базовые `LLM_*` переменные. Target-модель вызывается с `temperature=0`.

### Настройки сервера и маршрутизации

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `MCP_LIGHTRAG_STORAGE_DIR` | `artifacts/lightrag/<db_name>` | Путь к LightRAG storage |
| `MCP_DB_KNOWLEDGE_PATH` | `artifacts/db_knowledge/<db_name>_knowledge.json` | Путь к структурированным знаниям БД |
| `MCP_SCHEMA_TOKEN_THRESHOLD` | `150` | Порог полной схемы в токенах `cl100k_base` |
| `MCP_VALIDATE_EXPLAIN` | `true` | Включить проверку плана PostgreSQL |
| `MCP_EXPLAIN_TIMEOUT_MS` | `3000` | Timeout одного `EXPLAIN` |
| `MCP_TRANSPORT` | `stdio` | `stdio` или `http` |
| `MCP_HOST` | `127.0.0.1` | Адрес HTTP-сервера |
| `MCP_PORT` | `8000` | Порт HTTP-сервера |
| `LOG_LEVEL` | `INFO` | Уровень логирования Python |

Порог `150` — конфигурируемая эвристика router v1, а не доказанный оптимум.
Относительные artifact paths разрешаются от корня репозитория.

## Запуск

Команды ниже выполняются из корня репозитория.

### stdio

`stdio` подходит для клиента, который сам запускает MCP-процесс:

```powershell
.venv\Scripts\python.exe -m src.adv_text2sql.mcp_servers.text2sql_tool.main --transport stdio
```

На Linux/macOS путь к интерпретатору будет `.venv/bin/python`.

### HTTP

Локальный HTTP-сервер:

```powershell
.venv\Scripts\python.exe -m src.adv_text2sql.mcp_servers.text2sql_tool.main --transport http --host 127.0.0.1 --port 8000
```

Не публикуйте сервер наружу без отдельной аутентификации и сетевых ограничений.
Параметры CLI имеют приоритет над `MCP_TRANSPORT`, `MCP_HOST` и `MCP_PORT`.

### Один end-to-end smoke-вызов

После настройки `.env` и SSH-туннеля можно проверить полный MCP tool call без
отдельного HTTP-процесса:

```powershell
.venv\Scripts\python.exe -m src.adv_text2sql.mcp_servers.text2sql_tool.smoke_client --question "Сколько счетов есть в базе?" --mode auto
```

Это реальный, потенциально платный вызов target LLM и, в зависимости от
маршрута, LightRAG keyword-extraction LLM. Запускайте его сначала один раз, а
не на всём benchmark-наборе.

## MCP tool `generate_sql`

Сервер предоставляет один инструмент:

```text
generate_sql(
    question: str,
    mode: "auto" | "baseline" | "compact" | "structural" | "semantic_v3" = "auto",
    evidence: str | null = null,
)
```

- `question` — непустой вопрос о настроенной БД.
- `mode` — политика формирования schema context.
- `evidence` — необязательная подсказка: расшифровка кода, сокращения или BIRD
  evidence. Она добавляется к вопросу до retrieval и генерации.

Пример аргументов MCP-вызова:

```json
{
  "question": "Какие счета относятся к восточной Богемии?",
  "mode": "auto",
  "evidence": "east Bohemia refers to the district name"
}
```

## Режимы контекста

| Режим | Контекст | Retrieval |
|---|---|---|
| `baseline` | Полная схема: все table + column + type | Нет |
| `compact` | Только найденные table + column + type | Один LightRAG retrieval |
| `structural` | Найденная схема + PK/FK join paths + релевантные literal values, без длинных descriptions | Один LightRAG retrieval |
| `semantic_v3` | Полный semantic-контекст V3 LightRAG | Один LightRAG retrieval |
| `auto` | Дешёвый каскад между `baseline`, `compact` и `structural` | Зависит от размера схемы |

`structural` — V3-lite и текущий потолок автоматического режима. Этот формат
методологически обоснован, но его EX/VES пока не измерены, поэтому он считается
экспериментальным.

Сервер использует database-neutral target prompt версии `mcp_v1`: реальные
PostgreSQL-типы из контекста считаются источником истины. Исторический prompt
benchmark-прогонов не изменён, поэтому метрики опубликованных прогонов нельзя
автоматически приписывать MCP-режимам без отдельного воспроизводимого запуска.

`semantic_v3` сохранён для явного ручного запуска и абляций. **Режим `auto`
никогда его не выбирает**, в том числе при ошибке или неоднозначности.

## Политика `auto`

Сначала сервер считает размер полной сериализованной схемы в токенах
`cl100k_base`.

```text
full_schema_tokens <= threshold
└── baseline

full_schema_tokens > threshold
└── один LightRAG retrieval
    ├── одна таблица, нет FK path и literal values -> compact
    └── несколько таблиц, FK path или literal values -> structural
```

Если первая попытка возвращает `ambiguous`/`error`, не проходит parse,
read-only validation или включённый `EXPLAIN`, `auto` делает не более одного
fallback:

```text
baseline или compact -> structural
structural            -> baseline
```

Уже полученный retrieval result переиспользуется; второй retrieval для fallback
не запускается. Если retrieval недоступен или структурно непригоден, сервер
выбирает полную схему и записывает причину в ответ. Явно выбранные режимы не
делают автоматического fallback.

Такая маршрутизация экономит контекст на широких схемах, но не гарантирует
семантическую правильность SQL. Без gold SQL сервер способен обнаружить
синтаксическую ошибку, write-операцию, неизвестную таблицу/колонку при
`EXPLAIN` и ambiguity-сигнал модели, но не доказать, что запрос правильно
отвечает на вопрос.

В частности, условие «все нужные поля найдены» нельзя доказать по одному
retrieval result без gold-разметки: router приближённо считает результат
compact-кандидатом, если найдена одна таблица и нет FK/literal-сигналов. Это
нужно отдельно проверить на сохранённом evaluation-наборе.

## Формат ответа

Инструмент возвращает словарь со следующими основными полями:

| Поле | Описание |
|---|---|
| `status` | `success`, `ambiguous` или `error` |
| `query` | SQL при успехе, иначе `null` |
| `database` | Настроенная БД |
| `requested_mode` | Запрошенный режим |
| `initial_mode` | Режим первой попытки после routing |
| `effective_mode` | Режим последней попытки |
| `fallback_used` | Был ли выполнен один fallback |
| `routing_reason` | Проверяемая причина решения router |
| `attempts` | Статус, ошибка и размер контекста каждой попытки |
| `error` | Ошибка последней неуспешной попытки |
| `metadata` | Версии router/prompt, token usage, retrieval diagnostics и параметры validation |

В `metadata` всегда возвращается `sql_executed: false`. Если включённая
EXPLAIN-проверка прошла, `explain_validated` будет `true`. Token usage зависит
от того, возвращает ли используемый OpenAI-compatible endpoint usage-поля.

## Безопасность и ограничения

- Разрешён ровно один PostgreSQL read-only query expression (`SELECT`, CTE,
  set operation).
- DML, DDL, команды и несколько statements отклоняются до ответа клиенту.
- Locking reads (`FOR UPDATE`/`FOR SHARE`) также отклоняются.
- `EXPLAIN (FORMAT JSON)` выполняется без `ANALYZE`, в read-only транзакции и с
  ограничением времени. Он строит план, но не исполняет пользовательский SQL.
- `EXPLAIN` проверяет синтаксис, имена объектов и возможность построить план,
  но не логическую эквивалентность вопросу.
- Общий mutable schema context защищён lock: запросы одного процесса
  обрабатываются последовательно.
- В ответ не включается сырой schema context, который может содержать
  чувствительные literal values.

AST-проверка не способна доказать отсутствие побочных эффектов у произвольной
пользовательской PostgreSQL-функции. Поэтому MCP сам SQL не исполняет; клиенту,
который захочет исполнить результат, всё равно нужны read-only роль, транзакция
и timeout.

Для полностью автономной работы без обращения к PostgreSQL после introspection
можно установить `MCP_VALIDATE_EXPLAIN=false`, но первичное подключение к БД всё
равно необходимо, чтобы прочитать полную схему.
