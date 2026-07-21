# Database Knowledge Extractor

Скрипт для извлечения структурированных знаний о PostgreSQL-базе данных в JSON-формат.

Собирает:
- список таблиц и колонок с типами
- первичные (PK) и внешние ключи (FK)
- комментарии к таблицам и колонкам (`COMMENT ON`)
- статистики из `pg_stats` (n_distinct, most_common_vals, null_frac, гистограммы)
- пример строк (sample rows)

Результат — машинно-читаемый JSON, готовый для загрузки в LightRAG (Неделя 3 плана).

## Требования

1. **Python-зависимости:**
   ```bash
   pip install psycopg python-dotenv
   ```

2. **Переменные окружения** в `.env` в корне проекта:
   ```env
   DB_USER=postgres
   DB_PASS=your_password
   DB_HOST=localhost
   DB_PORT=5444
   ```

3. **Активный SSH-туннель** к БД (порт 5444 должен быть доступен локально).

## Быстрый старт

Обработать одну конкретную БД:
```bash
python scripts/explore_database.py --db financial
```

Результат появится в `db_knowledge/financial_knowledge.json`.

## Режимы работы

### Вариант 1: Одна конкретная БД
```bash
python scripts/explore_database.py --db financial
```

### Вариант 2: Все БД из BIRD dev
```bash
python scripts/explore_database.py --all-bird
```

### Вариант 3: Список БД из файла
Создай файл `databases.txt` в корне проекта:
```text
california_schools
toxicology
financial
# можно добавлять свои БД, комментарии через #
```
Запусти:
```bash
python scripts/explore_database.py
```

### Вариант 4: Свой конфиг и своя папка вывода
```bash
python scripts/explore_database.py --config my_list.txt --output-dir ./custom_output
```

## Формат выходного файла

Для каждой БД создаётся файл `<db_name>_knowledge.json` со структурой:
```json
{
  "tables": {
    "financial_accounts": {
      "table_type": "BASE TABLE",
      "comment": "Счета клиентов",
      "primary_keys": ["account_id"],
      "columns": [
        {
          "column_name": "account_id",
          "data_type": "integer",
          "is_nullable": false,
          "default": null,
          "max_length": null,
          "comment": "Уникальный идентификатор счёта"
        }
      ],
      "foreign_keys": [
        {
          "constraint_name": "fk_accounts_client",
          "column_name": "client_id",
          "foreign_table": "clients",
          "foreign_column": "client_id"
        }
      ],
      "column_statistics": {
        "account_id": {
          "n_distinct": -1.0,
          "null_frac": 0.0,
          "avg_width": 4,
          "most_common_vals": null,
          "most_common_freqs": null,
          "histogram_bounds": "{1,2500,5000,7500,10000}"
        }
      },
      "sample_rows": [
        {"account_id": 1, "client_id": 42, ...}
      ]
    }
  }
}
```

## Аргументы командной строки

| Аргумент | Описание |
|---|---|
| `--db NAME` | Имя одной БД для обработки |
| `--all-bird` | Обработать все стандартные БД из BIRD dev |
| `--config PATH` | Путь к файлу со списком БД (по умолчанию: `databases.txt`) |
| `--output-dir PATH` | Папка для сохранения JSON (по умолчанию: `db_knowledge/`) |
| `-h, --help` | Показать справку |