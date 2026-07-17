import logging
import os
import re
from textwrap import dedent
from typing import Any

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Result
from sqlglot import Dialects, exp, parse_one
from sqlglot.errors import ParseError

from .prompts import (
    AMBIGUITY_PROMPT_TEMPLATE,
    SQL_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    VERIFICATION_PROMPT_TEMPLATE,
)
from .utils import print_result

# Load environment variables for the main block
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 7))


logger = logging.getLogger("text2sql_tool")


class Text2SQLGenerator:
    def __init__(self, db_uri: str, llm_client: OpenAIChatCompletionClient):
        """
        Initializes the Text2SQL generator with a database URI.

        Args:
            db_uri (str): SQL database URI.
            llm_client (openai.AsyncOpenAI): An initialized asynchronous OpenAI client.
        """
        self.db_uri = db_uri
        self.engine: Engine = create_engine(
            db_uri,
            pool_pre_ping=True,
        )

        self.llm_client = llm_client

        logger.info("Initialized Text2SQLGenerator")

    def build(self):
        self.db_schema = self._get_db_schema_light()
        # self.db_schema = self._get_db_schema_heavy()
        self.system_prompt = self._create_system_prompt()

    def _update_db_schema(self, db_uri):
        self.db_uri = db_uri
        self.engine.dispose()
        self.engine = create_engine(db_uri, pool_pre_ping=True)
        self.build()

    def _get_db_schema_heavy(
        self,
        sample_rows_limit: int = 3,
        get_unique_values: bool = True,
        unique_threshold: int = 10,
        sample_tables_with_more_rows: dict | None = None,
    ) -> str:
        """
        Extract db_schema with a few more features (they all spend tokens).

        Get PostgreSQL schema info with:
        - table definitions (columns + types)
        - sample rows
        - unique values for low-cardinality columns (optional)
        """

        # print(self.db_uri)
        if sample_tables_with_more_rows is None:
            sample_tables_with_more_rows = {}

        inspector = inspect(self.engine)
        schema_parts: list[str] = []
        unique_values_parts: list[str] = []

        with self.engine.connect() as conn:
            tables = inspector.get_table_names()

            for table in tables:
                schema_parts.append(f"-- Table: {table}")

                # 1. Columns
                columns = inspector.get_columns(table)
                schema_parts.append(f"CREATE TABLE {table} (")
                for col in columns:
                    schema_parts.append(f"    {col['name']} {col['type']},")
                schema_parts.append(");")

                # 2. Sample rows
                limit = sample_tables_with_more_rows.get(table, sample_rows_limit)
                try:
                    result = conn.execute(
                        text(f'SELECT * FROM "{table}" LIMIT :limit'),
                        {"limit": limit},
                    )
                    rows = result.fetchall()

                    if rows:
                        col_names = result.keys()
                        schema_parts.append("/*")
                        schema_parts.append("\t".join(col_names))
                        for row in rows:
                            schema_parts.append("\t".join(map(str, row)))
                        schema_parts.append("*/")

                except Exception:
                    logger.exception(f"Could not fetch sample rows for table {table}")

                schema_parts.append("\n")

                # 3. Unique values per column (low-cardinality only)
                if not get_unique_values:
                    continue

                for col in columns:
                    col_name = col["name"]

                    try:
                        res = conn.execute(
                            text(
                                f'''
                                SELECT DISTINCT "{col_name}"
                                FROM "{table}"
                                WHERE "{col_name}" IS NOT NULL
                                '''
                            )
                        )
                        values = [r[0] for r in res.fetchall()]

                        if 0 < len(values) <= unique_threshold:
                            formatted = ", ".join(repr(v) for v in values)
                            unique_values_parts.append(
                                f'Possible values for "{table}.{col_name}": [{formatted}]'
                            )

                    except Exception:
                        # silently skip problematic columns (JSON, arrays, etc.)
                        continue

        if unique_values_parts:
            schema_parts.append("\n### Low-cardinality value hints")
            schema_parts.extend(unique_values_parts)

        return "\n".join(schema_parts).strip()

    def _get_db_schema_light(self) -> str:
        """
        Lightweight schema extraction for LLM prompts.

        Includes:
        - table names
        - column names
        - column types
        """
        inspector = inspect(self.engine)
        schema_parts = []

        # Default schema for PostgreSQL
        # print(self.db_uri)
        tables = inspector.get_table_names(schema="public")
        if not tables:
            print("this db failed: ", self.db_uri)
            raise RuntimeError("No tables found in public schema")

        for table in tables:
            schema_parts.append(f"TABLE {table}")

            columns = inspector.get_columns(table, schema="public")
            for col in columns:
                col_type = str(col["type"])
                schema_parts.append(f"  - {col['name']} ({col_type})")

            schema_parts.append("")

        return "\n".join(schema_parts).strip()

    def _create_system_prompt(self) -> str:
        """Создает системный промпт с описанием схемы БД"""
        return SYSTEM_PROMPT_TEMPLATE.format(
            db_schema=self.db_schema, sql_dialect="PostgreSQL"
        )

    async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
        """
        Проверяет пользовательский запрос на неоднозначность с помощью LLM.
        """
        ambiguity_prompt = AMBIGUITY_PROMPT_TEMPLATE.format(
            db_schema=self.db_schema,
            user_query=user_query,
        )

        messages = [
            SystemMessage(content=ambiguity_prompt),
            UserMessage(
                source="user",
                content="Проверь запрос на однозначность.",
            ),
        ]

        """
        content=(
            "Определи, можно ли вообще ответить на этот вопрос, "
            "используя ТОЛЬКО данную схему БД. "
            "Если ответить нельзя — напиши NOT_ANSWERABLE. "
            "Если ответ возможен, но запрос неоднозначен — опиши, что нужно уточнить. "
            "Если запрос полностью однозначен и ответим — напиши OK."
        ),
        """

        try:
            result = await self.llm_client.create(messages)
            response_text = result.content.strip()

            if response_text.lower() == "ok":
                return {"status": "success", "ambiguous": False}
            else:
                return {
                    "status": "success",
                    "ambiguous": True,
                    "clarification_needed": response_text,
                }

        except Exception as e:
            logger.exception(f"Failed to check ambiguity for query: {user_query}")
            return {"status": "error"}

    def _strip_sql_comments(self, sql: str) -> str:
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.IGNORECASE)

        return sql.strip()

    def _is_sql_complete(self, sql: str) -> bool:
        upper = sql.upper().strip()

        required = ["SELECT", "FROM"]
        if not all(k in upper for k in required):
            return False

        if upper.endswith(
            (
                "WHERE",
                "JOIN",
                "FROM",
                "ON",
                "AND",
                "OR",
            )
        ):
            return False

        return True

    def _validate_sql(self, sql: str) -> bool:
        """Валидация SQL запроса с помощью SQLGlot"""
        try:
            if not self._is_sql_complete(sql):
                raise ValueError("Incomplete SQL generated")

            parsed = parse_one(sql, dialect=Dialects.POSTGRES)

            # Проверка на запрещенные операции
            for node in parsed.walk():
                if isinstance(node, (exp.Drop | exp.Delete | exp.Update)):
                    raise ValueError(f"Запрещенная операция: {node.sql()}")

            return True
        except (ParseError, ValueError):
            logger.exception("SQL validation error.")
            return False

    def sanitize_sql(self, sql: str) -> str:
        """
        Minimal SQL sanitizer for Postgres execution.
        Fixes common LLM-generated SQL issues:
        1. Strips markdown/code fences
        2. Wraps DISTINCT queries with ORDER BY in a subquery
        3. Normalizes CAST and NULLIF usage
        """
        sql = sql.strip()

        # Remove markdown/code fences
        sql = re.sub(r"^```[a-z]*|```$", "", sql, flags=re.IGNORECASE).strip()

        # Detect DISTINCT + ORDER BY and wrap in a subquery
        distinct_order_by_pattern = re.compile(
            r"SELECT\s+DISTINCT\s+(.*?)\s+FROM\s+(.*?)\s+ORDER\s+BY\s+(.*?)(LIMIT\s+\d+)?;",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = distinct_order_by_pattern.search(sql)
        if match:
            select_cols, from_clause, order_by_clause, limit_clause = match.groups()
            limit_clause = limit_clause or ""
            sql = f"""
            SELECT *
            FROM (
                SELECT DISTINCT {select_cols}
                FROM {from_clause}
            ) sub
            ORDER BY {order_by_clause} {limit_clause};
            """.strip()

        # Replace CAST(... AS REAL) if inside math operations
        sql = re.sub(
            r"CAST\((.*?)\s+AS\s+REAL\)", r"\1::REAL", sql, flags=re.IGNORECASE
        )
        # Remove invalid nested NULLIF patterns like NULLIF(NULLIF,0)(x)
        sql = re.sub(
            r"NULLIF\(NULLIF,0\)\((.*?)\)", r"NULLIF(\1,0)", sql, flags=re.IGNORECASE
        )

        # remove multiple newlines
        sql = re.sub(r"\n\s*\n", "\n", sql)

        return sql

    async def generate_sql(self, user_query: str) -> dict[str, Any]:
        """
        Генерирует SQL запрос из естественно-языкового запроса

        Args:
            user_query (str): Запрос на естественном языке

        Returns:
            Dict[str, Any]: Результат в формате Model Context Protocol
        """
        sql_prompt = SQL_PROMPT_TEMPLATE.format(
            user_query=user_query, sql_dialect="PostgreSQL"
        )
        try:
            # Создаем цепочку обработки запроса
            messages = [
                SystemMessage(content=self.system_prompt),
                UserMessage(source="user", content=dedent(sql_prompt)),
            ]

            # Прямой вызов LLM
            result = await self.llm_client.create(messages)
            raw_sql = result.content.strip()

            # Очистка и валидация
            # sanitized_sql = self._sanitize_sql(raw_sql) # TODO сделать нормально
            sanitized_sql = self.sanitize_sql(raw_sql)

            sanitized_sql = self._strip_sql_comments(sanitized_sql)

            if not self._validate_sql(sanitized_sql):
                raise ValueError("Сгенерированный SQL не прошел валидацию")

            return {
                "status": "success",
                "user_query": user_query,
                "sql_query": sanitized_sql,
                "metadata": {
                    "tables_accessed": self._get_accessed_tables(sanitized_sql),
                    "validation_passed": True,
                    "sanitization_passed": True,
                },
            }
        except Exception as e:
            logger.exception(f"Failed to generate SQL from {user_query=}")
            return {
                "status": "error",
                "error": str(e),
                "user_query": user_query,
                "metadata": {
                    "validation_passed": False,
                    "sanitization_passed": False,
                },
            }

    def _get_accessed_tables(self, sql: str) -> list[str]:
        try:
            parsed = parse_one(sql, dialect=Dialects.POSTGRES)
            return sorted({table.name.lower() for table in parsed.find_all(exp.Table)})
        except Exception:
            return []

    def execute_safe(self, sql: str) -> dict[str, Any]:
        """
        Safe execution of SQL query (SELECT-only) using SQLAlchemy.
        """
        try:
            sanitized_sql = sql
            if not self._validate_sql(sanitized_sql):
                raise ValueError("Запрос не прошел валидацию")

            with self.engine.connect() as conn:
                result: Result = conn.execute(text(sanitized_sql))
                rows = result.fetchall()

            if not rows:
                raise ValueError("Запрос вернулся пустым")

            columns = result.keys()
            results = [dict(zip(columns, row)) for row in rows]

            return {
                "status": "success",
                "results": results,
                "columns": list(columns),
                "row_count": len(results),
                "sql_executed": sanitized_sql,
            }

        except Exception as e:
            logger.exception(f"Failed to execute query {sql}.")
            return {
                "status": "error",
                "error": str(e),
                "sql_attempted": sql,
            }

    async def _verify_sql_against_query(
        self, user_query: str, sql_query: str
    ) -> dict[str, Any]:
        """
        Проверяет, соответствует ли сгенерированный SQL-запрос оригинальному запросу пользователя.
        """
        verification_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            user_query=user_query,
            sql_query=sql_query,
        )

        messages = [SystemMessage(content=verification_prompt)]

        try:
            result = await self.llm_client.create(messages)
            response_text = result.content.strip()

            if response_text.lower() == "ok":
                return {"status": "success", "is_correct": True}
            else:
                return {
                    "status": "success",
                    "is_correct": False,
                    "reason": response_text,
                }
        except Exception as e:
            logger.exception(
                f"Failed to verify generated query matches user query: {sql_query=} {user_query=}"
            )
            return {"status": "error", "error": str(e)}

    @print_result()
    async def query(
        self,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Полный цикл: генерация SQL + выполнение
        Args:
            user_query (str): Запрос на естественном языке
            check_sql_query (bool): Флаг, требуется ли проверять SQL-запрос на корректность
        Returns:
            Dict[str, Any]: Объединенные результаты генерации и выполнения
        """

        # Ambiguity checking
        logger.info("Проверяю запрос на неоднозначность...")
        ambiguity_check = await self._check_ambiguity(user_query)

        if ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]:
            logger.info(f"Запрос неоднозначен. Требуется уточнение по причине: {ambiguity_check['clarification_needed']}")
            return {"status": "ambiguous", "query": "ambiguous"}

        if ambiguity_check["status"] == "error":
            logger.info("Произошла ошибка при проверка неоднозначности.")
            return {"status": "error", "query": "error"}


        # Main query generation
        retries = 0
        success = False
        raw_sql = ""
        final_result = {"status": "error", "query": "error"}

        while not success and retries < MAX_RETRIES:
            logger.info(
                f"Попытка {retries + 1}/{MAX_RETRIES}. Генерирую валидный SQL-запрос... "
            )
            generation_result = await self.generate_sql(user_query)

            if generation_result["status"] != "success":
                retries += 1
                continue

            raw_sql = generation_result["sql_query"]
            # Извлекает SQL-запрос из markdown блока, если ответ полностью обернут в ```
            if raw_sql[:3] == "```":
                raw_sql = re.sub(
                    r"^```(?:sql)?\s*|\s*```$",
                    "",
                    raw_sql.strip(),
                    flags=re.IGNORECASE,
                )

            final_result = {"status": "success", "query": raw_sql}
            success = True

        return final_result
