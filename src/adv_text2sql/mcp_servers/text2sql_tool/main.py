import asyncio
import os
from typing import Annotated
import logging

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from .src.generate_tool_description import generate_description_text2sql
from .src.text2sql_implementation import Text2SQLGenerator
from tabulate import tabulate
# from unilog import setup_logging

load_dotenv(override=True)

logger = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_URL")

server = FastMCP("text2sql_tool_server")

llm_client = OpenAIChatCompletionClient(
    model=os.environ["LLM_MODEL_NAME"],
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    temperature=0.6,
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
        "structured_output": False,
    },
)

processed_path = "processed_documents"
db_path = "text2sql_generated.db"

logger.info("Инициализация инструмента ...")
text2sql_agent = Text2SQLGenerator(
    db_path=db_path,
    llm_client=llm_client,
)

logger.info("Генерация описания ...")

generated_description = None
if not generated_description:
    generated_description = asyncio.run(
        generate_description_text2sql(
            tool_description=(
                "Генератор SQL-запросов на естественном языке с контролем безопасности."
            ),
            text2sql_agent=text2sql_agent,
        )
    )

logger.info("Генерация описания завершена")
logger.info("Инициализация инструмента завершена")

logger.debug(f"Системная инструкция:\n\n{text2sql_agent.system_prompt}")


@server.tool(description=generated_description)
async def text2sql(
    user_query_text: Annotated[
        str,
        Field(
            description=(
                "Текстовый запрос (напр.'все отчёты по западному округу за первый квартал')"
            )
        ),
    ],
) -> str | None:
    global text2sql_agent

    result = await text2sql_agent.query(
        user_query_text,
        check_ambiguity=False,
        check_sql_query=False,
    )

    root = result.get("params", result) if isinstance(result, dict) else {}
    details = root.get("data", {}).get("details", {})
    exec_info = details.get("execution", {})

    if exec_info.get("status") == "success":
        data = exec_info.get("results", [])
        markdown_table = (
            tabulate(data, headers="keys", tablefmt="pipe") if data else "Нет данных"
        )
        return markdown_table

    error_msg = (
        details.get("error") or root.get("message") or "Выполнение запроса неуспешно"
    )
    return f"_Ошибка_: {error_msg}"


logger.info("Tool description: %s", text2sql.description)

if __name__ == "__main__":
    server.run(transport="http", host="0.0.0.0", port=8000, show_banner=False)
