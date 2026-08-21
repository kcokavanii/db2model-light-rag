import asyncio
import os
import logging

# import pprint
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StreamableHttpServerParams, mcp_server_tools
from dotenv import load_dotenv
from typing import Any

load_dotenv(override=True)

logger = logging.getLogger(__name__)

LLM_URL = os.environ["LLM_BASE_URL"]
LLM_MODEL_NAME = os.environ["LLM_MODEL_NAME"]
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# INTERACTIVE = os.getenv("INTERACTIVE")  # run agent in chat-mode in CLI if set. Otherwise, run once with DEFAULT_TASK

SYSTEM_MESSAGE = "Ты ассистент по делам коммерческого предприятия. Твоя роль -- отвечать на вопросы менеджеров."
"""Assistant Agent's system message."""
DEFAULT_TASK = "Какой средний процент отказов за январь по всем до?"
"""Default task run if client is not in interactive mode."""


async def main():
    tools: list[Any] = await mcp_server_tools(
        server_params=StreamableHttpServerParams(url="http://127.0.0.1:8000/mcp/")
    )

    # pprint.pp([tool._tool.model_dump() for tool in tools], indent=4, width=120)

    model_client = OpenAIChatCompletionClient(
        model=LLM_MODEL_NAME,
        base_url=LLM_URL,
        api_key="-",
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    agent = AssistantAgent(
        name="agent",
        model_client=model_client,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    # run agent
    if True:  # :)
        while True:
            request = input(">> ")
            await Console(
                agent.run_stream(task=request, cancellation_token=CancellationToken())
            )
    else:
        await Console(agent.run_stream(task=DEFAULT_TASK))


if __name__ == "__main__":
    asyncio.run(main())
