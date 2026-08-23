"""FastMCP entrypoint for the adaptive database-specific Text-to-SQL tool."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy.engine import URL, make_url

from .src.context_modes import ContextMode
from .src.service import AdaptiveText2SQLService, UsageTrackingClient


PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / ".env", override=False)

log = logging.getLogger(__name__)
_service: AdaptiveText2SQLService | None = None
_service_lock = asyncio.Lock()


@asynccontextmanager
async def _server_lifespan(_: Any) -> AsyncIterator[None]:
    global _service
    try:
        yield
    finally:
        service = _service
        _service = None
        if service is not None:
            await service.close()


server = FastMCP(
    "adv_text2sql",
    instructions=(
        "Generate one validated PostgreSQL read-only query for the configured "
        "database. SQL is returned to the caller and is never executed."
    ),
    lifespan=_server_lifespan,
    mask_error_details=True,
)


def _required_env(primary: str, fallback: str | None = None) -> str:
    value = os.getenv(primary)
    if not value and fallback is not None:
        value = os.getenv(fallback)
    if value:
        return value
    names = f"{primary} or {fallback}" if fallback else primary
    raise RuntimeError(f"Missing required environment variable: {names}")


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


def _database_config() -> tuple[str, str]:
    explicit_uri = os.getenv("MCP_DB_URI")
    configured_name = os.getenv("MCP_DB_NAME")

    if explicit_uri:
        parsed_url = make_url(explicit_uri)
        if parsed_url.get_backend_name() != "postgresql":
            raise RuntimeError("MCP_DB_URI must use a PostgreSQL driver")
        uri_db_name = parsed_url.database
        if not uri_db_name:
            raise RuntimeError("MCP_DB_URI must contain a database name")
        if configured_name and configured_name != uri_db_name:
            raise RuntimeError(
                "MCP_DB_NAME must match the database name in MCP_DB_URI"
            )
        db_name = configured_name or uri_db_name
        return explicit_uri, db_name

    db_name = _required_env("MCP_DB_NAME")
    url = URL.create(
        drivername="postgresql+psycopg",
        username=_required_env("DB_USER"),
        password=_required_env("DB_PASS"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5444")),
        database=db_name,
    )
    return url.render_as_string(hide_password=False), db_name


def _path_env(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    configured_path = Path(raw_value).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


def _build_service() -> AdaptiveText2SQLService:
    db_uri, db_name = _database_config()
    target_model = _required_env("TARGET_LLM_MODEL_NAME", "LLM_MODEL_NAME")
    target_base_url = _required_env("TARGET_LLM_BASE_URL", "LLM_BASE_URL")
    target_api_key = _required_env("TARGET_LLM_API_KEY", "LLM_API_KEY")

    retrieval_model = os.getenv("LIGHTRAG_LLM_MODEL_NAME") or target_model
    retrieval_base_url = os.getenv("LIGHTRAG_LLM_BASE_URL") or target_base_url
    retrieval_api_key = os.getenv("LIGHTRAG_LLM_API_KEY") or target_api_key

    raw_target_client = OpenAIChatCompletionClient(
        model=target_model,
        base_url=target_base_url,
        api_key=target_api_key,
        temperature=0,
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
    )
    target_client = UsageTrackingClient(raw_target_client)

    return AdaptiveText2SQLService(
        db_uri=db_uri,
        db_name=db_name,
        target_client=target_client,
        retrieval_model_name=retrieval_model,
        retrieval_base_url=retrieval_base_url,
        retrieval_api_key=retrieval_api_key,
        lightrag_storage_dir=_path_env(
            "MCP_LIGHTRAG_STORAGE_DIR",
            PROJECT_ROOT / "artifacts" / "lightrag" / db_name,
        ),
        db_knowledge_path=_path_env(
            "MCP_DB_KNOWLEDGE_PATH",
            PROJECT_ROOT
            / "artifacts"
            / "db_knowledge"
            / f"{db_name}_knowledge.json",
        ),
        schema_token_threshold=int(os.getenv("MCP_SCHEMA_TOKEN_THRESHOLD", "150")),
        validate_with_explain=_bool_env("MCP_VALIDATE_EXPLAIN", True),
        explain_timeout_ms=int(os.getenv("MCP_EXPLAIN_TIMEOUT_MS", "3000")),
    )


async def _get_service() -> AdaptiveText2SQLService:
    global _service
    if _service is not None:
        return _service
    async with _service_lock:
        if _service is None:
            _service = _build_service()
    return _service


@server.tool(
    name="generate_sql",
    description=(
        "Generate one PostgreSQL read-only query for the configured database. "
        "Auto mode routes only between baseline, compact, and structural; it "
        "never uses semantic_v3. Explicit semantic_v3 reproduces the expensive "
        "semantic LightRAG context for manual comparison. The SQL is validated "
        "but never executed."
    ),
)
async def generate_sql(
    question: Annotated[
        str,
        Field(description="Natural-language question about the configured database."),
    ],
    mode: Annotated[
        Literal["auto", "baseline", "compact", "structural", "semantic_v3"],
        Field(
            description=(
                "Context policy. Auto excludes semantic_v3 and escalates at most once."
            )
        ),
    ] = "auto",
    evidence: Annotated[
        str | None,
        Field(
            description=(
                "Optional schema hint, abbreviation explanation, or BIRD evidence."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    service = await _get_service()
    return await service.generate_sql(
        question=question,
        mode=ContextMode(mode),
        evidence=evidence,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AdvText2SQL MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default=os.getenv("MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8000")),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(
            transport="http",
            host=args.host,
            port=args.port,
            show_banner=False,
        )


if __name__ == "__main__":
    main()
