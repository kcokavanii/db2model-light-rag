"""Run one explicit, potentially billable end-to-end MCP tool call."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from fastmcp import Client

from .main import server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the MCP generate_sql tool")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "baseline", "compact", "structural", "semantic_v3"),
        default="auto",
    )
    parser.add_argument("--evidence")
    return parser.parse_args()


async def call_generate_sql(args: argparse.Namespace) -> dict[str, Any] | None:
    async with Client(server) as client:
        result = await client.call_tool(
            "generate_sql",
            {
                "question": args.question,
                "mode": args.mode,
                "evidence": args.evidence,
            },
        )
    payload = result.data
    if payload is not None and not isinstance(payload, dict):
        raise RuntimeError("generate_sql returned a non-dictionary payload")
    return payload


def main() -> None:
    payload = asyncio.run(call_generate_sql(parse_args()))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
