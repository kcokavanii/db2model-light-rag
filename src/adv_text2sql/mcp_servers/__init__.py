"""MCP servers exposed by the AdvText2SQL package.

Exports are lazy so running a concrete server with ``python -m`` does not
import its entrypoint once through this package and then execute it a second
time as ``__main__``.
"""

from typing import Any


__all__ = ["servers", "text2sql_server"]
servers: list[Any]
text2sql_server: Any


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .text2sql_tool.main import server

    if name == "text2sql_server":
        return server
    return [server]
