from __future__ import annotations

import unittest
from unittest.mock import patch

from src.adv_text2sql.mcp_servers.text2sql_tool.main import _database_config


class DatabaseConfigTests(unittest.TestCase):
    def test_postgresql_uri_supplies_database_name(self) -> None:
        uri = "postgresql+psycopg://user:secret@localhost:5444/financial"

        with patch.dict("os.environ", {"MCP_DB_URI": uri}, clear=True):
            configured_uri, db_name = _database_config()

        self.assertEqual(configured_uri, uri)
        self.assertEqual(db_name, "financial")

    def test_uri_and_explicit_database_name_must_match(self) -> None:
        environment = {
            "MCP_DB_URI": (
                "postgresql+psycopg://user:secret@localhost:5444/financial"
            ),
            "MCP_DB_NAME": "toxicology",
        }

        with patch.dict("os.environ", environment, clear=True):
            with self.assertRaises(RuntimeError):
                _database_config()

    def test_non_postgresql_uri_is_rejected(self) -> None:
        with patch.dict(
            "os.environ",
            {"MCP_DB_URI": "sqlite:///unexpected.db"},
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                _database_config()


if __name__ == "__main__":
    unittest.main()
