from __future__ import annotations

import unittest

from src.adv_text2sql.mcp_servers.text2sql_tool.src.context_modes import (
    validate_read_only_sql,
)


class ReadOnlySQLValidationTests(unittest.TestCase):
    def test_accepts_select_cte_and_set_operations(self) -> None:
        valid_queries = [
            "SELECT account_id, balance FROM accounts;",
            (
                "WITH positive AS ("
                "SELECT account_id FROM accounts WHERE balance > 0"
                ") SELECT account_id FROM positive"
            ),
            "SELECT district_id FROM accounts UNION SELECT district_id FROM districts",
            "SELECT 'DROP TABLE accounts' AS harmless_text",
            "-- read-only report\nSELECT COUNT(*) FROM accounts",
        ]

        for sql in valid_queries:
            with self.subTest(sql=sql):
                self.assertIsNone(validate_read_only_sql(sql))

    def test_rejects_empty_invalid_multiple_and_non_query_statements(self) -> None:
        invalid_queries = [
            "",
            "SELEC FROM",
            "SELECT 1; SELECT 2",
            "INSERT INTO accounts(account_id) VALUES (1)",
            "UPDATE accounts SET balance = 0",
            "DELETE FROM accounts",
            "CREATE TABLE unsafe(id integer)",
            "ALTER TABLE accounts ADD COLUMN unsafe text",
            "DROP TABLE accounts",
            "TRUNCATE TABLE accounts",
            "COPY accounts TO '/tmp/accounts.csv'",
            "CALL refresh_accounts()",
            "EXPLAIN SELECT * FROM accounts",
            "SELECT * FROM accounts FOR UPDATE",
            "SELECT * FROM accounts FOR SHARE",
        ]

        for sql in invalid_queries:
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    validate_read_only_sql(sql)

    def test_rejects_write_hidden_inside_cte(self) -> None:
        sql = (
            "WITH removed AS ("
            "DELETE FROM accounts WHERE balance < 0 RETURNING account_id"
            ") SELECT account_id FROM removed"
        )

        with self.assertRaises(ValueError):
            validate_read_only_sql(sql)

    def test_rejects_select_into_because_it_creates_a_table(self) -> None:
        with self.assertRaises(ValueError):
            validate_read_only_sql(
                "SELECT account_id INTO archived_accounts FROM accounts"
            )


if __name__ == "__main__":
    unittest.main()
