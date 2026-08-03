"""The API model and the database schema must describe the same columns.

This is the regression test for the bug that broke cloud history sync in the
original code: the frontend sent `variant` and `locality`, the documented schema
had neither, PostgREST rejected the insert, and the failure was swallowed by a
console.warn. Nothing in the codebase related the two definitions, so the drift
was invisible.

Now backend/routers/history.py and supabase/migrations/*.sql are checked against
each other on every push, and adding a field to one without the other fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.routers.history import EvaluationIn, EvaluationOut, ProfileIn

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"

# Columns the server assigns and the client may not supply.
SERVER_OWNED_COLUMNS = {"id", "user_id", "created_at", "updated_at"}


def _migration_sql() -> str:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        pytest.fail(f"no migrations found in {MIGRATIONS_DIR}")
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def _table_columns(sql: str, table: str) -> set[str]:
    """Extract column names from a `create table ... ( ... )` block.

    A deliberately simple parser: it reads the column list up to the closing
    paren and takes the first identifier of each line. That is sufficient for
    hand-written DDL and avoids adding a SQL-parsing dependency for one test.
    """
    match = re.search(
        rf"create\s+table\s+(?:if\s+not\s+exists\s+)?(?:public\.)?{table}\s*\((.*?)\n\);",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        pytest.fail(f"could not locate the `{table}` table definition in the migrations")

    columns: set[str] = set()
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        # Skip table-level constraints, which start with a keyword not a column name.
        if re.match(r"^(primary|foreign|unique|check|constraint)\b", stripped, re.IGNORECASE):
            continue
        identifier = re.match(r"^([a-z_][a-z0-9_]*)", stripped, re.IGNORECASE)
        if identifier:
            columns.add(identifier.group(1).lower())
    return columns


@pytest.fixture(scope="module")
def sql() -> str:
    return _migration_sql()


class TestEvaluationsTable:
    def test_every_api_field_has_a_column(self, sql):
        columns = _table_columns(sql, "evaluations")
        api_fields = set(EvaluationIn.model_fields)
        missing = api_fields - columns
        assert not missing, (
            f"EvaluationIn accepts fields with no column in the evaluations table: "
            f"{sorted(missing)}. Add them to a new migration."
        )

    def test_every_column_is_either_api_or_server_owned(self, sql):
        columns = _table_columns(sql, "evaluations")
        api_fields = set(EvaluationOut.model_fields)
        unexplained = columns - api_fields - SERVER_OWNED_COLUMNS
        assert not unexplained, (
            f"evaluations has columns no API field writes or returns: "
            f"{sorted(unexplained)}. Either expose them or drop them."
        )

    def test_the_previously_missing_columns_are_present(self, sql):
        """Explicit guard on the two columns whose absence caused the original bug."""
        columns = _table_columns(sql, "evaluations")
        assert "variant" in columns
        assert "locality" in columns

    def test_client_cannot_supply_server_owned_fields(self):
        """id / created_at / user_id must not be settable from a request body."""
        for field in ("id", "created_at", "user_id"):
            assert field not in EvaluationIn.model_fields

    def test_user_id_is_a_uuid_column_not_text(self, sql):
        """The old code wrote the string 'guest' here; guests no longer touch the DB."""
        match = re.search(r"^\s*user_id\s+(\w+)", sql, re.MULTILINE | re.IGNORECASE)
        assert match, "user_id column not found"
        assert match.group(1).lower() == "uuid"

    def test_money_columns_are_numeric_not_float(self, sql):
        """Binary floating point cannot represent decimal currency exactly."""
        for column in ("market_value", "buy_price", "sell_price", "expected_profit"):
            match = re.search(rf"^\s*{column}\s+(\w+)", sql, re.MULTILINE | re.IGNORECASE)
            assert match, f"{column} column not found"
            assert match.group(1).lower() == "numeric", (
                f"{column} should be numeric, found {match.group(1)}"
            )


class TestProfilesTable:
    def test_every_api_field_has_a_column(self, sql):
        columns = _table_columns(sql, "profiles")
        missing = set(ProfileIn.model_fields) - columns
        assert not missing, f"ProfileIn fields with no column: {sorted(missing)}"


class TestRowLevelSecurity:
    def test_rls_is_enabled_on_both_tables(self, sql):
        for table in ("profiles", "evaluations"):
            assert re.search(
                rf"alter\s+table\s+public\.{table}\s+enable\s+row\s+level\s+security",
                sql, re.IGNORECASE,
            ), f"RLS not enabled on {table}"

    def test_rls_is_forced_so_it_also_applies_to_the_table_owner(self, sql):
        for table in ("profiles", "evaluations"):
            assert re.search(
                rf"alter\s+table\s+public\.{table}\s+force\s+row\s+level\s+security",
                sql, re.IGNORECASE,
            ), f"RLS not forced on {table}"

    def test_every_policy_is_scoped_to_the_calling_user(self, sql):
        """A policy without an auth.uid() comparison would expose every row."""
        policies = re.findall(
            r"create\s+policy\s+\"([^\"]+)\"(.*?)(?=create\s+policy|\Z)",
            sql, re.IGNORECASE | re.DOTALL,
        )
        assert policies, "no RLS policies found"
        for name, body in policies:
            assert "auth.uid()" in body, f"policy {name!r} is not scoped to auth.uid()"

    def test_anon_role_has_no_access(self, sql):
        """Guests keep history in localStorage; the anon role needs nothing."""
        for table in ("profiles", "evaluations"):
            assert re.search(
                rf"revoke\s+all\s+on\s+public\.{table}\s+from\s+anon", sql, re.IGNORECASE
            ), f"anon access not revoked on {table}"

    def test_evaluations_has_no_update_policy(self, sql):
        """A valuation is a point-in-time record; editing one breaks the audit trail."""
        assert not re.search(
            r"create\s+policy[^;]*on\s+public\.evaluations\s+for\s+update", sql, re.IGNORECASE
        )
