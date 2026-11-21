"""Pydantic loader and DuckDB conformance verifier for data/schemas/catalog.yaml."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import duckdb
import yaml
from pydantic import BaseModel, Field

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"


class TemporalCoverage(BaseModel):
    """Dataset start and end date coverage."""

    start_date: str
    end_date: str


class ColumnCatalogEntry(BaseModel):
    """Metadata and documentation for a single table column."""

    name: str
    type: str
    nullable: bool = True
    description: str


class ForeignKeyEntry(BaseModel):
    """Explicit foreign-key relationship and join condition between two tables."""

    column: str
    target_table: str
    target_column: str
    join_condition: str
    description: str


class TableCatalogEntry(BaseModel):
    """Metadata, primary key, foreign keys, and columns for an analytical table."""

    name: str
    description: str
    grain: str
    allow_query: bool = True
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyEntry] = Field(default_factory=list)
    columns: list[ColumnCatalogEntry]

    def column_names(self) -> set[str]:
        """Return the set of column names defined on this table."""
        return {col.name for col in self.columns}


class SchemaCatalog(BaseModel):
    """Top-level schema catalog model representing catalog.yaml."""

    version: str
    database: str
    dialect: str = "duckdb"
    as_of_date: str
    temporal_coverage: TemporalCoverage
    tables: list[TableCatalogEntry]

    def table_map(self) -> dict[str, TableCatalogEntry]:
        """Return a mapping of table name -> TableCatalogEntry."""
        return {tbl.name: tbl for tbl in self.tables}

    def get_table(self, table_name: str) -> TableCatalogEntry:
        """Retrieve a table entry by name or raise KeyError."""
        mapping = self.table_map()
        if table_name not in mapping:
            raise KeyError(f"Unknown table '{table_name}' in schema catalog.")
        return mapping[table_name]

    def allowlisted_table_names(self) -> set[str]:
        """Return all table names where allow_query is True."""
        return {tbl.name for tbl in self.tables if tbl.allow_query}

    def allowlisted_columns_by_table(self) -> dict[str, set[str]]:
        """Return {table_name: {col1, col2, ...}} for all allowlisted tables."""
        return {
            tbl.name: tbl.column_names()
            for tbl in self.tables
            if tbl.allow_query
        }

    def find_connecting_tables_and_joins(
        self,
        seed_tables: set[str],
    ) -> tuple[set[str], list[ForeignKeyEntry]]:
        """Complete the foreign-key subgraph connecting all `seed_tables`.

        Returns the expanded set of table names and the relevant ForeignKeyEntry join edges.
        """
        valid_tables = self.allowlisted_table_names()
        active_seeds = [t for t in seed_tables if t in valid_tables]
        if not active_seeds:
            return set(), []

        # Build undirected adjacency list over FK edges
        adj: dict[str, list[tuple[str, ForeignKeyEntry]]] = {t: [] for t in valid_tables}
        for tbl in self.tables:
            if not tbl.allow_query:
                continue
            for fk in tbl.foreign_keys:
                if fk.target_table in valid_tables:
                    adj[tbl.name].append((fk.target_table, fk))
                    adj[fk.target_table].append((tbl.name, fk))

        expanded_tables: set[str] = set(active_seeds)
        root = active_seeds[0]

        for target in active_seeds[1:]:
            if target in expanded_tables and target == root:
                continue
            # BFS shortest path from root to target
            queue: deque[tuple[str, list[str]]] = deque([(root, [root])])
            visited: set[str] = {root}
            while queue:
                curr, path = queue.popleft()
                if curr == target:
                    expanded_tables.update(path)
                    break
                for neighbor, _ in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path, neighbor]))

        selected_joins: list[ForeignKeyEntry] = []
        for tbl in self.tables:
            if tbl.name not in expanded_tables:
                continue
            for fk in tbl.foreign_keys:
                if fk.target_table in expanded_tables:
                    selected_joins.append(fk)

        return expanded_tables, selected_joins


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> SchemaCatalog:
    """Load and validate catalog.yaml into a SchemaCatalog instance."""
    if not path.is_file():
        raise FileNotFoundError(f"Catalog YAML not found at {path}")
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SchemaCatalog.model_validate(raw_data)


def verify_catalog_against_duckdb(
    conn: duckdb.DuckDBPyConnection,
    catalog: SchemaCatalog | None = None,
) -> list[str]:
    """Compare catalog.yaml tables and columns against an active DuckDB connection.

    Returns a list of discrepancy strings (empty list means 100% conformance).
    """
    cat = catalog or load_catalog()
    discrepancies: list[str] = []

    db_rows = conn.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position;
        """
    ).fetchall()

    db_tables: dict[str, dict[str, str]] = {}
    for tbl_name, col_name, dtype in db_rows:
        db_tables.setdefault(str(tbl_name), {})[str(col_name)] = str(dtype).upper()

    catalog_tables = cat.table_map()

    # Check every table in catalog exists in DuckDB with matching columns
    for tbl_name, tbl_entry in catalog_tables.items():
        if tbl_name not in db_tables:
            discrepancies.append(f"Table '{tbl_name}' in catalog.yaml is missing from DuckDB.")
            continue

        actual_cols = db_tables[tbl_name]
        expected_cols = {c.name: c.type.upper() for c in tbl_entry.columns}

        for col_name, exp_type in expected_cols.items():
            if col_name not in actual_cols:
                discrepancies.append(
                    f"Column '{tbl_name}.{col_name}' in catalog.yaml is missing from DuckDB."
                )
            else:
                act_type = actual_cols[col_name]
                if exp_type != act_type:
                    discrepancies.append(
                        f"Type mismatch on '{tbl_name}.{col_name}': "
                        f"catalog='{exp_type}' vs duckdb='{act_type}'."
                    )

        for actual_col in actual_cols:
            if actual_col not in expected_cols:
                discrepancies.append(
                    f"Column '{tbl_name}.{actual_col}' exists in DuckDB "
                    "but is undocumented in catalog.yaml."
                )

    # Check for undocumented tables in DuckDB
    for db_tbl in db_tables:
        if db_tbl not in catalog_tables:
            discrepancies.append(
                f"Table '{db_tbl}' exists in DuckDB but is undocumented in catalog.yaml."
            )

    return discrepancies
