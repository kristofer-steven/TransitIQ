"""Read-only DuckDB connection manager with security guardrails and query timeout (§16 PRD)."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from data.ingestion.config import settings

DEFAULT_MEMORY_LIMIT = "4GB"
DEFAULT_THREADS = 4
DEFAULT_TIMEOUT_SECONDS = 30.0


class QueryTimeoutError(TimeoutError):
    """Raised when an analytical DuckDB query exceeds the configured execution timeout."""


@dataclass(frozen=True)
class QueryExecutionResult:
    """Structured result returned from a read-only analytical query execution."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    execution_time_ms: float


def get_readonly_connection(
    db_path: Path | None = None,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection with external file/network access disabled."""
    target_path = db_path or settings.duckdb_path
    if not target_path.is_file():
        raise FileNotFoundError(
            f"DuckDB database file not found at {target_path}. Run 'make ingest' first."
        )

    conn = duckdb.connect(str(target_path), read_only=True)
    try:
        conn.execute(f"SET memory_limit = '{memory_limit}';")
        conn.execute(f"SET threads = {int(threads)};")
        conn.execute("SET enable_external_access = false;")
        conn.execute("SET lock_configuration = true;")
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def readonly_connection(
    db_path: Path | None = None,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager yielding a hardened read-only DuckDB connection."""
    conn = get_readonly_connection(
        db_path=db_path,
        memory_limit=memory_limit,
        threads=threads,
    )
    try:
        yield conn
    finally:
        conn.close()


def execute_readonly_query(
    sql: str,
    params: Sequence[Any] | None = None,
    db_path: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> QueryExecutionResult:
    """Execute a read-only SQL query with an enforced timeout interrupt."""
    timed_out = False

    with readonly_connection(db_path=db_path) as conn:
        def _interrupt_on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            conn.interrupt()

        timer = threading.Timer(timeout_seconds, _interrupt_on_timeout)
        timer.daemon = True
        start_ns = time.perf_counter_ns()
        timer.start()
        try:
            cursor = conn.execute(sql, list(params)) if params is not None else conn.execute(sql)
            rows = cursor.fetchall()
            description = cursor.description or []
            columns = [ str(col[0]) for col in description ]
        except duckdb.InterruptException as exc:
            raise QueryTimeoutError(
                f"Query exceeded the {timeout_seconds:.1f}s timeout limit."
            ) from exc
        except duckdb.Error as exc:
            if timed_out:
                raise QueryTimeoutError(
                    f"Query exceeded the {timeout_seconds:.1f}s timeout limit."
                ) from exc
            raise
        finally:
            timer.cancel()

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    return QueryExecutionResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        execution_time_ms=round(elapsed_ms, 2),
    )
