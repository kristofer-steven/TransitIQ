"""Build the DuckDB star schema and Data Quality (DQ) reconciliation report."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from data.ingestion.config import IngestionSettings, settings
from data.quality.rules import (
    FACT_TRIPS_COLUMNS,
    QUARANTINE_COLUMNS,
    REJECTION_REASONS,
    DQFileSummary,
    ServiceType,
    process_trip_parquet_file,
)

logger = logging.getLogger(__name__)

TRIP_FILENAME_RE = re.compile(r"^(yellow|green)_tripdata_(\d{4}-\d{2})\.parquet$")

FACT_TRIPS_DDL = """
CREATE TABLE fact_trips (
    trip_id BIGINT PRIMARY KEY,
    service_type VARCHAR NOT NULL,
    vendor_id INTEGER NOT NULL,
    pickup_datetime TIMESTAMP NOT NULL,
    dropoff_datetime TIMESTAMP NOT NULL,
    pickup_zone_id INTEGER NOT NULL,
    dropoff_zone_id INTEGER NOT NULL,
    passenger_count INTEGER,
    trip_distance_mi DOUBLE NOT NULL,
    trip_duration_min DOUBLE NOT NULL,
    rate_code_id INTEGER NOT NULL,
    payment_type_id INTEGER NOT NULL,
    fare_amount DOUBLE NOT NULL,
    tip_amount DOUBLE NOT NULL,
    tolls_amount DOUBLE NOT NULL,
    congestion_surcharge DOUBLE NOT NULL,
    airport_fee DOUBLE NOT NULL,
    total_amount DOUBLE NOT NULL
);
"""

TRIPS_QUARANTINE_DDL = """
CREATE TABLE trips_quarantine (
    trip_id BIGINT PRIMARY KEY,
    service_type VARCHAR NOT NULL,
    vendor_id INTEGER,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    pickup_zone_id INTEGER,
    dropoff_zone_id INTEGER,
    passenger_count INTEGER,
    trip_distance_mi DOUBLE,
    trip_duration_min DOUBLE,
    rate_code_id INTEGER,
    payment_type_id INTEGER,
    fare_amount DOUBLE,
    tip_amount DOUBLE,
    tolls_amount DOUBLE,
    congestion_surcharge DOUBLE,
    airport_fee DOUBLE,
    total_amount DOUBLE,
    source_file VARCHAR NOT NULL,
    year_month VARCHAR NOT NULL,
    rejection_reason VARCHAR NOT NULL
);
"""

AGG_ZONE_HOURLY_DEMAND_DDL = """
CREATE TABLE agg_zone_hourly_demand AS
WITH pickup_agg AS (
    SELECT
        pickup_zone_id AS zone_id,
        date_trunc('hour', pickup_datetime) AS hour_ts,
        COUNT(*) AS pickups,
        AVG(fare_amount) AS avg_fare,
        AVG(trip_distance_mi) AS avg_distance_mi
    FROM fact_trips
    GROUP BY pickup_zone_id, date_trunc('hour', pickup_datetime)
),
dropoff_agg AS (
    SELECT
        dropoff_zone_id AS zone_id,
        date_trunc('hour', dropoff_datetime) AS hour_ts,
        COUNT(*) AS dropoffs
    FROM fact_trips
    GROUP BY dropoff_zone_id, date_trunc('hour', dropoff_datetime)
)
SELECT
    COALESCE(p.zone_id, d.zone_id) AS zone_id,
    COALESCE(p.hour_ts, d.hour_ts) AS hour_ts,
    COALESCE(p.pickups, 0)::BIGINT AS pickups,
    COALESCE(d.dropoffs, 0)::BIGINT AS dropoffs,
    p.avg_fare AS avg_fare,
    p.avg_distance_mi AS avg_distance_mi
FROM pickup_agg p
FULL OUTER JOIN dropoff_agg d
    ON p.zone_id = d.zone_id AND p.hour_ts = d.hour_ts
ORDER BY hour_ts, zone_id;
"""


def build_dim_zone_dataframe(zone_csv_path: Path) -> pl.DataFrame:
    """Load and clean taxi_zone_lookup.csv into the dim_zone table schema."""
    if not zone_csv_path.is_file():
        raise FileNotFoundError(f"Missing taxi zone lookup CSV at {zone_csv_path}")

    raw_zones = pl.read_csv(zone_csv_path, null_values=["N/A", "NA", ""])
    df = raw_zones.select(
        pl.col("LocationID").cast(pl.Int32).alias("zone_id"),
        pl.col("Zone").fill_null("Unknown").cast(pl.String).alias("zone"),
        pl.col("Borough").fill_null("Unknown").cast(pl.String).alias("borough"),
        pl.col("service_zone").fill_null("Unknown").cast(pl.String).alias("service_zone"),
    ).with_columns(
        pl.col("zone_id").is_between(1, 263).alias("is_valid_nyc_zone")
    )
    return df.sort("zone_id")


def _load_dimensions_into_duckdb(conn: duckdb.DuckDBPyConnection, cfg: IngestionSettings) -> None:
    """Load dim_zone, dim_date, lookup tables, and weather_daily into DuckDB."""
    zone_csv = cfg.reference_raw_dir / "taxi_zone_lookup.csv"
    dim_zone_df = build_dim_zone_dataframe(zone_csv)
    conn.register("_dim_zone_df", dim_zone_df)
    conn.execute("CREATE TABLE dim_zone AS SELECT * FROM _dim_zone_df;")
    conn.unregister("_dim_zone_df")

    ref_tables = ("dim_date", "dim_payment_type", "dim_rate_code", "dim_vendor")
    for table_name in ref_tables:
        parquet_path = cfg.reference_raw_dir / f"{table_name}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing reference file: {parquet_path}")
        sql_path = parquet_path.as_posix()
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{sql_path}');")

    weather_stem = f"central_park_weather_{cfg.tlc_start_year}_{cfg.tlc_end_year}.parquet"
    weather_path = cfg.weather_raw_dir / weather_stem
    if not weather_path.is_file():
        raise FileNotFoundError(f"Missing weather Parquet file: {weather_path}")
    weather_sql_path = weather_path.as_posix()
    conn.execute(
        f"CREATE TABLE weather_daily AS SELECT * FROM read_parquet('{weather_sql_path}');"
    )


def discover_trip_files(cfg: IngestionSettings) -> list[tuple[Path, ServiceType, str]]:
    """Find all raw monthly Yellow and Green trip Parquet files sorted chronologically."""
    discovered: list[tuple[Path, ServiceType, str]] = []
    for folder in (cfg.yellow_raw_dir, cfg.green_raw_dir):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.parquet")):
            match = TRIP_FILENAME_RE.match(path.name)
            if not match:
                logger.warning("Skipping unrecognized Parquet filename: %s", path.name)
                continue
            service: ServiceType = "yellow" if match.group(1) == "yellow" else "green"
            year_month = match.group(2)
            discovered.append((path, service, year_month))
    return sorted(discovered, key=lambda item: (item[2], item[1], item[0].name))


def render_dq_markdown(report: dict[str, Any]) -> str:
    """Render the Data Quality reconciliation report as a Markdown document."""
    totals = report["totals"]
    raw_total = totals["raw_rows"]
    clean_pct = (totals["clean_rows"] / raw_total * 100.0) if raw_total > 0 else 0.0
    quar_pct = (totals["quarantine_rows"] / raw_total * 100.0) if raw_total > 0 else 0.0

    lines: list[str] = [
        "# Data Quality & Reconciliation Report",
        "",
        f"**Generated At (UTC):** `{report['generated_at_utc']}`  ",
        f"**DuckDB Path:** `{report['duckdb_path']}`  ",
        f"**Reconciled (`raw == clean + quarantine`):** `{totals['is_reconciled']}`",
        "",
        "## 1. Overall Totals",
        "",
        "| Metric | Count | Share |",
        "| :--- | ---: | ---: |",
        f"| Files Processed | {totals['files_processed']} | - |",
        f"| Raw Trip Rows | {totals['raw_rows']:,} | 100.00% |",
        f"| Clean Trip Rows (`fact_trips`) | {totals['clean_rows']:,} | {clean_pct:.2f}% |",
        (
            f"| Quarantined Rows (`trips_quarantine`) | "
            f"{totals['quarantine_rows']:,} | {quar_pct:.2f}% |"
        ),
        "",
        "## 2. Quarantine Breakdown by Rule",
        "",
        "| Rejection Reason | Quarantined Rows | % of Raw Rows |",
        "| :--- | ---: | ---: |",
    ]

    for reason, count in totals["rejection_counts"].items():
        pct = (count / raw_total * 100.0) if raw_total > 0 else 0.0
        lines.append(f"| `{reason}` | {count:,} | {pct:.4f}% |")

    lines.extend(
        [
            "",
            "## 3. Monthly File Reconciliation",
            "",
            "| Source File | Service | Month | Raw Rows | Clean Rows | Quarantined | Reconciled |",
            "| :--- | :---: | :---: | ---: | ---: | ---: | :---: |",
        ]
    )

    for f in report["files"]:
        lines.append(
            f"| `{f['source_file']}` | {f['service_type']} | {f['year_month']} | "
            f"{f['raw_rows']:,} | {f['clean_rows']:,} | {f['quarantine_rows']:,} | "
            f"`{f['is_reconciled']}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Analytical Star Schema Table Row Counts",
            "",
            "| Table Name | Row Count |",
            "| :--- | ---: |",
        ]
    )

    for table_name, count in report["table_row_counts"].items():
        lines.append(f"| `{table_name}` | {count:,} |")

    lines.append("")
    return "\n".join(lines)


def build_star_schema(
    cfg: IngestionSettings = settings,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Build the complete DuckDB star schema from raw files and produce DQ reports."""
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.duckdb_path

    if db_path.exists():
        if not overwrite:
            raise FileExistsError(f"DuckDB database already exists at {db_path}")
        db_path.unlink()
        wal_path = db_path.with_suffix(db_path.suffix + ".wal")
        if wal_path.exists():
            wal_path.unlink()

    trip_files = discover_trip_files(cfg)
    if not trip_files:
        raise FileNotFoundError(
            f"No raw trip Parquet files found in {cfg.yellow_raw_dir} or {cfg.green_raw_dir}."
        )

    file_summaries: list[DQFileSummary] = []
    trip_id_offset = 0

    conn = duckdb.connect(str(db_path))
    try:
        logger.info("Loading reference and weather dimension tables into %s...", db_path)
        _load_dimensions_into_duckdb(conn, cfg)

        conn.execute(FACT_TRIPS_DDL)
        conn.execute(TRIPS_QUARANTINE_DDL)

        fact_cols_sql = ", ".join(FACT_TRIPS_COLUMNS)
        quar_cols_sql = ", ".join(QUARANTINE_COLUMNS)

        for parquet_path, service_type, year_month in trip_files:
            logger.info("Processing %s (%s, %s)...", parquet_path.name, service_type, year_month)
            batch = process_trip_parquet_file(
                parquet_path=parquet_path,
                service_type=service_type,
                year_month=year_month,
                trip_id_offset=trip_id_offset,
            )
            trip_id_offset += batch.summary.raw_rows
            file_summaries.append(batch.summary)

            if batch.clean_df.height > 0:
                conn.register("_clean_batch", batch.clean_df)
                conn.execute(
                    f"INSERT INTO fact_trips ({fact_cols_sql}) "
                    f"SELECT {fact_cols_sql} FROM _clean_batch;"
                )
                conn.unregister("_clean_batch")

            if batch.quarantine_df.height > 0:
                conn.register("_quar_batch", batch.quarantine_df)
                conn.execute(
                    f"INSERT INTO trips_quarantine ({quar_cols_sql}) "
                    f"SELECT {quar_cols_sql} FROM _quar_batch;"
                )
                conn.unregister("_quar_batch")

            logger.info(
                "[OK  ] %s: raw=%s, clean=%s, quarantine=%s",
                parquet_path.name,
                f"{batch.summary.raw_rows:,}",
                f"{batch.summary.clean_rows:,}",
                f"{batch.summary.quarantine_rows:,}",
            )

        logger.info("Building pre-aggregated table agg_zone_hourly_demand...")
        conn.execute(AGG_ZONE_HOURLY_DEMAND_DDL)

        all_tables = (
            "fact_trips",
            "trips_quarantine",
            "agg_zone_hourly_demand",
            "dim_zone",
            "dim_date",
            "dim_payment_type",
            "dim_rate_code",
            "dim_vendor",
            "weather_daily",
        )
        table_row_counts: dict[str, int] = {}
        for tbl in all_tables:
            row = conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()
            table_row_counts[tbl] = int(row[0]) if row else 0

    finally:
        conn.close()

    total_raw = sum(s.raw_rows for s in file_summaries)
    total_clean = sum(s.clean_rows for s in file_summaries)
    total_quarantine = sum(s.quarantine_rows for s in file_summaries)

    total_rejections: dict[str, int] = {reason: 0 for reason in REJECTION_REASONS}
    for s in file_summaries:
        for reason, cnt in s.rejection_counts.items():
            total_rejections[reason] = total_rejections.get(reason, 0) + cnt

    db_reconciled = (
        total_raw == total_clean + total_quarantine
        and table_row_counts["fact_trips"] == total_clean
        and table_row_counts["trips_quarantine"] == total_quarantine
        and all(s.is_reconciled for s in file_summaries)
    )

    if not db_reconciled:
        raise RuntimeError(
            "Database reconciliation check failed: "
            f"raw={total_raw}, clean={total_clean} (db={table_row_counts['fact_trips']}), "
            f"quarantine={total_quarantine} (db={table_row_counts['trips_quarantine']})"
        )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "duckdb_path": db_path.as_posix(),
        "totals": {
            "files_processed": len(file_summaries),
            "raw_rows": total_raw,
            "clean_rows": total_clean,
            "quarantine_rows": total_quarantine,
            "is_reconciled": db_reconciled,
            "rejection_counts": total_rejections,
        },
        "table_row_counts": table_row_counts,
        "files": [
            {
                **asdict(s),
                "is_reconciled": s.is_reconciled,
            }
            for s in file_summaries
        ],
    }

    report_json_path = cfg.processed_dir / "dq_report.json"
    report_md_path = cfg.processed_dir / "DQ_REPORT.md"

    report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_md_path.write_text(render_dq_markdown(report), encoding="utf-8")

    logger.info(
        "[DONE] Star schema built at %s | raw=%s, clean=%s, quarantine=%s | Report: %s",
        db_path,
        f"{total_raw:,}",
        f"{total_clean:,}",
        f"{total_quarantine:,}",
        report_md_path,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for building the DuckDB star schema."""
    parser = argparse.ArgumentParser(
        description="Build the DuckDB analytical star schema and DQ reconciliation report."
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of overwriting an existing DuckDB database file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    build_star_schema(overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
