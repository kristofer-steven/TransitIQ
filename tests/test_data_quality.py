"""Unit and integration tests for Week 2 Data Quality, Star Schema, Catalog, Glossary, and DB."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from data.db import execute_readonly_query, readonly_connection
from data.ingestion.config import IngestionSettings
from data.ingestion.generate_dimensions import generate_all_dimensions
from data.preprocessing.build_duckdb import build_star_schema
from data.quality.rules import apply_data_quality_rules
from data.schemas.date_resolver import (
    extract_and_resolve_dates,
    find_out_of_range_years,
    resolve_relative_expression,
)
from data.schemas.loader import (
    load_catalog,
    load_glossary,
    verify_catalog_against_duckdb,
)


def test_dq_rules_and_exact_reconciliation() -> None:
    """Verify all 6 quarantine rules, non-quarantine cleanups, and raw == clean + quarantine."""
    raw_df = pl.DataFrame(
        {
            "VendorID": [1, 1, 2, 1, 1, 1, 1, 2, 99],
            "tpep_pickup_datetime": [
                datetime(2024, 1, 10, 8, 0, 0),   # 1. Valid trip
                datetime(2024, 1, 10, 8, 0, 0),   # 2. Exact duplicate of row 1
                datetime(2024, 1, 10, 9, 30, 0),  # 3. Invalid timestamp (dropoff <= pickup)
                datetime(2023, 12, 31, 23, 50, 0),  # 4. Out-of-month pickup (Dec 2023 in Jan 2024)
                datetime(2024, 1, 15, 10, 0, 0),  # 5. Excessive duration (> 24h)
                datetime(2024, 1, 16, 12, 0, 0),  # 6. Invalid distance (<= 0)
                datetime(2024, 1, 17, 14, 0, 0),  # 7. Invalid distance (> 200 mi)
                datetime(2024, 1, 18, 15, 0, 0),  # 8. Negative fare/total amount
                datetime(2024, 1, 19, 18, 0, 0),  # 9. Valid trip with pax=0 and unknown zone=999
            ],
            "tpep_dropoff_datetime": [
                datetime(2024, 1, 10, 8, 20, 0),
                datetime(2024, 1, 10, 8, 20, 0),
                datetime(2024, 1, 10, 9, 15, 0),  # dropoff before pickup
                datetime(2024, 1, 1, 0, 10, 0),
                datetime(2024, 1, 16, 12, 0, 0),  # 26 hours duration
                datetime(2024, 1, 16, 12, 15, 0),
                datetime(2024, 1, 17, 15, 0, 0),
                datetime(2024, 1, 18, 15, 20, 0),
                datetime(2024, 1, 19, 18, 30, 0),
            ],
            "PULocationID": [132, 132, 100, 100, 100, 100, 100, 100, 999],
            "DOLocationID": [138, 138, 101, 101, 101, 101, 101, 101, 230],
            "passenger_count": [2, 2, 1, 1, 1, 1, 1, 1, 0],
            "trip_distance": [12.5, 12.5, 3.0, 2.5, 10.0, 0.0, 250.0, 4.0, 5.5],
            "RatecodeID": [2, 2, 1, 1, 1, 1, 1, 1, 88],
            "payment_type": [1, 1, 1, 1, 1, 1, 1, 2, 77],
            "fare_amount": [45.0, 45.0, 15.0, 12.0, 50.0, 10.0, 80.0, -15.0, 22.0],
            "tip_amount": [9.0, 9.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0],
            "tolls_amount": [6.94, 6.94, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "congestion_surcharge": [2.5, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0, -2.5, 2.5],
            "Airport_fee": [1.75, 1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "total_amount": [65.19, 65.19, 20.5, 16.5, 50.0, 10.0, 80.0, -17.5, 28.5],
        }
    )

    result = apply_data_quality_rules(
        raw_df=raw_df,
        service_type="yellow",
        year_month="2024-01",
        source_file="yellow_tripdata_2024-01.parquet",
    )

    assert result.summary.is_reconciled is True
    assert result.summary.raw_rows == 9
    assert result.summary.clean_rows == 2
    assert result.summary.quarantine_rows == 7

    counts = result.summary.rejection_counts
    assert counts["null_or_invalid_timestamp"] == 1
    assert counts["out_of_month_pickup"] == 1
    assert counts["excessive_duration"] == 1
    assert counts["invalid_distance"] == 2
    assert counts["negative_amount"] == 1
    assert counts["duplicate_trip"] == 1

    # Verify non-quarantine normalization on row 9 (trip_id == 9)
    row9 = result.clean_df.filter(pl.col("trip_id") == 9).row(0, named=True)
    assert row9["passenger_count"] is None
    assert row9["pickup_zone_id"] == 264  # out-of-range 999 mapped to 264 (Unknown)
    assert row9["vendor_id"] == 1  # out-of-range 99 mapped to 1
    assert row9["rate_code_id"] == 99  # out-of-range 88 mapped to 99 (Unknown)
    assert row9["payment_type_id"] == 5  # out-of-range 77 mapped to 5 (Unknown)
    assert row9["trip_duration_min"] == pytest.approx(30.0, rel=1e-4)


def _create_mini_raw_environment(tmp_path: Path) -> IngestionSettings:
    """Create a self-contained mini raw dataset in tmp_path for star-schema integration testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    cfg = IngestionSettings(
        data_dir=tmp_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        duckdb_path=processed_dir / "test_transitiq.duckdb",
    )

    for folder in (
        cfg.yellow_raw_dir,
        cfg.green_raw_dir,
        cfg.reference_raw_dir,
        cfg.weather_raw_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Generate static reference dimensions (dim_date, payment, rate_code, vendor)
    generate_all_dimensions(cfg)

    # 2. Write mini taxi_zone_lookup.csv covering used zones + 264/265
    zone_csv = cfg.reference_raw_dir / "taxi_zone_lookup.csv"
    zone_csv.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "1,EWR,Newark Airport,EWR\n"
        "132,Queens,JFK Airport,Airports\n"
        "138,Queens,LaGuardia Airport,Airports\n"
        "161,Manhattan,Midtown Center,Yellow Zone\n"
        "264,Unknown,NV,N/A\n"
        "265,Unknown,NA,N/A\n",
        encoding="utf-8",
    )

    # 3. Write mini weather_daily parquet
    weather_df = pl.DataFrame(
        {
            "date": [date(2024, 1, 10), date(2024, 1, 11)],
            "precip_in": [0.15, 0.0],
            "snow_in": [0.0, 0.0],
            "tmax_f": [45.0, 42.0],
            "tmin_f": [34.0, 30.0],
        },
        schema={
            "date": pl.Date,
            "precip_in": pl.Float64,
            "snow_in": pl.Float64,
            "tmax_f": pl.Float64,
            "tmin_f": pl.Float64,
        },
    )
    weather_path = (
        cfg.weather_raw_dir
        / f"central_park_weather_{cfg.tlc_start_year}_{cfg.tlc_end_year}.parquet"
    )
    weather_df.write_parquet(weather_path)

    # 4. Write sample yellow and green monthly trip Parquet files
    yellow_df = pl.DataFrame(
        {
            "VendorID": [1, 2],
            "tpep_pickup_datetime": [
                datetime(2024, 1, 10, 8, 15, 0),
                datetime(2024, 1, 10, 9, 0, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2024, 1, 10, 8, 45, 0),
                datetime(2024, 1, 10, 9, 10, 0),
            ],
            "PULocationID": [132, 161],
            "DOLocationID": [161, 138],
            "passenger_count": [1, 2],
            "trip_distance": [17.2, -1.0],  # 2nd row quarantined (invalid_distance)
            "RatecodeID": [2, 1],
            "payment_type": [1, 2],
            "fare_amount": [70.0, 10.0],
            "tip_amount": [14.0, 0.0],
            "tolls_amount": [6.94, 0.0],
            "congestion_surcharge": [2.5, 2.5],
            "Airport_fee": [1.75, 0.0],
            "total_amount": [95.19, 12.5],
        }
    )
    yellow_df.write_parquet(cfg.yellow_raw_dir / "yellow_tripdata_2024-01.parquet")

    green_df = pl.DataFrame(
        {
            "VendorID": [2],
            "lpep_pickup_datetime": [datetime(2024, 1, 11, 18, 0, 0)],
            "lpep_dropoff_datetime": [datetime(2024, 1, 11, 18, 20, 0)],
            "PULocationID": [138],
            "DOLocationID": [132],
            "passenger_count": [1],
            "trip_distance": [8.4],
            "RatecodeID": [1],
            "payment_type": [1],
            "fare_amount": [32.0],
            "tip_amount": [5.0],
            "tolls_amount": [0.0],
            "congestion_surcharge": [0.0],
            "total_amount": [37.0],
        }
    )
    green_df.write_parquet(cfg.green_raw_dir / "green_tripdata_2024-01.parquet")

    return cfg


def test_star_schema_build_referential_integrity_and_catalog(tmp_path: Path) -> None:
    """Verify full DuckDB build, DQ reconciliation, FK integrity, and catalog conformance."""
    cfg = _create_mini_raw_environment(tmp_path)
    report = build_star_schema(cfg=cfg)

    assert report["totals"]["is_reconciled"] is True
    assert report["totals"]["raw_rows"] == 3
    assert report["totals"]["clean_rows"] == 2
    assert report["totals"]["quarantine_rows"] == 1

    with readonly_connection(db_path=cfg.duckdb_path) as conn:
        # 1. Verify catalog.yaml matches the built DuckDB schema 100%
        discrepancies = verify_catalog_against_duckdb(conn)
        assert discrepancies == []

        # 2. Verify referential integrity across all foreign keys
        orphan_pu = conn.execute(
            "SELECT COUNT(*) FROM fact_trips t "
            "LEFT JOIN dim_zone z ON t.pickup_zone_id = z.zone_id WHERE z.zone_id IS NULL;"
        ).fetchone()
        assert orphan_pu is not None and orphan_pu[0] == 0

        orphan_do = conn.execute(
            "SELECT COUNT(*) FROM fact_trips t "
            "LEFT JOIN dim_zone z ON t.dropoff_zone_id = z.zone_id WHERE z.zone_id IS NULL;"
        ).fetchone()
        assert orphan_do is not None and orphan_do[0] == 0

        orphan_date = conn.execute(
            "SELECT COUNT(*) FROM fact_trips t "
            "LEFT JOIN dim_date d ON CAST(t.pickup_datetime AS DATE) = d.date "
            "WHERE d.date IS NULL;"
        ).fetchone()
        assert orphan_date is not None and orphan_date[0] == 0

        # 3. Verify read-only guardrails reject write statements and external file reads
        with pytest.raises(duckdb.Error):
            conn.execute("DROP TABLE fact_trips;")

        with pytest.raises(duckdb.Error):
            conn.execute("SELECT * FROM read_parquet('nonexistent.parquet');")

    res = execute_readonly_query(
        "SELECT SUM(pickups) FROM agg_zone_hourly_demand;",
        db_path=cfg.duckdb_path,
    )
    assert res.rows[0][0] == 2


def test_catalog_fk_join_graph_and_glossary() -> None:
    """Verify catalog FK join completion and business glossary term matching."""
    catalog = load_catalog()
    assert len(catalog.allowlisted_table_names()) == 9

    expanded, joins = catalog.find_connecting_tables_and_joins({"fact_trips", "dim_zone"})
    assert "fact_trips" in expanded
    assert "dim_zone" in expanded
    assert len(joins) >= 2  # pickup_zone_id and dropoff_zone_id joins

    glossary = load_glossary()
    assert len(glossary.terms) == 14
    matched = glossary.match_terms_in_text(
        "What was the average tip and pickup demand during evening peak?"
    )
    matched_names = {m.term for m in matched}
    assert {"average_tip", "demand", "evening_peak"}.issubset(matched_names)


def test_relative_date_resolver() -> None:
    """Verify deterministic relative date resolution against as_of_date = 2024-12-31."""
    as_of = date(2024, 12, 31)  # Tuesday

    yesterday = resolve_relative_expression("yesterday", as_of_date=as_of)
    assert yesterday.start_date == date(2024, 12, 30)
    assert yesterday.end_date == date(2024, 12, 30)

    last_friday = resolve_relative_expression("last Friday", as_of_date=as_of)
    assert last_friday.start_date == date(2024, 12, 27)
    assert last_friday.end_date == date(2024, 12, 27)

    last_tuesday = resolve_relative_expression("last Tuesday", as_of_date=as_of)
    assert last_tuesday.start_date == date(2024, 12, 24)

    last_month = resolve_relative_expression("last month", as_of_date=as_of)
    assert last_month.start_date == date(2024, 11, 1)
    assert last_month.end_date == date(2024, 11, 30)

    last_week = resolve_relative_expression("last week", as_of_date=as_of)
    assert last_week.start_date == date(2024, 12, 23)
    assert last_week.end_date == date(2024, 12, 29)

    last_7_days = resolve_relative_expression("last 7 days", as_of_date=as_of)
    assert last_7_days.start_date == date(2024, 12, 25)
    assert last_7_days.end_date == date(2024, 12, 31)

    extracted = extract_and_resolve_dates(
        "Compare demand yesterday vs last Friday",
        as_of_date=as_of,
    )
    assert len(extracted) == 2

    out_of_range = find_out_of_range_years("How many trips occurred in 2019 vs 2024?")
    assert out_of_range == [2019]
