"""Unit tests for Week 1 raw data ingestion, reference generators, and manifest verification."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.ingestion.download_tlc import (
    build_month_range,
    is_valid_parquet,
    validate_year_month,
)
from data.ingestion.download_weather import (
    mm_to_inches,
    parse_noaa_csv_to_dataframe,
    tenths_c_to_fahrenheit,
    tenths_mm_to_inches,
)
from data.ingestion.generate_dimensions import (
    build_dim_date,
    build_dim_payment_type,
    build_dim_rate_code,
    build_dim_vendor,
)
from data.ingestion.verify_raw import detect_schema_drift, inspect_parquet_file


def test_dim_date_generation() -> None:
    """Verify dim_date covers all 731 days of 2023-2024 with accurate weekend & holiday flags."""
    df = build_dim_date(2023, 2024)
    assert df.height == 731
    assert df["date"].min() == date(2023, 1, 1)
    assert df["date"].max() == date(2024, 12, 31)
    assert df["date"].n_unique() == 731

    # Check major holidays in 2024
    for holiday_dt in (date(2024, 1, 1), date(2024, 7, 4), date(2024, 12, 25)):
        row = df.filter(pl.col("date") == holiday_dt).row(0, named=True)
        assert row["is_holiday"] is True
        assert row["holiday_name"] is not None and len(row["holiday_name"]) > 0

    # Check weekend logic matches ISO weekday (6=Saturday, 7=Sunday)
    weekend_mismatch = df.filter(
        pl.col("is_weekend") != pl.col("day_of_week").is_in([6, 7])
    )
    assert weekend_mismatch.height == 0


def test_lookup_tables_schema() -> None:
    """Ensure payment_type, rate_code, and vendor dimensions have unique keys and non-null names."""
    payment_df = build_dim_payment_type()
    assert payment_df.height == 7
    assert payment_df["payment_type_id"].n_unique() == payment_df.height
    assert payment_df["payment_type"].null_count() == 0

    rate_df = build_dim_rate_code()
    assert rate_df.height == 7
    assert rate_df["rate_code_id"].n_unique() == rate_df.height
    assert rate_df["rate_code"].null_count() == 0

    vendor_df = build_dim_vendor()
    assert vendor_df.height == 4
    assert vendor_df["vendor_id"].n_unique() == vendor_df.height
    assert vendor_df["vendor_name"].null_count() == 0


def test_weather_unit_conversion() -> None:
    """Verify GHCN-Daily unit conversions and sentinel (-9999) handling."""
    # 254 tenths of mm = 25.4 mm = 1.0 inch
    assert tenths_mm_to_inches(254) == pytest.approx(1.0, rel=1e-4)
    assert tenths_mm_to_inches(0) == 0.0
    assert tenths_mm_to_inches(-9999) is None
    assert tenths_mm_to_inches(None) is None

    # 25.4 mm snow = 1.0 inch
    assert mm_to_inches(25.4) == pytest.approx(1.0, rel=1e-4)
    assert mm_to_inches(-9999) is None

    # 0.0 C -> 32.0 F, 100 tenths C (10.0 C) -> 50.0 F, 250 tenths C (25.0 C) -> 77.0 F
    assert tenths_c_to_fahrenheit(0) == pytest.approx(32.0, rel=1e-4)
    assert tenths_c_to_fahrenheit(100) == pytest.approx(50.0, rel=1e-4)
    assert tenths_c_to_fahrenheit(250) == pytest.approx(77.0, rel=1e-4)
    assert tenths_c_to_fahrenheit(-9999) is None


def test_parse_noaa_csv_to_dataframe(tmp_path: Path) -> None:
    """Verify NOAA CSV parsing fills the full date spine and converts units properly."""
    sample_csv = tmp_path / "station.csv"
    sample_csv.write_text(
        "STATION,DATE,PRCP,SNOW,TMAX,TMIN\n"
        "USW00094728,2024-01-01,254,0,100,0\n"
        "USW00094728,2024-01-02,-9999,-9999,-9999,-9999\n",
        encoding="utf-8",
    )

    df = parse_noaa_csv_to_dataframe(
        raw_csv_path=sample_csv,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )
    assert df.height == 3

    jan1 = df.filter(pl.col("date") == date(2024, 1, 1)).row(0, named=True)
    assert jan1["precip_in"] == pytest.approx(1.0, rel=1e-4)
    assert jan1["snow_in"] == 0.0
    assert jan1["tmax_f"] == pytest.approx(50.0, rel=1e-4)
    assert jan1["tmin_f"] == pytest.approx(32.0, rel=1e-4)

    # Sentinel row (Jan 2) and missing date spine row (Jan 3) should have None metrics
    jan2 = df.filter(pl.col("date") == date(2024, 1, 2)).row(0, named=True)
    assert jan2["precip_in"] is None
    assert jan2["tmax_f"] is None

    jan3 = df.filter(pl.col("date") == date(2024, 1, 3)).row(0, named=True)
    assert jan3["precip_in"] is None


def test_parquet_manifest_inspector_and_schema_drift(tmp_path: Path) -> None:
    """Verify Parquet metadata inspection and detection of column case/type/extra drift."""
    p1 = tmp_path / "yellow_tripdata_2024-01.parquet"
    p2 = tmp_path / "yellow_tripdata_2024-02.parquet"

    table1 = pa.table(
        {
            "VendorID": pa.array([1, 2, 1], type=pa.int32()),
            "Airport_fee": pa.array([0.0, 1.75, 0.0], type=pa.float64()),
            "fare_amount": pa.array([12.5, 45.0, 8.0], type=pa.float64()),
        }
    )
    table2 = pa.table(
        {
            "VendorID": pa.array([1, 2], type=pa.int64()),  # type drift: int32 -> int64
            "airport_fee": pa.array([0.0, 1.75], type=pa.float64()),  # case drift
            "extra_col": pa.array(["a", "b"], type=pa.string()),  # extra / missing col
        }
    )

    pq.write_table(table1, p1)
    pq.write_table(table2, p2)

    info1 = inspect_parquet_file(p1, base_dir=tmp_path)
    info2 = inspect_parquet_file(p2, base_dir=tmp_path)

    assert info1["service_type"] == "yellow"
    assert info1["year_month"] == "2024-01"
    assert info1["num_rows"] == 3
    assert info1["sha256"] is not None

    assert info2["num_rows"] == 2

    drifts = detect_schema_drift([info1, info2])
    drift_types = {d["drift_type"] for d in drifts}
    assert "case_mismatch" in drift_types
    assert "type_mismatch" in drift_types
    assert "missing_column" in drift_types
    assert "extra_column" in drift_types


def test_tlc_helpers(tmp_path: Path) -> None:
    """Verify month range builder, YYYY-MM validator, and Parquet footer validator."""
    months = build_month_range(2023, 2024)
    assert len(months) == 24
    assert months[0] == "2023-01"
    assert months[-1] == "2024-12"

    assert validate_year_month("2024-01") == "2024-01"
    with pytest.raises(ValueError):
        validate_year_month("2024-13")

    bad_parquet = tmp_path / "corrupt.parquet"
    bad_parquet.write_bytes(b"not a parquet file")
    valid, rows = is_valid_parquet(bad_parquet)
    assert valid is False
    assert rows is None
