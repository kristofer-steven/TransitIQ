"""Deterministic generators for calendar and TLC lookup reference dimension tables."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import holidays
import polars as pl

from data.ingestion.config import IngestionSettings, settings

logger = logging.getLogger(__name__)


def build_dim_date(start_year: int = 2023, end_year: int = 2024) -> pl.DataFrame:
    """Generate the dim_date table for [start_year-01-01, end_year-12-31] with US holidays."""
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")

    years = list(range(start_year, end_year + 1))
    us_holidays = holidays.country_holidays("US", years=years)

    start_dt = date(start_year, 1, 1)
    end_dt = date(end_year, 12, 31)
    total_days = (end_dt - start_dt).days + 1

    rows = []
    for offset in range(total_days):
        current = start_dt + timedelta(days=offset)
        iso_weekday = current.isoweekday()  # 1 = Monday ... 7 = Sunday
        holiday_label = us_holidays.get(current)
        rows.append(
            {
                "date": current,
                "year": current.year,
                "month": current.month,
                "day_of_week": iso_weekday,
                "day_name": current.strftime("%A"),
                "is_weekend": iso_weekday in (6, 7),
                "is_holiday": holiday_label is not None,
                "holiday_name": str(holiday_label) if holiday_label is not None else None,
            }
        )

    schema = {
        "date": pl.Date,
        "year": pl.Int32,
        "month": pl.Int32,
        "day_of_week": pl.Int32,
        "day_name": pl.String,
        "is_weekend": pl.Boolean,
        "is_holiday": pl.Boolean,
        "holiday_name": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).sort("date")


def build_dim_payment_type() -> pl.DataFrame:
    """Generate the official NYC TLC payment_type lookup table."""
    rows = [
        {"payment_type_id": 0, "payment_type": "Flex Fare trip"},
        {"payment_type_id": 1, "payment_type": "Credit card"},
        {"payment_type_id": 2, "payment_type": "Cash"},
        {"payment_type_id": 3, "payment_type": "No charge"},
        {"payment_type_id": 4, "payment_type": "Dispute"},
        {"payment_type_id": 5, "payment_type": "Unknown"},
        {"payment_type_id": 6, "payment_type": "Voided trip"},
    ]
    return pl.DataFrame(
        rows,
        schema={"payment_type_id": pl.Int32, "payment_type": pl.String},
    ).sort("payment_type_id")


def build_dim_rate_code() -> pl.DataFrame:
    """Generate the official NYC TLC rate_code lookup table."""
    rows = [
        {"rate_code_id": 1, "rate_code": "Standard rate"},
        {"rate_code_id": 2, "rate_code": "JFK"},
        {"rate_code_id": 3, "rate_code": "Newark"},
        {"rate_code_id": 4, "rate_code": "Nassau or Westchester"},
        {"rate_code_id": 5, "rate_code": "Negotiated fare"},
        {"rate_code_id": 6, "rate_code": "Group ride"},
        {"rate_code_id": 99, "rate_code": "Unknown"},
    ]
    return pl.DataFrame(
        rows,
        schema={"rate_code_id": pl.Int32, "rate_code": pl.String},
    ).sort("rate_code_id")


def build_dim_vendor() -> pl.DataFrame:
    """Generate the official NYC TLC vendor lookup table."""
    rows = [
        {"vendor_id": 1, "vendor_name": "Creative Mobile Technologies, LLC"},
        {"vendor_id": 2, "vendor_name": "Curb Mobility, LLC"},
        {"vendor_id": 6, "vendor_name": "Myle Technologies Inc."},
        {"vendor_id": 7, "vendor_name": "Helix"},
    ]
    return pl.DataFrame(
        rows,
        schema={"vendor_id": pl.Int32, "vendor_name": pl.String},
    ).sort("vendor_id")


def generate_all_dimensions(cfg: IngestionSettings = settings) -> dict[str, Path]:
    """Generate and persist all static reference tables as Parquet and CSV files."""
    ref_dir = cfg.reference_raw_dir
    ref_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pl.DataFrame] = {
        "dim_date": build_dim_date(cfg.tlc_start_year, cfg.tlc_end_year),
        "dim_payment_type": build_dim_payment_type(),
        "dim_rate_code": build_dim_rate_code(),
        "dim_vendor": build_dim_vendor(),
    }

    written_paths: dict[str, Path] = {}
    for name, df in tables.items():
        parquet_path = ref_dir / f"{name}.parquet"
        csv_path = ref_dir / f"{name}.csv"
        df.write_parquet(parquet_path)
        df.write_csv(csv_path)
        written_paths[name] = parquet_path
        logger.info("[DONE] Generated %s (%d rows) -> %s", name, df.height, parquet_path)

    return written_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dimension generation."""
    parser = argparse.ArgumentParser(
        description="Generate calendar and NYC TLC lookup dimension tables."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parse_args(argv)
    generate_all_dimensions()


if __name__ == "__main__":
    main()
