"""Downloader and processor for NOAA GHCN-Daily Central Park weather data."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

from data.ingestion.config import IngestionSettings, settings
from data.ingestion.download_tlc import is_valid_parquet, stream_download_with_retry

logger = logging.getLogger(__name__)

MISSING_SENTINEL = -9999.0


def tenths_mm_to_inches(value: float | int | None) -> float | None:
    """Convert tenths of millimeters (GHCN-Daily PRCP) to inches."""
    if value is None or value <= MISSING_SENTINEL:
        return None
    if value < 0:
        return None
    return round((float(value) / 10.0) / 25.4, 4)


def mm_to_inches(value: float | int | None) -> float | None:
    """Convert millimeters (GHCN-Daily SNOW) to inches."""
    if value is None or value <= MISSING_SENTINEL:
        return None
    if value < 0:
        return None
    return round(float(value) / 25.4, 4)


def tenths_c_to_fahrenheit(value: float | int | None) -> float | None:
    """Convert tenths of degrees Celsius (GHCN-Daily TMAX/TMIN) to degrees Fahrenheit."""
    if value is None or value <= MISSING_SENTINEL:
        return None
    celsius = float(value) / 10.0
    return round((celsius * 9.0 / 5.0) + 32.0, 2)


def _parse_optional_float(raw: str | None) -> float | None:
    """Parse a raw CSV string into a float, returning None for empty or sentinel values."""
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        val = float(stripped)
    except ValueError:
        return None
    if val <= MISSING_SENTINEL:
        return None
    return val


def build_date_spine(start_date: date, end_date: date) -> list[date]:
    """Generate a contiguous list of dates from start_date to end_date inclusive."""
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")
    total_days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(total_days)]


def parse_noaa_csv_to_dataframe(
    raw_csv_path: Path,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Parse a NOAA GHCN-Daily station CSV into the clean weather_daily schema."""
    records_by_date: dict[date, dict[str, float | None]] = {}

    with raw_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_date = (row.get("DATE") or "").strip()
            if not raw_date:
                continue
            try:
                obs_date = date.fromisoformat(raw_date)
            except ValueError:
                continue

            if obs_date < start_date or obs_date > end_date:
                continue

            prcp_raw = _parse_optional_float(row.get("PRCP"))
            snow_raw = _parse_optional_float(row.get("SNOW"))
            tmax_raw = _parse_optional_float(row.get("TMAX"))
            tmin_raw = _parse_optional_float(row.get("TMIN"))

            records_by_date[obs_date] = {
                "precip_in": tenths_mm_to_inches(prcp_raw),
                "snow_in": mm_to_inches(snow_raw),
                "tmax_f": tenths_c_to_fahrenheit(tmax_raw),
                "tmin_f": tenths_c_to_fahrenheit(tmin_raw),
            }

    spine = build_date_spine(start_date, end_date)
    rows = []
    for d in spine:
        metrics = records_by_date.get(
            d,
            {"precip_in": None, "snow_in": None, "tmax_f": None, "tmin_f": None},
        )
        rows.append(
            {
                "date": d,
                "precip_in": metrics["precip_in"],
                "snow_in": metrics["snow_in"],
                "tmax_f": metrics["tmax_f"],
                "tmin_f": metrics["tmin_f"],
            }
        )

    schema = {
        "date": pl.Date,
        "precip_in": pl.Float64,
        "snow_in": pl.Float64,
        "tmax_f": pl.Float64,
        "tmin_f": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort("date")


def run_weather_ingestion(
    cfg: IngestionSettings = settings,
    force: bool = False,
) -> Path:
    """Download NOAA station CSV and write curated weather_daily Parquet and CSV files."""
    start_date = date(cfg.tlc_start_year, 1, 1)
    end_date = date(cfg.tlc_end_year, 12, 31)
    expected_days = (end_date - start_date).days + 1

    cfg.weather_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_station_csv = cfg.weather_raw_dir / f"{cfg.noaa_station_id}_raw.csv"
    output_stem = f"central_park_weather_{cfg.tlc_start_year}_{cfg.tlc_end_year}"
    output_parquet = cfg.weather_raw_dir / f"{output_stem}.parquet"
    output_csv = cfg.weather_raw_dir / f"{output_stem}.csv"

    if not force:
        valid, existing_rows = is_valid_parquet(output_parquet)
        if valid and existing_rows == expected_days:
            logger.info(
                "[SKIP] %s already exists (%d days)",
                output_parquet.name,
                existing_rows,
            )
            return output_parquet

    url = cfg.noaa_station_url()
    if force or not raw_station_csv.is_file() or raw_station_csv.stat().st_size == 0:
        logger.info("[GET ] Downloading NOAA station data %s -> %s", url, raw_station_csv)
        timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
        headers = {"User-Agent": "TransitIQ-DataIngestion/0.1"}
        with httpx.Client(timeout=timeout, headers=headers) as client:
            stream_download_with_retry(client, url, raw_station_csv)

    df = parse_noaa_csv_to_dataframe(
        raw_csv_path=raw_station_csv,
        start_date=start_date,
        end_date=end_date,
    )

    df.write_parquet(output_parquet)
    df.write_csv(output_csv)

    non_null_days = df.filter(pl.col("tmax_f").is_not_null()).height
    logger.info(
        "[DONE] Wrote %s (%d total days, %d days with observed temperature)",
        output_parquet.name,
        df.height,
        non_null_days,
    )
    return output_parquet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for weather ingestion."""
    parser = argparse.ArgumentParser(
        description="Download and process NOAA GHCN-Daily Central Park weather data."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-process weather files even if they already exist.",
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
    run_weather_ingestion(force=args.force)


if __name__ == "__main__":
    main()
