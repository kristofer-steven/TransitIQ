"""Idempotent downloader for NYC TLC trip Parquet files and taxi zone lookup."""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import pyarrow.parquet as pq

from data.ingestion.config import IngestionSettings, settings

logger = logging.getLogger(__name__)

ServiceType = Literal["yellow", "green"]
VALID_SERVICES: tuple[ServiceType, ...] = ("yellow", "green")
SAMPLE_MONTHS: tuple[str, ...] = ("2024-01",)
YEAR_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class DownloadResult:
    """Result metadata for a single downloaded or skipped file."""

    target_path: Path
    url: str
    status: Literal["downloaded", "skipped", "failed"]
    size_bytes: int = 0
    num_rows: int | None = None
    error: str | None = None


def build_month_range(start_year: int, end_year: int) -> list[str]:
    """Generate a sorted list of 'YYYY-MM' strings for the given inclusive year range."""
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")
    return [
        f"{year:04d}-{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def validate_year_month(value: str) -> str:
    """Validate that a string matches the 'YYYY-MM' format."""
    if not YEAR_MONTH_PATTERN.match(value):
        raise ValueError(f"Invalid year-month '{value}'. Expected format 'YYYY-MM'.")
    return value


def is_valid_parquet(path: Path) -> tuple[bool, int | None]:
    """Check whether a local Parquet file exists, is non-empty, and has a readable footer."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, None
    try:
        metadata = pq.read_metadata(path)
        return True, metadata.num_rows
    except Exception as exc:
        logger.warning("Existing Parquet file %s is invalid (%s); will re-download.", path, exc)
        return False, None


def is_valid_csv(path: Path) -> bool:
    """Check whether a local CSV file exists and has at least a header and one data row."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline()
            second_line = fh.readline()
        return bool(first_line.strip()) and bool(second_line.strip())
    except OSError:
        return False


def stream_download_with_retry(
    client: httpx.Client,
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> int:
    """Download a URL to dest_path atomically via a temporary '.part' file with retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code in TRANSIENT_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Transient HTTP {response.status_code} for {url}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()

                bytes_written = 0
                with part_path.open("wb") as out_file:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE_BYTES):
                        out_file.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written == 0:
                raise OSError(f"Downloaded 0 bytes from {url}")

            part_path.replace(dest_path)
            return bytes_written

        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink(missing_ok=True)

            if attempt < max_retries:
                sleep_time = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d failed for %s (%s). Retrying in %.1fs...",
                    attempt,
                    max_retries,
                    url,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_error}")


def download_trip_parquet(
    client: httpx.Client,
    service_type: ServiceType,
    year_month: str,
    cfg: IngestionSettings = settings,
    force: bool = False,
) -> DownloadResult:
    """Download and verify a single monthly NYC TLC Parquet file."""
    validate_year_month(year_month)
    target_dir = cfg.yellow_raw_dir if service_type == "yellow" else cfg.green_raw_dir
    filename = f"{service_type}_tripdata_{year_month}.parquet"
    dest_path = target_dir / filename
    url = cfg.tlc_parquet_url(service_type, year_month)

    if not force:
        valid, existing_rows = is_valid_parquet(dest_path)
        if valid:
            size_bytes = dest_path.stat().st_size
            logger.info(
                "[SKIP] %s already exists (%s rows, %.2f MiB)",
                dest_path.name,
                f"{existing_rows:,}" if existing_rows is not None else "?",
                size_bytes / (1024 * 1024),
            )
            return DownloadResult(
                target_path=dest_path,
                url=url,
                status="skipped",
                size_bytes=size_bytes,
                num_rows=existing_rows,
            )

    logger.info("[GET ] Downloading %s -> %s", url, dest_path)
    try:
        size_bytes = stream_download_with_retry(client, url, dest_path)
        valid, num_rows = is_valid_parquet(dest_path)
        if not valid:
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file {dest_path} failed Parquet footer verification.")

        logger.info(
            "[DONE] %s (%s rows, %.2f MiB)",
            dest_path.name,
            f"{num_rows:,}" if num_rows is not None else "?",
            size_bytes / (1024 * 1024),
        )
        return DownloadResult(
            target_path=dest_path,
            url=url,
            status="downloaded",
            size_bytes=size_bytes,
            num_rows=num_rows,
        )
    except Exception as exc:
        logger.error("[FAIL] %s: %s", dest_path.name, exc)
        return DownloadResult(
            target_path=dest_path,
            url=url,
            status="failed",
            error=str(exc),
        )


def download_zone_lookup(
    client: httpx.Client,
    cfg: IngestionSettings = settings,
    force: bool = False,
) -> DownloadResult:
    """Download the NYC TLC taxi_zone_lookup.csv reference file."""
    dest_path = cfg.reference_raw_dir / "taxi_zone_lookup.csv"
    url = cfg.tlc_zone_lookup_url

    if not force and is_valid_csv(dest_path):
        size_bytes = dest_path.stat().st_size
        with dest_path.open("r", encoding="utf-8") as fh:
            row_count = max(sum(1 for _ in fh) - 1, 0)
        logger.info("[SKIP] %s already exists (%d zones)", dest_path.name, row_count)
        return DownloadResult(
            target_path=dest_path,
            url=url,
            status="skipped",
            size_bytes=size_bytes,
            num_rows=row_count,
        )

    logger.info("[GET ] Downloading %s -> %s", url, dest_path)
    try:
        size_bytes = stream_download_with_retry(client, url, dest_path)
        if not is_valid_csv(dest_path):
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded CSV {dest_path} is empty or malformed.")

        with dest_path.open("r", encoding="utf-8") as fh:
            row_count = max(sum(1 for _ in fh) - 1, 0)

        logger.info("[DONE] %s (%d zones, %d bytes)", dest_path.name, row_count, size_bytes)
        return DownloadResult(
            target_path=dest_path,
            url=url,
            status="downloaded",
            size_bytes=size_bytes,
            num_rows=row_count,
        )
    except Exception as exc:
        logger.error("[FAIL] %s: %s", dest_path.name, exc)
        return DownloadResult(
            target_path=dest_path,
            url=url,
            status="failed",
            error=str(exc),
        )


def run_download(
    mode: Literal["sample", "full"] = "sample",
    months: list[str] | None = None,
    services: list[ServiceType] | None = None,
    include_zones: bool = True,
    force: bool = False,
    cfg: IngestionSettings = settings,
) -> list[DownloadResult]:
    """Execute the TLC download batch for the selected mode/months and services."""
    if months:
        target_months = [validate_year_month(m) for m in months]
    elif mode == "sample":
        target_months = list(SAMPLE_MONTHS)
    else:
        target_months = build_month_range(cfg.tlc_start_year, cfg.tlc_end_year)

    target_services: list[ServiceType] = list(services) if services else list(VALID_SERVICES)
    results: list[DownloadResult] = []

    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
    headers = {"User-Agent": "TransitIQ-DataIngestion/0.1"}

    with httpx.Client(timeout=timeout, headers=headers) as client:
        if include_zones:
            results.append(download_zone_lookup(client, cfg=cfg, force=force))

        for service in target_services:
            for ym in target_months:
                res = download_trip_parquet(
                    client=client,
                    service_type=service,
                    year_month=ym,
                    cfg=cfg,
                    force=force,
                )
                results.append(res)

    downloaded = sum(1 for r in results if r.status == "downloaded")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    total_bytes = sum(r.size_bytes for r in results)
    total_rows = sum(r.num_rows or 0 for r in results)

    logger.info(
        "Summary: %d downloaded, %d skipped, %d failed | Total rows: %s | Total size: %.2f MiB",
        downloaded,
        skipped,
        failed,
        f"{total_rows:,}",
        total_bytes / (1024 * 1024),
    )

    if failed > 0:
        failed_files = ", ".join(r.target_path.name for r in results if r.status == "failed")
        raise RuntimeError(f"{failed} download(s) failed: {failed_files}")

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the NYC TLC downloader."""
    parser = argparse.ArgumentParser(
        description="Download NYC TLC Yellow/Green trip Parquet files and taxi zone lookup CSV."
    )
    parser.add_argument(
        "--mode",
        choices=("sample", "full"),
        default="sample",
        help="Download mode: 'sample' (2024-01 only) or 'full' (all months in 2023-2024).",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Explicit list of months in YYYY-MM format (overrides --mode).",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=VALID_SERVICES,
        default=list(VALID_SERVICES),
        help="Taxi service types to download (default: yellow green).",
    )
    parser.add_argument(
        "--skip-zones",
        action="store_true",
        help="Skip downloading taxi_zone_lookup.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist and are valid.",
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
    run_download(
        mode=args.mode,
        months=args.months,
        services=args.services,
        include_zones=not args.skip_zones,
        force=args.force,
    )


if __name__ == "__main__":
    main()

