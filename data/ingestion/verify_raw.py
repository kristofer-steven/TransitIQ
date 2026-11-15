"""Raw data inventory, SHA-256 manifest builder, and Parquet schema-drift inspector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.ingestion.config import IngestionSettings, settings

logger = logging.getLogger(__name__)

TRIP_FILENAME_RE = re.compile(r"^(yellow|green)_tripdata_(\d{4}-\d{2})\.parquet$")
HASH_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_parquet_file(
    path: Path,
    base_dir: Path | None = None,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Read Parquet footer metadata and schema without loading row data into memory."""
    metadata = pq.read_metadata(path)
    arrow_schema = pq.read_schema(path)

    match = TRIP_FILENAME_RE.match(path.name)
    service_type = match.group(1) if match else "unknown"
    year_month = match.group(2) if match else "unknown"

    columns = [
        {"name": field.name, "type": str(field.type)}
        for field in arrow_schema
    ]

    rel_path = path.relative_to(base_dir).as_posix() if base_dir else path.name
    return {
        "file_name": path.name,
        "relative_path": rel_path,
        "service_type": service_type,
        "year_month": year_month,
        "file_size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path) if compute_hash else None,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "columns": columns,
    }


def detect_schema_drift(file_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect column name casing, type differences, or missing/extra columns across files.

    Groups files by `service_type` and compares each subsequent file against the
    earliest file (by `year_month` / `file_name`) for that service.
    """
    by_service: dict[str, list[dict[str, Any]]] = {}
    for entry in file_entries:
        service = entry.get("service_type", "unknown")
        by_service.setdefault(service, []).append(entry)

    drifts: list[dict[str, Any]] = []

    for service, entries in sorted(by_service.items()):
        sorted_entries = sorted(
            entries,
            key=lambda e: (e.get("year_month", ""), e.get("file_name", "")),
        )
        if len(sorted_entries) < 2:
            continue

        baseline = sorted_entries[0]
        base_cols: dict[str, str] = {
            c["name"]: c["type"] for c in baseline.get("columns", [])
        }
        base_lower_to_exact: dict[str, str] = {
            name.lower(): name for name in base_cols
        }

        for candidate in sorted_entries[1:]:
            cand_cols: dict[str, str] = {
                c["name"]: c["type"] for c in candidate.get("columns", [])
            }
            cand_lower_to_exact: dict[str, str] = {
                name.lower(): name for name in cand_cols
            }

            # 1. Casing mismatch (same lowercase name, different exact name)
            for lower_name, base_exact in base_lower_to_exact.items():
                if lower_name in cand_lower_to_exact:
                    cand_exact = cand_lower_to_exact[lower_name]
                    if base_exact != cand_exact:
                        drifts.append(
                            {
                                "service_type": service,
                                "baseline_file": baseline["file_name"],
                                "candidate_file": candidate["file_name"],
                                "drift_type": "case_mismatch",
                                "baseline_column": base_exact,
                                "candidate_column": cand_exact,
                                "details": (
                                    f"Column casing changed: '{base_exact}' -> '{cand_exact}'"
                                ),
                            }
                        )

            # 2. Type mismatch (comparing matched lowercase columns)
            for lower_name, base_exact in base_lower_to_exact.items():
                if lower_name in cand_lower_to_exact:
                    cand_exact = cand_lower_to_exact[lower_name]
                    base_type = base_cols[base_exact]
                    cand_type = cand_cols[cand_exact]
                    if base_type != cand_type:
                        drifts.append(
                            {
                                "service_type": service,
                                "baseline_file": baseline["file_name"],
                                "candidate_file": candidate["file_name"],
                                "drift_type": "type_mismatch",
                                "baseline_column": base_exact,
                                "candidate_column": cand_exact,
                                "baseline_type": base_type,
                                "candidate_type": cand_type,
                                "details": (
                                    f"Type changed for '{cand_exact}': "
                                    f"{base_type} -> {cand_type}"
                                ),
                            }
                        )

            # 3. Missing or extra columns (by lowercase name)
            missing_lower = sorted(set(base_lower_to_exact) - set(cand_lower_to_exact))
            extra_lower = sorted(set(cand_lower_to_exact) - set(base_lower_to_exact))

            for lower_name in missing_lower:
                col_name = base_lower_to_exact[lower_name]
                drifts.append(
                    {
                        "service_type": service,
                        "baseline_file": baseline["file_name"],
                        "candidate_file": candidate["file_name"],
                        "drift_type": "missing_column",
                        "baseline_column": col_name,
                        "candidate_column": None,
                        "details": f"Column '{col_name}' missing in {candidate['file_name']}",
                    }
                )

            for lower_name in extra_lower:
                col_name = cand_lower_to_exact[lower_name]
                drifts.append(
                    {
                        "service_type": service,
                        "baseline_file": baseline["file_name"],
                        "candidate_file": candidate["file_name"],
                        "drift_type": "extra_column",
                        "baseline_column": None,
                        "candidate_column": col_name,
                        "details": f"Extra column '{col_name}' present in {candidate['file_name']}",
                    }
                )

    return drifts


def _inspect_reference_file(
    path: Path,
    expected_rows: int | None,
    base_dir: Path,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Inspect a supplementary Parquet or CSV reference file."""
    rel_path = path.relative_to(base_dir).as_posix()
    if not path.is_file():
        return {
            "file_name": path.name,
            "relative_path": rel_path,
            "exists": False,
            "num_rows": 0,
            "expected_rows": expected_rows,
            "status": "missing",
        }

    size_bytes = path.stat().st_size
    if path.suffix == ".parquet":
        meta = pq.read_metadata(path)
        num_rows = meta.num_rows
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            num_rows = max(sum(1 for _ in reader) - 1, 0)
    else:
        num_rows = 0

    rows_ok = expected_rows is None or num_rows == expected_rows
    status = "ok" if (size_bytes > 0 and rows_ok) else "row_count_mismatch"

    return {
        "file_name": path.name,
        "relative_path": rel_path,
        "exists": True,
        "file_size_bytes": size_bytes,
        "sha256": compute_sha256(path) if compute_hash else None,
        "num_rows": num_rows,
        "expected_rows": expected_rows,
        "status": status,
    }


def build_manifest(
    cfg: IngestionSettings = settings,
    compute_hash: bool = True,
) -> dict[str, Any]:
    """Scan all raw trip, weather, and reference files and build the raw data manifest."""
    trip_files: list[dict[str, Any]] = []
    for folder in (cfg.yellow_raw_dir, cfg.green_raw_dir):
        if not folder.is_dir():
            continue
        for parquet_path in sorted(folder.glob("*.parquet")):
            entry = inspect_parquet_file(
                parquet_path,
                base_dir=cfg.raw_dir,
                compute_hash=compute_hash,
            )
            trip_files.append(entry)

    drifts = detect_schema_drift(trip_files)

    expected_calendar_days = (
        datetime(cfg.tlc_end_year, 12, 31) - datetime(cfg.tlc_start_year, 1, 1)
    ).days + 1
    weather_stem = f"central_park_weather_{cfg.tlc_start_year}_{cfg.tlc_end_year}.parquet"

    reference_targets: list[tuple[Path, int | None]] = [
        (cfg.reference_raw_dir / "taxi_zone_lookup.csv", 265),
        (cfg.weather_raw_dir / weather_stem, expected_calendar_days),
        (cfg.reference_raw_dir / "dim_date.parquet", expected_calendar_days),
        (cfg.reference_raw_dir / "dim_payment_type.parquet", 7),
        (cfg.reference_raw_dir / "dim_rate_code.parquet", 7),
        (cfg.reference_raw_dir / "dim_vendor.parquet", 4),
    ]

    reference_files = [
        _inspect_reference_file(
            path=target_path,
            expected_rows=exp_rows,
            base_dir=cfg.raw_dir,
            compute_hash=compute_hash,
        )
        for target_path, exp_rows in reference_targets
    ]

    yellow_rows = sum(f["num_rows"] for f in trip_files if f["service_type"] == "yellow")
    green_rows = sum(f["num_rows"] for f in trip_files if f["service_type"] == "green")
    total_trip_bytes = sum(f["file_size_bytes"] for f in trip_files)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "as_of_date": str(cfg.as_of_date),
        "summary": {
            "total_trip_files": len(trip_files),
            "yellow_files": sum(1 for f in trip_files if f["service_type"] == "yellow"),
            "green_files": sum(1 for f in trip_files if f["service_type"] == "green"),
            "yellow_total_rows": yellow_rows,
            "green_total_rows": green_rows,
            "total_trip_rows": yellow_rows + green_rows,
            "total_trip_size_bytes": total_trip_bytes,
            "schema_drift_count": len(drifts),
        },
        "trip_files": trip_files,
        "reference_files": reference_files,
        "schema_drifts": drifts,
    }


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Format the manifest dictionary as a readable Markdown report."""
    summary = manifest["summary"]
    lines: list[str] = [
        "# Raw Data Manifest Summary",
        "",
        f"**Generated At (UTC):** `{manifest['generated_at_utc']}`  ",
        f"**Dataset As-Of Date:** `{manifest['as_of_date']}`",
        "",
        "## 1. Trip Dataset Totals",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Yellow Taxi Files | {summary['yellow_files']} |",
        f"| Green Taxi Files | {summary['green_files']} |",
        f"| Yellow Taxi Rows | {summary['yellow_total_rows']:,} |",
        f"| Green Taxi Rows | {summary['green_total_rows']:,} |",
        f"| **Total Trip Rows** | **{summary['total_trip_rows']:,}** |",
        f"| Total Trip Size (MiB) | {summary['total_trip_size_bytes'] / (1024 * 1024):,.2f} |",
        f"| Schema Drift Warnings | {summary['schema_drift_count']} |",
        "",
        "## 2. Supplementary & Reference Files",
        "",
        "| File | Status | Rows | Expected Rows | Size (Bytes) |",
        "| :--- | :---: | ---: | ---: | ---: |",
    ]

    for ref in manifest["reference_files"]:
        size_val = f"{ref.get('file_size_bytes', 0):,}" if ref.get("exists") else "-"
        exp_val = f"{ref['expected_rows']:,}" if ref["expected_rows"] is not None else "-"
        lines.append(
            f"| `{ref['relative_path']}` | `{ref['status']}` | "
            f"{ref['num_rows']:,} | {exp_val} | {size_val} |"
        )

    lines.extend(
        [
            "",
            "## 3. Monthly Trip Parquet Files",
            "",
            "| File | Service | Month | Rows | Row Groups | Columns | Size (MiB) |",
            "| :--- | :---: | :---: | ---: | ---: | ---: | ---: |",
        ]
    )

    if manifest["trip_files"]:
        for tf in manifest["trip_files"]:
            mib = tf["file_size_bytes"] / (1024 * 1024)
            lines.append(
                f"| `{tf['file_name']}` | {tf['service_type']} | {tf['year_month']} | "
                f"{tf['num_rows']:,} | {tf['num_row_groups']} | {len(tf['columns'])} | {mib:,.2f} |"
            )
    else:
        lines.append("| *(no trip Parquet files found)* | - | - | 0 | 0 | 0 | 0.00 |")

    lines.extend(["", "## 4. Schema Drift Report", ""])
    if manifest["schema_drifts"]:
        lines.extend(
            [
                "| Service | Baseline File | Candidate File | Drift Type | Details |",
                "| :---: | :--- | :--- | :---: | :--- |",
            ]
        )
        for d in manifest["schema_drifts"]:
            lines.append(
                f"| {d['service_type']} | `{d['baseline_file']}` | "
                f"`{d['candidate_file']}` | `{d['drift_type']}` | {d['details']} |"
            )
    else:
        lines.append("No schema drift detected across available monthly Parquet files.")

    lines.append("")
    return "\n".join(lines)


def run_verify_raw(
    cfg: IngestionSettings = settings,
    compute_hash: bool = True,
) -> tuple[Path, Path]:
    """Generate and save manifest.json and MANIFEST_SUMMARY.md in data/raw/."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg=cfg, compute_hash=compute_hash)

    manifest_json_path = cfg.raw_dir / "manifest.json"
    manifest_md_path = cfg.raw_dir / "MANIFEST_SUMMARY.md"

    manifest_json_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_md_path.write_text(
        render_manifest_markdown(manifest),
        encoding="utf-8",
    )

    summary = manifest["summary"]
    logger.info(
        "[DONE] Manifest written: %d trip files (%s rows), %d schema drift(s) -> %s",
        summary["total_trip_files"],
        f"{summary['total_trip_rows']:,}",
        summary["schema_drift_count"],
        manifest_json_path,
    )
    return manifest_json_path, manifest_md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for raw data verification."""
    parser = argparse.ArgumentParser(
        description="Inspect raw Parquet/CSV files and build data/raw/manifest.json."
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA-256 hash calculation for faster metadata-only inspection.",
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
    run_verify_raw(compute_hash=not args.skip_hash)


if __name__ == "__main__":
    main()
