"""Schema normalization and Data Quality (DQ) quarantine engine for NYC TLC trips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

ServiceType = Literal["yellow", "green"]

REJECTION_REASONS: tuple[str, ...] = (
    "null_or_invalid_timestamp",
    "out_of_month_pickup",
    "excessive_duration",
    "invalid_distance",
    "negative_amount",
    "duplicate_trip",
)

VALID_RATE_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 99)
VALID_PAYMENT_TYPES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
VALID_VENDORS: tuple[int, ...] = (1, 2, 6, 7)

DEDUP_KEY_COLUMNS: tuple[str, ...] = (
    "service_type",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance_mi",
    "fare_amount",
    "total_amount",
)

FACT_TRIPS_COLUMNS: tuple[str, ...] = (
    "trip_id",
    "service_type",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_zone_id",
    "dropoff_zone_id",
    "passenger_count",
    "trip_distance_mi",
    "trip_duration_min",
    "rate_code_id",
    "payment_type_id",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "congestion_surcharge",
    "airport_fee",
    "total_amount",
)

QUARANTINE_COLUMNS: tuple[str, ...] = (
    *FACT_TRIPS_COLUMNS,
    "source_file",
    "year_month",
    "rejection_reason",
)


@dataclass(frozen=True)
class DQFileSummary:
    """Reconciliation metrics for a single processed monthly trip file."""

    source_file: str
    service_type: ServiceType
    year_month: str
    raw_rows: int
    clean_rows: int
    quarantine_rows: int
    rejection_counts: dict[str, int]

    @property
    def is_reconciled(self) -> bool:
        """Verify the fundamental DQ invariant: raw == clean + quarantine."""
        return (
            self.raw_rows == self.clean_rows + self.quarantine_rows
            and sum(self.rejection_counts.values()) == self.quarantine_rows
        )


@dataclass(frozen=True)
class DQBatchResult:
    """Output of applying normalization and DQ rules to a monthly trip DataFrame."""

    clean_df: pl.DataFrame
    quarantine_df: pl.DataFrame
    summary: DQFileSummary


def _resolve_column_case_insensitive(columns: list[str], target_lower: str) -> str | None:
    """Find a column name in `columns` matching `target_lower` case-insensitively."""
    for col in columns:
        if col.lower() == target_lower:
            return col
    return None


def _month_bounds(year_month: str) -> tuple[datetime, datetime]:
    """Return inclusive start and exclusive end datetimes for 'YYYY-MM'."""
    parts = year_month.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year_month '{year_month}'; expected 'YYYY-MM'.")
    year, month = int(parts[0]), int(parts[1])
    start_dt = datetime(year, month, 1, 0, 0, 0)
    if month == 12:
        end_dt = datetime(year + 1, 1, 1, 0, 0, 0)
    else:
        end_dt = datetime(year, month + 1, 1, 0, 0, 0)
    return start_dt, end_dt


def normalize_raw_trips(
    raw_df: pl.DataFrame,
    service_type: ServiceType,
    trip_id_offset: int = 0,
) -> pl.DataFrame:
    """Normalize raw Yellow or Green taxi columns to canonical fact_trips schema types."""
    cols = raw_df.columns

    pickup_src = _resolve_column_case_insensitive(
        cols,
        "tpep_pickup_datetime" if service_type == "yellow" else "lpep_pickup_datetime",
    ) or _resolve_column_case_insensitive(cols, "pickup_datetime")
    dropoff_src = _resolve_column_case_insensitive(
        cols,
        "tpep_dropoff_datetime" if service_type == "yellow" else "lpep_dropoff_datetime",
    ) or _resolve_column_case_insensitive(cols, "dropoff_datetime")

    if pickup_src is None or dropoff_src is None:
        raise ValueError(
            f"Missing pickup/dropoff timestamp columns in {service_type} DataFrame: {cols}"
        )

    vendor_src = _resolve_column_case_insensitive(cols, "vendorid") or (
        _resolve_column_case_insensitive(cols, "vendor_id")
    )
    pu_zone_src = _resolve_column_case_insensitive(cols, "pulocationid") or (
        _resolve_column_case_insensitive(cols, "pickup_zone_id")
    )
    do_zone_src = _resolve_column_case_insensitive(cols, "dolocationid") or (
        _resolve_column_case_insensitive(cols, "dropoff_zone_id")
    )
    pax_src = _resolve_column_case_insensitive(cols, "passenger_count")
    dist_src = _resolve_column_case_insensitive(cols, "trip_distance") or (
        _resolve_column_case_insensitive(cols, "trip_distance_mi")
    )
    rate_src = _resolve_column_case_insensitive(cols, "ratecodeid") or (
        _resolve_column_case_insensitive(cols, "rate_code_id")
    )
    pay_src = _resolve_column_case_insensitive(cols, "payment_type") or (
        _resolve_column_case_insensitive(cols, "payment_type_id")
    )
    fare_src = _resolve_column_case_insensitive(cols, "fare_amount")
    tip_src = _resolve_column_case_insensitive(cols, "tip_amount")
    tolls_src = _resolve_column_case_insensitive(cols, "tolls_amount")
    cong_src = _resolve_column_case_insensitive(cols, "congestion_surcharge")
    airport_src = _resolve_column_case_insensitive(cols, "airport_fee")
    total_src = _resolve_column_case_insensitive(cols, "total_amount")

    def _opt_col(src_name: str | None, dtype: pl.DataType, default: object = None) -> pl.Expr:
        if src_name is not None:
            return pl.col(src_name).cast(dtype, strict=False)
        return pl.lit(default, dtype=dtype)

    df = raw_df.with_row_index(name="_row_idx", offset=trip_id_offset + 1)

    pickup_expr = (
        pl.col(pickup_src)
        .cast(pl.Datetime("us"), strict=False)
        .dt.replace_time_zone(None)
    )
    dropoff_expr = (
        pl.col(dropoff_src)
        .cast(pl.Datetime("us"), strict=False)
        .dt.replace_time_zone(None)
    )

    raw_vendor = _opt_col(vendor_src, pl.Int32, default=1)
    raw_pu_zone = _opt_col(pu_zone_src, pl.Int32, default=264)
    raw_do_zone = _opt_col(do_zone_src, pl.Int32, default=264)
    raw_pax = _opt_col(pax_src, pl.Int32, default=None)
    raw_rate = _opt_col(rate_src, pl.Int32, default=99)
    raw_pay = _opt_col(pay_src, pl.Int32, default=5)

    normalized = df.select(
        pl.col("_row_idx").cast(pl.Int64).alias("trip_id"),
        pl.lit(service_type, dtype=pl.String).alias("service_type"),
        pl.when(raw_vendor.is_in(VALID_VENDORS))
        .then(raw_vendor)
        .otherwise(pl.lit(1, dtype=pl.Int32))
        .alias("vendor_id"),
        pickup_expr.alias("pickup_datetime"),
        dropoff_expr.alias("dropoff_datetime"),
        pl.when(raw_pu_zone.is_between(1, 265))
        .then(raw_pu_zone)
        .otherwise(pl.lit(264, dtype=pl.Int32))
        .alias("pickup_zone_id"),
        pl.when(raw_do_zone.is_between(1, 265))
        .then(raw_do_zone)
        .otherwise(pl.lit(264, dtype=pl.Int32))
        .alias("dropoff_zone_id"),
        pl.when(raw_pax.is_not_null() & (raw_pax > 0))
        .then(raw_pax)
        .otherwise(pl.lit(None, dtype=pl.Int32))
        .alias("passenger_count"),
        _opt_col(dist_src, pl.Float64, default=None).alias("trip_distance_mi"),
        pl.lit(None, dtype=pl.Float64).alias("trip_duration_min"),
        pl.when(raw_rate.is_in(VALID_RATE_CODES))
        .then(raw_rate)
        .otherwise(pl.lit(99, dtype=pl.Int32))
        .alias("rate_code_id"),
        pl.when(raw_pay.is_in(VALID_PAYMENT_TYPES))
        .then(raw_pay)
        .otherwise(pl.lit(5, dtype=pl.Int32))
        .alias("payment_type_id"),
        _opt_col(fare_src, pl.Float64, default=None).alias("fare_amount"),
        _opt_col(tip_src, pl.Float64, default=0.0).fill_null(0.0).alias("tip_amount"),
        _opt_col(tolls_src, pl.Float64, default=0.0).fill_null(0.0).alias("tolls_amount"),
        _opt_col(cong_src, pl.Float64, default=0.0).fill_null(0.0).alias("congestion_surcharge"),
        _opt_col(airport_src, pl.Float64, default=0.0).fill_null(0.0).alias("airport_fee"),
        _opt_col(total_src, pl.Float64, default=None).alias("total_amount"),
    )

    # Compute trip_duration_min in minutes from microsecond difference
    normalized = normalized.with_columns(
        (
            (
                pl.col("dropoff_datetime").cast(pl.Int64)
                - pl.col("pickup_datetime").cast(pl.Int64)
            ).cast(pl.Float64)
            / 60_000_000.0
        ).alias("trip_duration_min")
    )
    return normalized


def apply_data_quality_rules(
    raw_df: pl.DataFrame,
    service_type: ServiceType,
    year_month: str,
    source_file: str = "in_memory",
    trip_id_offset: int = 0,
) -> DQBatchResult:
    """Normalize a raw trip DataFrame and partition it into clean and quarantine sets."""
    raw_count = raw_df.height
    month_start, month_end = _month_bounds(year_month)

    norm_df = normalize_raw_trips(
        raw_df=raw_df,
        service_type=service_type,
        trip_id_offset=trip_id_offset,
    )

    cond_invalid_ts = (
        pl.col("pickup_datetime").is_null()
        | pl.col("dropoff_datetime").is_null()
        | (pl.col("dropoff_datetime") <= pl.col("pickup_datetime"))
    )
    cond_out_of_month = (pl.col("pickup_datetime") < pl.lit(month_start)) | (
        pl.col("pickup_datetime") >= pl.lit(month_end)
    )
    cond_excessive_duration = pl.col("trip_duration_min") > 1440.0
    cond_invalid_distance = (
        pl.col("trip_distance_mi").is_null()
        | (pl.col("trip_distance_mi") <= 0.0)
        | (pl.col("trip_distance_mi") > 200.0)
    )
    cond_negative_amount = (
        pl.col("fare_amount").is_null()
        | (pl.col("fare_amount") < 0.0)
        | pl.col("total_amount").is_null()
        | (pl.col("total_amount") < 0.0)
    )

    # Evaluate rules 1-5 in strict priority order
    flagged = norm_df.with_columns(
        pl.when(cond_invalid_ts)
        .then(pl.lit("null_or_invalid_timestamp"))
        .when(cond_out_of_month)
        .then(pl.lit("out_of_month_pickup"))
        .when(cond_excessive_duration)
        .then(pl.lit("excessive_duration"))
        .when(cond_invalid_distance)
        .then(pl.lit("invalid_distance"))
        .when(cond_negative_amount)
        .then(pl.lit("negative_amount"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("rejection_reason")
    )

    # Rule 6: Exact duplicate detection among rows that passed rules 1-5
    passed_rules_1_to_5 = flagged.filter(pl.col("rejection_reason").is_null())
    failed_rules_1_to_5 = flagged.filter(pl.col("rejection_reason").is_not_null())

    passed_with_dup_flag = passed_rules_1_to_5.with_columns(
        pl.struct(DEDUP_KEY_COLUMNS).is_duplicated().alias("_is_dup"),
        pl.struct(DEDUP_KEY_COLUMNS).cum_count().over(DEDUP_KEY_COLUMNS).alias("_dup_rank"),
    )

    clean_df = passed_with_dup_flag.filter(pl.col("_dup_rank") == 1).select(FACT_TRIPS_COLUMNS)
    dup_quarantine = (
        passed_with_dup_flag.filter(pl.col("_dup_rank") > 1)
        .with_columns(pl.lit("duplicate_trip", dtype=pl.String).alias("rejection_reason"))
        .select([*FACT_TRIPS_COLUMNS, "rejection_reason"])
    )

    quarantine_combined = (
        pl.concat(
            [
                failed_rules_1_to_5.select([*FACT_TRIPS_COLUMNS, "rejection_reason"]),
                dup_quarantine,
            ],
            how="vertical",
        )
        .with_columns(
            pl.lit(source_file, dtype=pl.String).alias("source_file"),
            pl.lit(year_month, dtype=pl.String).alias("year_month"),
        )
        .select(QUARANTINE_COLUMNS)
        .sort("trip_id")
    )

    rejection_counts: dict[str, int] = {reason: 0 for reason in REJECTION_REASONS}
    if quarantine_combined.height > 0:
        grouped = quarantine_combined.group_by("rejection_reason").len()
        for row in grouped.iter_rows(named=True):
            reason_key = row["rejection_reason"]
            if reason_key in rejection_counts:
                rejection_counts[reason_key] = int(row["len"])

    summary = DQFileSummary(
        source_file=source_file,
        service_type=service_type,
        year_month=year_month,
        raw_rows=raw_count,
        clean_rows=clean_df.height,
        quarantine_rows=quarantine_combined.height,
        rejection_counts=rejection_counts,
    )

    if not summary.is_reconciled:
        raise RuntimeError(
            f"DQ reconciliation failure for {source_file}: "
            f"raw={summary.raw_rows} != clean={summary.clean_rows} + "
            f"quarantine={summary.quarantine_rows}"
        )

    return DQBatchResult(
        clean_df=clean_df,
        quarantine_df=quarantine_combined,
        summary=summary,
    )


def process_trip_parquet_file(
    parquet_path: Path,
    service_type: ServiceType,
    year_month: str,
    trip_id_offset: int = 0,
) -> DQBatchResult:
    """Read a raw monthly trip Parquet file and apply normalization and DQ rules."""
    raw_df = pl.read_parquet(parquet_path)
    return apply_data_quality_rules(
        raw_df=raw_df,
        service_type=service_type,
        year_month=year_month,
        source_file=parquet_path.name,
        trip_id_offset=trip_id_offset,
    )
