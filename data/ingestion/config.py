"""Central configuration for TransitIQ data ingestion."""

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Settings for raw data download, reference generation, and storage paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"), validation_alias="TRANSITIQ_DATA_DIR")
    raw_dir: Path = Field(default=Path("data/raw"), validation_alias="TRANSITIQ_RAW_DIR")
    processed_dir: Path = Field(
        default=Path("data/processed"),
        validation_alias="TRANSITIQ_PROCESSED_DIR",
    )
    duckdb_path: Path = Field(
        default=Path("data/processed/transitiq.duckdb"),
        validation_alias="TRANSITIQ_DUCKDB_PATH",
    )

    as_of_date: date = Field(
        default=date(2024, 12, 31),
        validation_alias="TRANSITIQ_AS_OF_DATE",
    )
    tlc_start_year: int = Field(default=2023, validation_alias="TLC_START_YEAR")
    tlc_end_year: int = Field(default=2024, validation_alias="TLC_END_YEAR")
    noaa_station_id: str = Field(default="USW00094728", validation_alias="NOAA_STATION_ID")

    tlc_base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    tlc_zone_lookup_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    noaa_base_url: str = (
        "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
    )

    @property
    def yellow_raw_dir(self) -> Path:
        return self.raw_dir / "yellow"

    @property
    def green_raw_dir(self) -> Path:
        return self.raw_dir / "green"

    @property
    def reference_raw_dir(self) -> Path:
        return self.raw_dir / "reference"

    @property
    def weather_raw_dir(self) -> Path:
        return self.raw_dir / "weather"

    def tlc_parquet_url(self, service_type: str, year_month: str) -> str:
        """Return the CloudFront URL for a given service ('yellow'|'green') and 'YYYY-MM'."""
        return f"{self.tlc_base_url}/{service_type}_tripdata_{year_month}.parquet"

    def noaa_station_url(self) -> str:
        """Return the NOAA GHCN-Daily CSV URL for the configured station."""
        return f"{self.noaa_base_url}/{self.noaa_station_id}.csv"


settings = IngestionSettings()

