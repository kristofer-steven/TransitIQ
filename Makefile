.PHONY: setup download-sample download-full download-weather generate-dims verify-raw lint test

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

download-sample:
	$(PYTHON) -m data.ingestion.download_tlc --mode sample
	$(PYTHON) -m data.ingestion.download_weather
	$(PYTHON) -m data.ingestion.generate_dimensions

download-full:
	$(PYTHON) -m data.ingestion.download_tlc --mode full
	$(PYTHON) -m data.ingestion.download_weather
	$(PYTHON) -m data.ingestion.generate_dimensions

download-weather:
	$(PYTHON) -m data.ingestion.download_weather

generate-dims:
	$(PYTHON) -m data.ingestion.generate_dimensions

verify-raw:
	$(PYTHON) -m data.ingestion.verify_raw

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -v

