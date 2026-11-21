"""Deterministic relative-date resolution against the dataset as-of date (§7.5 PRD)."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_AS_OF_DATE = date(2024, 12, 31)
DATASET_START_YEAR = 2023
DATASET_END_YEAR = 2024

WEEKDAY_NAME_TO_ISO: dict[str, int] = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}

LAST_N_DAYS_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+days?\b", re.IGNORECASE)
LAST_WEEKDAY_RE = re.compile(
    r"\b(?:last|previous)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
EXPLICIT_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class ResolvedDateRange:
    """Resolved inclusive calendar date range and human-readable assumption string."""

    expression: str
    start_date: date
    end_date: date

    @property
    def assumption(self) -> str:
        """Format a clear assumption statement for API/UI explanation display."""
        if self.start_date == self.end_date:
            return f"'{self.expression}' resolved to {self.start_date.isoformat()}"
        return (
            f"'{self.expression}' resolved to "
            f"{self.start_date.isoformat()} through {self.end_date.isoformat()}"
        )


def resolve_relative_expression(
    expression: str,
    as_of_date: date = DEFAULT_AS_OF_DATE,
) -> ResolvedDateRange:
    """Resolve a single relative date expression against `as_of_date`."""
    norm = " ".join(expression.strip().lower().split())

    if norm == "today":
        return ResolvedDateRange(expression=expression, start_date=as_of_date, end_date=as_of_date)

    if norm == "yesterday":
        target = as_of_date - timedelta(days=1)
        return ResolvedDateRange(expression=expression, start_date=target, end_date=target)

    if norm in ("this week", "current week"):
        days_since_monday = as_of_date.isoweekday() - 1
        monday = as_of_date - timedelta(days=days_since_monday)
        return ResolvedDateRange(expression=expression, start_date=monday, end_date=as_of_date)

    if norm in ("last week", "previous week", "past week"):
        days_since_monday = as_of_date.isoweekday() - 1
        current_monday = as_of_date - timedelta(days=days_since_monday)
        prev_monday = current_monday - timedelta(days=7)
        prev_sunday = current_monday - timedelta(days=1)
        return ResolvedDateRange(
            expression=expression,
            start_date=prev_monday,
            end_date=prev_sunday,
        )

    if norm in ("this month", "current month"):
        first_day = date(as_of_date.year, as_of_date.month, 1)
        return ResolvedDateRange(expression=expression, start_date=first_day, end_date=as_of_date)

    if norm in ("last month", "previous month", "past month"):
        if as_of_date.month == 1:
            prev_year, prev_month = as_of_date.year - 1, 12
        else:
            prev_year, prev_month = as_of_date.year, as_of_date.month - 1
        last_day_num = calendar.monthrange(prev_year, prev_month)[1]
        return ResolvedDateRange(
            expression=expression,
            start_date=date(prev_year, prev_month, 1),
            end_date=date(prev_year, prev_month, last_day_num),
        )

    if norm in ("this year", "current year"):
        return ResolvedDateRange(
            expression=expression,
            start_date=date(as_of_date.year, 1, 1),
            end_date=as_of_date,
        )

    if norm in ("last year", "previous year", "past year"):
        prev_year = as_of_date.year - 1
        return ResolvedDateRange(
            expression=expression,
            start_date=date(prev_year, 1, 1),
            end_date=date(prev_year, 12, 31),
        )

    n_days_match = LAST_N_DAYS_RE.fullmatch(norm)
    if n_days_match:
        n_days = int(n_days_match.group(1))
        if n_days <= 0:
            raise ValueError(f"Number of days must be positive in '{expression}'")
        start_dt = as_of_date - timedelta(days=n_days - 1)
        return ResolvedDateRange(expression=expression, start_date=start_dt, end_date=as_of_date)

    weekday_match = LAST_WEEKDAY_RE.fullmatch(norm)
    if weekday_match:
        target_iso = WEEKDAY_NAME_TO_ISO[weekday_match.group(1).lower()]
        current_iso = as_of_date.isoweekday()
        days_back = (current_iso - target_iso) % 7
        if days_back == 0:
            days_back = 7
        target_dt = as_of_date - timedelta(days=days_back)
        return ResolvedDateRange(expression=expression, start_date=target_dt, end_date=target_dt)

    raise ValueError(f"Unrecognized relative date expression: '{expression}'")


def extract_and_resolve_dates(
    question: str,
    as_of_date: date = DEFAULT_AS_OF_DATE,
) -> list[ResolvedDateRange]:
    """Scan a natural-language question for known relative time expressions and resolve them."""
    lower_q = question.lower()
    resolved: list[ResolvedDateRange] = []

    fixed_phrases = (
        "today",
        "yesterday",
        "this week",
        "last week",
        "previous week",
        "this month",
        "last month",
        "previous month",
        "this year",
        "last year",
        "previous year",
    )
    for phrase in fixed_phrases:
        if re.search(rf"\b{re.escape(phrase)}\b", lower_q):
            resolved.append(resolve_relative_expression(phrase, as_of_date=as_of_date))

    for match in LAST_N_DAYS_RE.finditer(question):
        resolved.append(resolve_relative_expression(match.group(0), as_of_date=as_of_date))

    for match in LAST_WEEKDAY_RE.finditer(question):
        resolved.append(resolve_relative_expression(match.group(0), as_of_date=as_of_date))

    return resolved


def find_out_of_range_years(
    question: str,
    start_year: int = DATASET_START_YEAR,
    end_year: int = DATASET_END_YEAR,
) -> list[int]:
    """Return any four-digit years in `question` outside [start_year, end_year]."""
    years = [int(m.group(1)) for m in EXPLICIT_YEAR_RE.finditer(question)]
    return sorted({y for y in years if y < start_year or y > end_year})
