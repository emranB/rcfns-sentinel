'''Check profit-and-loss totals from the original report columns.'''

import re
from calendar import month_abbr, monthrange
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from html import unescape
from pathlib import PurePosixPath

from src.backend.current_sources import MONTH_HINT, MONTHS, YEAR, requested_years

STATEMENT_TITLE = re.compile(
    r"(?:profit\s*(?:and|&|/)?\s*loss|income statement|statement of operations)"
    r"(?:\s+by\s+class)?",
    re.IGNORECASE,
)
STATEMENT_TOPIC = re.compile(
    rf"\b(?:{STATEMENT_TITLE.pattern}s?|p\s*[/&]\s*l)\b", re.IGNORECASE
)
CHECK_REQUEST = re.compile(
    r"\b(?:add(?:s|ed)? up|reconcil\w*|check|verif\w*|audit|"
    r"make sense of|tie(?:s)? out|sum correctly|arithmetic)\b",
    re.IGNORECASE,
)
TITLE_HINT = re.compile(
    r"\bprofi(?:t|[\s:_-]+loss)|income[ _-]+statement|statement[ _-]+of[ _-]+operations|"
    r"\bp\s*(?:[/&:_-]\s*)?l\b",
    re.IGNORECASE,
)
MONTH_WORD = re.compile(rf"\b(?:{MONTHS})\b", re.IGNORECASE)
PERIOD = re.compile(
    rf"(?P<begin>{MONTHS})(?:\s+(?P<first>\d{{1,2}}))?"
    r"(?:,?\s+(?P<begin_year>(?:19|20)\d{2}))?"
    rf"(?:\s*[-\u2013\u2014]\s*(?:(?P<end>{MONTHS})\s*)?(?P<last>\d{{1,2}})?)?"
    r",?\s*(?P<end_year>(?:19|20)\d{2})?",
    re.IGNORECASE,
)
TOTAL_COLUMN = re.compile(r"(?:^| {2,})total\s*$", re.IGNORECASE)
COMPARISON_COLUMN = re.compile(
    r"\((?:py|pp)\)|\b(?:previous|prior) (?:year|period)\b", re.IGNORECASE
)
AMOUNT = re.compile(r"(?<!\S)\(?(?:-?\$?|\$ ?-)\d[\d,]*\.\d{2}\)?(?!\S)")
WRAPPED_NEGATIVE = re.compile(r"(?<!\S)(?:\$ ?)?-\s*$")
ROW_NAMES = {
    "income": "income",
    "revenue": "income",
    "cost of goods sold": "cost",
    "cost of sales": "cost",
    "gross profit": "gross",
    "expenses": "expenses",
    "operating expenses": "expenses",
    "net operating income": "operating",
    "other income": "other_income",
    "other expenses": "other_expenses",
    "net other income": "other_net",
    "profit": "net",
    "net profit": "net",
    "net income": "net",
    "net loss": "net",
    "net income (loss)": "net",
}
DIRECT_TOTALS = frozenset({"gross", "operating", "other_net", "net"})
REQUIRED_TOTALS = frozenset({"income", "expenses", "net"})


@dataclass(frozen=True)
class Statement:
    start: date | None
    end: date | None
    rows: dict[str, Decimal]
    conflicts: frozenset[str]

    # Check the net equation and any intermediate totals printed in the report.
    def differences(self) -> dict[str, Decimal]:
        income = self.rows["income"]
        expenses = self.rows["expenses"]
        gross = income - self.rows.get("cost", Decimal(0))
        other = self.other_result()
        equations = {
            "gross": gross,
            "operating": gross - expenses,
            "other_net": other,
            "net": gross - expenses + other,
        }
        return {
            key: actual - expected
            for key, expected in equations.items()
            if (actual := self.rows.get(key)) is not None and actual != expected
        }

    # Combine other income and expenses, including negative adjustments.
    def other_result(self) -> Decimal:
        return self.rows.get("other_income", Decimal(0)) - self.rows.get(
            "other_expenses", Decimal(0)
        )

    # Combine cost of sales and the other result with the operating result.
    def adjustment(self) -> Decimal:
        return self.other_result() - self.rows.get("cost", Decimal(0))


# Recognize requests to reconcile a dated family of statements.
def statement_years(question: str) -> tuple[int, ...]:
    if MONTH_WORD.search(question) or not (
        STATEMENT_TOPIC.search(question) and CHECK_REQUEST.search(question)
    ):
        return ()
    return requested_years(question) or tuple(
        dict.fromkeys(int(year) for year in YEAR.findall(question))
    )


# Find original PDFs with a matching report title or abbreviation, regardless of folder.
def statement_candidates(documents: Iterable[tuple[str, str]]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for key, path in documents:
        original = PurePosixPath(path.replace("\\", "/"))
        title = unescape(original.stem)
        if original.suffix.casefold() == ".pdf" and TITLE_HINT.search(title):
            candidates[key] = path
    return candidates


# Read the printed reporting range, including ranges that omit a repeated month or year.
def statement_period(lines: list[str]) -> tuple[date | None, date | None]:
    for line in lines:
        match = PERIOD.fullmatch(line.strip())
        if match is None or not (year := match["end_year"] or match["begin_year"]):
            continue
        first_month = MONTH_HINT[match["begin"].casefold()]
        last_month = MONTH_HINT[(match["end"] or match["begin"]).casefold()]
        last_year = int(year)
        first_year = int(match["begin_year"] or year)
        if last_month < first_month and not match["begin_year"]:
            first_year -= 1
        try:
            start = date(first_year, first_month, int(match["first"] or 1))
            end = date(
                last_year,
                last_month,
                int(match["last"] or monthrange(last_year, last_month)[1]),
            )
        except ValueError:
            continue
        if start <= end:
            return start, end
    return None, None


# Take only exact summary labels whose figures align with the grand total column.
def parse_statement(text: str) -> Statement | None:
    pages = [page.splitlines() for page in text.split("\f") if page.strip()]
    if not pages:
        return None
    heading = [line.strip() for line in pages[0] if line.strip()][:8]
    if not any(STATEMENT_TITLE.fullmatch(line) for line in heading):
        return None
    rows: dict[str, Decimal] = {}
    conflicts: set[str] = set()
    for lines in pages:
        if (column := statement_column(lines)) is None:
            continue
        for index, line in enumerate(lines):
            label = line.lstrip().partition("  ")[0].casefold()
            total = label.startswith("total ")
            key = ROW_NAMES.get(label.removeprefix("total ").removeprefix("for "))
            if key is None or (not total and key not in DIRECT_TOTALS):
                continue
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if (value := statement_amount(line, following, column)) is None:
                continue
            if label == "net loss":
                value = -abs(value)
            if key in rows and rows[key] != value:
                conflicts.add(key)
            rows[key] = value
    start, end = statement_period(heading)
    return Statement(start, end, rows, frozenset(conflicts))


# Locate the current-period column when a total is split into current and prior figures.
def statement_column(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if not TOTAL_COLUMN.search(line):
            continue
        following = next(
            (row.rstrip() for row in lines[index + 1 : index + 4] if row.strip()), ""
        )
        if COMPARISON_COLUMN.search(following) and (
            current := re.search(r"\S.*?(?= {2,}|$)", following)
        ):
            return current.end()
        return len(line.rstrip())
    return None


# Read the amount aligned with the selected column, including a wrapped negative value.
def statement_amount(line: str, following: str, column: int) -> Decimal | None:
    if (
        match := min(
            AMOUNT.finditer(line),
            key=lambda item: abs(item.end() - column),
            default=None,
        )
    ) and abs(match.end() - column) <= 4:
        raw = match[0]
        value = Decimal(
            raw.strip("()").replace("$", "").replace(",", "").replace(" ", "")
        )
        return -abs(value) if raw.startswith("(") else value
    if (
        WRAPPED_NEGATIVE.search(line)
        and abs(len(line.rstrip()) - column) <= 4
        and abs(len(following.rstrip()) - column) <= 4
        and re.fullmatch(r"\d[\d,]*\.\d{2}", following.strip())
    ):
        return -Decimal(following.strip().replace(",", ""))
    return None


# Label whole months compactly and preserve exact partial-month or cross-year dates.
def period_label(start: date, end: date) -> str:
    if start.day == 1 and end.day == monthrange(end.year, end.month)[1]:
        first = (
            f"{month_abbr[start.month]} {start.year}"
            if start.year != end.year
            else month_abbr[start.month]
        )
        last = f"{month_abbr[end.month]} {end.year}"
        return (
            last
            if start.year == end.year and start.month == end.month
            else f"{first} to {last}"
        )
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
