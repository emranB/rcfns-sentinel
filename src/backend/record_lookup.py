'''Resolve explicit dates and document titles independently of semantic search.'''

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath

from src.backend.current_sources import (
    TITLE_BOUNDARY,
    YEAR,
    filename_date,
    source_terms,
)
from src.backend.named_sources import TITLE_STOP_WORDS, title_words

MONTH_NUMBER = {
    alias: number
    for number in range(1, 13)
    for alias in (
        calendar.month_name[number].lower(),
        calendar.month_abbr[number].lower(),
    )
}
MONTH_NUMBER["sept"] = 9
DATE = re.compile(
    r"\b(?P<month>"
    + "|".join(sorted(MONTH_NUMBER, key=len, reverse=True))
    + r")\.?\s+(?P<day>[0-9]{1,2})(?:st|nd|rd|th)?"
    r"(?:[,\s]+(?P<year>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
DAY_FIRST_DATE = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>"
    + "|".join(sorted(MONTH_NUMBER, key=len, reverse=True))
    + r")\.?[,\s]+(?P<year>(?:19|20)\d{2}|\d{2})\b",
    re.IGNORECASE,
)
RECORD_TYPE = re.compile(
    r"\b(?:balance sheets?|financial statements?|meeting minutes|minutes|"
    r"annual reports?|reports?|policies|policy|registers?|rosters?|ledgers?|budgets?|"
    r"presentations?|spreadsheets?|worksheets?|guidelines?|agendas?)\b",
    re.IGNORECASE,
)
DATE_CONTEXT_LIMIT = 40
MAX_DATE_CANDIDATES = 12
MAX_EXPLICIT_ORIGINALS = 6

FILE_REQUEST = re.compile(
    r"^(?:do (?:you|we) have|is there|where (?:is|are|can (?:i|we) find)|"
    r"(?:can|could) you (?:find|show|open|give)|find|show|open)\s+"
    r"(?:me\s+)?(?:a copy of\s+)?(?:(?:the|a|an|any)\s+)?(?P<target>[^?!.]{2,160})[?!.]*$",
    re.IGNORECASE,
)
FILE_PERIOD = re.compile(
    r"\s+(?:in|for|from|as of)\s+(?:(?P<month>[a-z]+)\s+(?:(?P<day>\d{1,2}),?\s+)?)?"
    r"(?P<year>(?:19|20)\d{2})$",
    re.IGNORECASE,
)
LEADING_FILE_YEAR = re.compile(r"^(?P<year>(?:19|20)\d{2})\s+(?P<title>.+)$")
FILE_REFERENCES = re.compile(
    r"\b(?:on|in|under|from|according to)\s+(?:(?:the|our)\s+)?", re.IGNORECASE
)
REPORT_PERIOD_LABEL = re.compile(
    r"\b(?:as of|(?:period|year|month|quarter)\s+end(?:ing|ed)?|reporting date)\s*[:-]?\s*",
    re.IGNORECASE,
)
TOTAL_REQUEST = re.compile(
    r"\btotal\s+(?:for\s+)?(?P<label>[^?!\n]+?)(?=\s+(?:on|in|from|according to|for|as of)\b|[?!]|$)",
    re.IGNORECASE,
)
TOTAL_ROW = re.compile(
    r"^\s*total\s+(?:for\s+)?(?P<label>\S.*?)(?:\s{2,}|\s+(?=[$£€]))\s*(?P<value>[$£€]?\s*-?\d[\d,]*\.\d{2})\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatedReference:
    day: date
    kind: str
    exact: tuple[str, ...]
    inspect: tuple[str, ...]


@dataclass(frozen=True)
class FileAvailability:
    title: str
    year: int
    month: int | None = None
    day: int | None = None


# Identify requests for a document itself and keep its period separate from its title.
def file_availability(question: str) -> FileAvailability | None:
    if not (match := FILE_REQUEST.fullmatch(" ".join(question.split()))):
        return None
    target = match["target"].strip()
    if period := FILE_PERIOD.search(target):
        month = MONTH_NUMBER.get((period["month"] or "").casefold())
        if period["month"] and month is None:
            return None
        title = target[: period.start()].strip()
        year = int(period["year"])
        day = int(period["day"]) if period["day"] else None
        return FileAvailability(title, year, month, day) if title else None
    if leading := LEADING_FILE_YEAR.fullmatch(target):
        return FileAvailability(leading["title"], int(leading["year"]))
    return None


# Recognize a document reference inside a factual question, keeping its requested period.
def referenced_file(question: str) -> FileAvailability | None:
    for boundary in FILE_REFERENCES.finditer(question):
        target = question[boundary.end() :]
        if target and (reference := file_availability(f"Find {target}")):
            return reference
    return None


# Prefer the reporting date in the original over a generated or printed date.
def document_period(
    title: str, heading: str, *, year_hint: int | None = None
) -> date | None:
    title_years = [int(token) for token in title_tokens(title) if YEAR.fullmatch(token)]
    reference = title_years[0] if len(title_years) == 1 else year_hint

    # Interpret a short year only when the original's title or folder supplies the century.
    def day_first(match: re.Match[str]) -> date | None:
        year = int(match["year"])
        if len(match["year"]) == 2:
            if reference is None:
                return None
            year += reference // 100 * 100
        try:
            return date(
                year, MONTH_NUMBER[match["month"].casefold()], int(match["day"])
            )
        except ValueError:
            return None

    for label in REPORT_PERIOD_LABEL.finditer(heading):
        if match := DATE.match(heading, label.end()):
            dated = mentioned_dates(match[0])
            if dated:
                return dated[0][0]
        if (legacy := DAY_FIRST_DATE.match(heading, label.end())) and (
            day := day_first(legacy)
        ):
            return day
    for line in heading.splitlines()[:6]:
        if (legacy := DAY_FIRST_DATE.fullmatch(line.strip())) and (
            day := day_first(legacy)
        ):
            return day
    dates = mentioned_dates(title)
    return dates[0][0] if dates else filename_date(title)


# Match an unambiguous single-value total to its full printed row label.
def exact_record_total(question: str, text: str) -> str:
    if not (request := TOTAL_REQUEST.search(question)):
        return ""
    label = source_terms(request["label"])
    if not label:
        return ""
    lines = {
        " ".join(line.split())
        for line in text.splitlines()
        if (row := TOTAL_ROW.fullmatch(line)) and source_terms(row["label"]) == label
    }
    return next(iter(lines)) if len(lines) == 1 else ""


# Split filename identifiers at word, case and digit boundaries.
def title_tokens(value: str) -> list[str]:
    split = TITLE_BOUNDARY.sub(" ", value)
    split = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", split)
    return title_words(split)


# Match filenames and identifiers written as a contiguous phrase in a question.
def explicit_ids(
    question: str, originals: Iterable[tuple[str, str]], *, filenames_only: bool = False
) -> list[str]:
    split_words = " " + " ".join(title_tokens(question)) + " "
    without_dates = " " + " ".join(title_tokens(DATE.sub(" ", question))) + " "
    matches = []
    for document_id, path in originals:
        name = PurePosixPath(path.replace("\\", "/"))
        split = title_tokens(name.stem)
        extension = name.suffix.lstrip(".").lower()
        phrase = " ".join(split)
        full_name = f" {phrase} {extension} "
        with_extension = bool(extension and full_name in split_words)
        if filenames_only and not with_extension:
            continue
        significant = [word for word in split if word not in TITLE_STOP_WORDS]
        if without_dates.find(f" {phrase} ") < 0 and not with_extension:
            clean = title_tokens(re.sub(r"\s*\([0-9]+\)$", "", name.stem))
            phrase = " ".join(clean)
            significant = [word for word in clean if word not in TITLE_STOP_WORDS]
        if not with_extension and (
            len(significant) < 3
            or f" {phrase} " not in without_dates
            or f" {phrase} " not in split_words
        ):
            continue
        matched = full_name if with_extension else f" {phrase} "
        # Adjacent filenames share a separator after normalization.
        matches.append(
            (
                split_words.index(matched) + 1,
                2 - len(matched),
                path.casefold(),
                document_id,
            )
        )
    picked: list[str] = []
    covered_until = -1
    for offset, length, _, document_id in sorted(matches):
        if offset < covered_until:
            continue
        picked.append(document_id)
        covered_until = offset + abs(length)
        if len(picked) == MAX_EXPLICIT_ORIGINALS:
            break
    return picked


# Read natural dates and dated lists without assigning a year across distant topics.
def mentioned_dates(value: str) -> list[tuple[date, int, int]]:
    matches = list(DATE.finditer(value))
    results = []
    for index, match in enumerate(matches):
        year = match["year"]
        if year is None:
            for later in matches[index + 1 :]:
                between = value[match.end() : later.start()]
                if len(between) > DATE_CONTEXT_LIMIT or re.search(r"[.!?;]", between):
                    break
                if later["year"]:
                    year = later["year"]
                    break
        if year is None:
            continue
        try:
            day = date(
                int(year), MONTH_NUMBER[match["month"].lower()], int(match["day"])
            )
        except ValueError:
            continue
        results.append((day, match.start(), match.end()))
    return results


# Recognize a document date from its actual heading when its filename is undated.
def has_record_date(text: str, day: date) -> bool:
    heading = "\n".join(text.splitlines()[:10])
    dates = mentioned_dates(heading)
    return dates[0][0] == day if dates else filename_date(heading[:340]) == day


# Match each requested date to its nearby document type and shortlist originals.
def dated_references(
    question: str, originals: Iterable[tuple[str, str]]
) -> list[DatedReference]:
    dates = mentioned_dates(question)
    kinds = list(RECORD_TYPE.finditer(question))
    if not dates or not kinds:
        return []
    mentions: dict[tuple[date, str], set[str]] = {}
    for day, start, end in dates:
        match = min(
            kinds,
            key=lambda k: min(abs(k.end() - start) + 16, abs(k.start() - end)),
        )
        if min(abs(match.end() - start), abs(match.start() - end)) > 85:
            continue
        between = question[min(match.start(), start) : max(match.end(), end)]
        kind = " ".join(title_words(match[0])).removeprefix("meeting ")
        if kind.endswith("s") and kind != "minutes":
            kind = kind.removesuffix("s")
        qualifiers = {
            word
            for word in title_words(between)
            if word.isalpha()
            and word not in MONTH_NUMBER
            and word not in TITLE_STOP_WORDS
            and word not in kind.split()
        }
        mentions.setdefault((day, kind), qualifiers)

    records = []
    for document_id, path in originals:
        original = PurePosixPath(path.replace("\\", "/"))
        stem = original.stem
        title = set(title_tokens(stem))
        folders = set(title_tokens("/".join(original.parts[-4:-1])))
        internal_dates = mentioned_dates(stem)
        stamp = filename_date(stem) or (
            internal_dates[0][0] if internal_dates else None
        )
        records.append((document_id, path, title, folders, stamp))

    results = []
    for (day, kind), qualifiers in list(mentions.items())[:MAX_EXPLICIT_ORIGINALS]:
        matched = []
        required = set(kind.split())
        for document_id, path, title, folders, stamp in records:
            if not required <= title or (stamp is not None and stamp != day):
                continue
            if {
                "draft",
                "template",
                "agenda",
            } & title and not required & title.intersection(
                {"draft", "template", "agenda"}
            ):
                continue
            years = {int(year) for year in YEAR.findall(path)}
            if stamp is None and years and day.year not in years:
                continue
            order = (
                stamp != day,
                -len(qualifiers & (title | folders)),
                day.year not in years,
                len(title - required),
                path.casefold(),
            )
            matched.append((order, document_id))
        matched.sort()
        exact = tuple(document_id for order, document_id in matched if not order[0])
        inspect = tuple(document_id for order, document_id in matched if order[0])
        results.append(DatedReference(day, kind, exact, inspect[:MAX_DATE_CANDIDATES]))
    return results
