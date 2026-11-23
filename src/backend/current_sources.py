'''Find dated originals that match the subject and periods of a question.'''

import re
from calendar import month_abbr, month_name
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import PurePosixPath

CURRENT_QUESTION = re.compile(
    r"\b(?:current(?:ly)?|latest|newest|most recent|today|now|at present)\b",
    re.IGNORECASE,
)
PRESENT_QUANTITY = re.compile(
    r"\bdo (?:we|i|you|they) have\b"
    r"|\bhow (?:many|much)\b[^?!.;]{0,120}\b(?:is|are|has|have|do|does)\b",
    re.IGNORECASE,
)
PAST_QUESTION = re.compile(r"\b(?:did|was|were|had)\b", re.IGNORECASE)
MONTHS = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    "aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
MONTH_HINT = {
    alias.lower(): month
    for month in range(1, 13)
    for alias in (month_name[month], month_abbr[month])
}
MONTH_HINT["sept"] = 9
EXPLICIT_PERIOD = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:[-_.]?\d{2}){0,2}(?!\d)"
    r"|\b(?:yesterday|tomorrow)\b"
    r"|\b(?:last|next|previous|prior|this)\s+(?:day|week|month|quarter|year)\b"
    rf"|\b(?:as of|on|in|during|for|by|since|until|before|after)\s+(?:{MONTHS})\b",
    re.IGNORECASE,
)
FILENAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})(?!\d)")
MONTH_FIRST_DATE = re.compile(
    r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-]((?:19|20)\d{2})(?!\d)"
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_QUALIFIER = re.compile(
    r"\s+(?:in|for|during)\s+(?:19|20)\d{2}[?.!]*$", re.IGNORECASE
)
YEAR_CONTEXT = re.compile(r"\b(?:in|for|during|from)\s+(?:19|20)\d{2}\b", re.IGNORECASE)
YEAR_FOLLOW_UP = re.compile(
    r"(?:(?:and|what about|how about)\s+)?(?:(?:in|for|during)\s+)?(?P<year>(?:19|20)\d{2})[?.!]*",
    re.IGNORECASE,
)
RECORD_KIND = re.compile(
    r"minutes|report|roster|register|statement|ledger|summary", re.IGNORECASE
)
TITLE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
WORD = re.compile(r"[^\W\d_]+")
OLDER_RECORD_WORDS = frozenset(
    {
        "agenda",
        "archive",
        "archived",
        "draft",
        "historical",
        "past",
        "superseded",
        "template",
    }
)
MAX_CURRENT_DOCUMENTS = 4
MAX_COMPARISON_DOCUMENTS = 9
MAX_COMPARISON_YEARS = 4
TITLE_PATH_COMPONENTS = 4


@dataclass(frozen=True)
class DatedSource:
    document_id: str
    path: str
    record_date: date
    topics: set[str]
    title_matches: int
    folder_matches: int
    month_known: bool = True
    day_known: bool = True


# Treat common singular and plural forms as the same search term.
def source_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for word in WORD.findall(text.casefold()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            word = word[:-1]
        terms.add(word)
    return terms


QUESTION_STOP_WORDS = frozenset(
    source_terms(
        "a about an and are as at be by can could count current currently did do does "
        "for from had has have how i in is it latest many me most much new newest now "
        "number of on or our present recent report show the their there they this to "
        "today total update us was we were what when where which who with would you your "
        "all compare comparison difference differ list name versus vs between"
    )
)


# Use dates stated in the filename and ignore invalid calendar dates.
def filename_date(stem: str) -> date | None:
    dates = []
    parts = chain_dates(stem)
    for year, month, day in parts:
        try:
            dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return max(dates, default=None)


# Read both year-first and separated month-first dates from filenames.
def chain_dates(stem: str) -> Iterable[tuple[str, str, str]]:
    yield from ((m[1], m[2], m[3]) for m in FILENAME_DATE.finditer(stem))
    yield from ((m[3], m[1], m[2]) for m in MONTH_FIRST_DATE.finditer(stem))


# Match a dated record by its title or the directories that describe it.
def match_source(
    document_id: str,
    path: str,
    topics: set[str],
    today: date,
    *,
    year_only: bool = False,
) -> DatedSource | None:
    original = PurePosixPath(path.replace("\\", "/"))
    record_date = filename_date(original.stem)
    day_known = month_known = record_date is not None
    if record_date is None and year_only:
        parts = (original.stem, *reversed(original.parts[:-1]))
        for index, part in enumerate(parts):
            if len(years := YEAR.findall(part)) == 1:
                month = next(
                    (
                        MONTH_HINT[word]
                        for section in parts[: index + 1]
                        for word in WORD.findall(section.casefold())
                        if word in MONTH_HINT
                    ),
                    None,
                )
                month_known = month is not None
                record_date = date(int(years[0]), month or 1, 1)
                break
    if record_date is None or record_date > today:
        return None
    if OLDER_RECORD_WORDS & source_terms(TITLE_BOUNDARY.sub(" ", original.as_posix())):
        return None
    title = topics & source_terms(original.stem)
    folders = [
        topics & source_terms(folder)
        for folder in original.parts[-TITLE_PATH_COMPONENTS:-1]
    ]
    matches = title.union(*folders)
    if not matches:
        return None
    folder_score = sum(
        proximity * len(terms) for proximity, terms in enumerate(folders, 1)
    )
    return DatedSource(
        document_id,
        path,
        record_date,
        matches,
        len(title),
        folder_score,
        month_known,
        day_known,
    )


# Rank already relevant records by date, then by where their subject matched.
def source_order(source: DatedSource) -> tuple[int, int, int, int, str, str]:
    return (
        -source.record_date.toordinal(),
        -len(source.topics),
        -source.title_matches,
        -source.folder_matches,
        source.path.casefold(),
        source.document_id,
    )


# Reserve places for distinct topics, then fill from the newest remaining records.
def latest_document_ids(
    question: str, documents: Iterable[tuple[str, str]], *, today: date | None = None
) -> list[str]:
    current = CURRENT_QUESTION.search(question) or (
        PRESENT_QUANTITY.search(question) and not PAST_QUESTION.search(question)
    )
    if not current or EXPLICIT_PERIOD.search(question):
        return []
    topics = source_terms(question) - QUESTION_STOP_WORDS
    if not topics:
        return []

    as_of = today or datetime.now(UTC).date()
    sources = [
        source
        for document_id, path in documents
        if (source := match_source(document_id, path, topics, as_of)) is not None
    ]
    sources.sort(key=source_order)
    selected: dict[str, None] = {}
    uncovered = topics.copy()
    for source in sources:
        if source.document_id not in selected and source.topics & uncovered:
            selected[source.document_id] = None
            uncovered -= source.topics
            if len(selected) == MAX_CURRENT_DOCUMENTS or not uncovered:
                break
    for source in sources:
        if len(selected) == MAX_CURRENT_DOCUMENTS:
            break
        selected[source.document_id] = None
    return list(selected)


# Identify explicit year scopes without interpreting incidental dates in file titles.
def requested_years(question: str) -> tuple[int, ...]:
    years = tuple(dict.fromkeys(int(match[0]) for match in YEAR.finditer(question)))
    if len(years) == 1 and YEAR_CONTEXT.search(question):
        return years
    return years if 2 <= len(years) <= MAX_COMPARISON_YEARS else ()


# Group records by the directory they share before their dated subfolders.
def source_series(path: str, year: int) -> tuple[str, ...]:
    original = PurePosixPath(path.replace("\\", "/"))
    folders = original.parts[:-1]
    kind = RECORD_KIND.search(original.stem)
    for index, folder in enumerate(folders):
        if str(year) in YEAR.findall(folder):
            folders = folders[:index]
            break
    return (*folders, kind[0].casefold() if kind else "")


# Keep a comparable record series when available and cover each year's timeline.
def dated_document_ids(
    question: str,
    documents: Iterable[tuple[str, str]],
    *,
    today: date | None = None,
    exact_dates_only: bool = False,
) -> dict[int, list[str]]:
    years = requested_years(question)
    if not years:
        return {}
    topics = source_terms(question) - QUESTION_STOP_WORDS
    as_of = today or datetime.now(UTC).date()
    periods: dict[int, list[DatedSource]] = {year: [] for year in years}
    series: dict[tuple[str, ...], list[DatedSource]] = defaultdict(list)
    for document_id, path in documents:
        source = match_source(document_id, path, topics, as_of, year_only=True)
        if (
            source is not None
            and source.record_date.year in periods
            and (source.day_known or not exact_dates_only)
        ):
            periods[source.record_date.year].append(source)
            series[source_series(path, source.record_date.year)].append(source)

    recurring = []
    for records in series.values():
        dates = {record.record_date for record in records if record.day_known}
        if all(sum(day.year == year for day in dates) >= 2 for year in years):
            recurring.append((len(dates), records))
    if recurring:
        _, chosen = max(
            recurring,
            key=lambda group: (
                max(len(record.topics) for record in group[1]),
                group[0],
            ),
        )
        periods = {
            year: [record for record in chosen if record.record_date.year == year]
            for year in years
        }

    return {year: representative_ids(records) for year, records in periods.items()}


# Reserve one record per month and sample across the full year when a series is large.
def representative_ids(records: list[DatedSource]) -> list[str]:
    monthly: dict[int, DatedSource] = {}
    undated = []
    for record in sorted(
        records,
        key=lambda item: (
            -len(item.topics),
            -item.title_matches,
            -item.folder_matches,
            *source_order(item),
        ),
    ):
        if record.month_known:
            monthly.setdefault(record.record_date.month, record)
        else:
            undated.append(record)
    ordered = sorted(monthly.values(), key=lambda record: record.record_date)
    if len(ordered) > MAX_COMPARISON_DOCUMENTS:
        last = MAX_COMPARISON_DOCUMENTS - 1
        ordered = [
            ordered[round(i * (len(ordered) - 1) / last)] for i in range(last + 1)
        ]
    return [
        record.document_id for record in [*ordered, *undated][:MAX_COMPARISON_DOCUMENTS]
    ]
