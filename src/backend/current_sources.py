'''Find recent originals that match the subject of a current-state question.'''

import re
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
EXPLICIT_PERIOD = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:[-_.]?\d{2}){0,2}(?!\d)"
    r"|\b(?:yesterday|tomorrow)\b"
    r"|\b(?:last|next|previous|prior|this)\s+(?:day|week|month|quarter|year)\b"
    rf"|\b(?:as of|on|in|during|for|by|since|until|before|after)\s+(?:{MONTHS})\b",
    re.IGNORECASE,
)
FILENAME_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})(?!\d)")
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
TITLE_PATH_COMPONENTS = 4


@dataclass(frozen=True)
class DatedSource:
    document_id: str
    path: str
    record_date: date
    topics: set[str]
    title_matches: int
    folder_matches: int


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
        "today total update us was we were what when where which who with would you your"
    )
)


# Use dates stated in the filename and ignore invalid calendar dates.
def filename_date(stem: str) -> date | None:
    dates = []
    for match in FILENAME_DATE.finditer(stem):
        try:
            dates.append(date(int(match[1]), int(match[2]), int(match[3])))
        except ValueError:
            continue
    return max(dates, default=None)


# Match a dated record by its title or the directories that describe it.
def match_source(
    document_id: str, path: str, topics: set[str], today: date
) -> DatedSource | None:
    original = PurePosixPath(path.replace("\\", "/"))
    record_date = filename_date(original.stem)
    if record_date is None or record_date > today:
        return None
    if OLDER_RECORD_WORDS & source_terms(original.as_posix()):
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
        document_id, path, record_date, matches, len(title), folder_score
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
