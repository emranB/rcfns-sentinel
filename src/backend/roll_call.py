'''Read names and counts from an unambiguous roll call.'''

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from typing import Literal
from unicodedata import normalize

from src.backend.current_sources import YEAR

WORD = r"[^\W\d_]+"
GROUP = rf"{WORD}(?:[ -]{WORD}){{0,3}}?"
QUESTION = re.compile(
    rf"how many (?:(?P<group>{GROUP}) )?(?:members|directors|trustees) "
    rf"(?:do we (?:currently )?have|are there|"
    rf"are (?:on|in) (?:the |our )?(?P<where>{GROUP}))"
    r"(?: currently| today| right now)?[?.]?",
    re.IGNORECASE,
)
DATED_COUNT = re.compile(
    rf"how many (?:(?P<group>{GROUP}) )?(?P<role>members|directors|trustees)"
    rf"(?:(?: are| were) (?:on|in) (?:the |our )?(?P<where>{GROUP})"
    r"| (?:(?:do|did) we have|are there|were there))?"
    r" (?:in|for|during) (?P<year>(?:19|20)\d{2})[?.]?",
    re.IGNORECASE,
)
ATTENDANCE_COUNT = re.compile(
    rf"\bhow many (?P<group>{GROUP}) (?:members|directors|trustees)\b", re.IGNORECASE
)
NAME_WORD = rf"{WORD}(?:['’\u2010\u2011-]{WORD})*\.?"
NAME = re.compile(rf"{NAME_WORD}(?: {NAME_WORD}){{1,5}}")
NAME_PARTICLES = frozenset(
    {"da", "de", "del", "den", "der", "di", "du", "la", "van", "von"}
)
NAME_EXCLUSIONS = frozenset(
    {
        "absent",
        "agenda",
        "and",
        "consultant",
        "draft",
        "guest",
        "members",
        "minutes",
        "none",
        "observer",
        "or",
        "present",
        "staff",
        "unknown",
    }
)
GROUP_EXCLUSIONS = frozenset(
    {
        "absent",
        "and",
        "attending",
        "former",
        "guest",
        "guests",
        "new",
        "of",
        "old",
        "or",
        "past",
        "present",
        "staff",
        "visitors",
        "voting",
    }
)
PRESENT = "present"
ABSENT = frozenset({"absent", "apologies", "regrets"})
ROLES = frozenset({"directors", "members", "trustees"})
NONMEMBERS = frozenset({"staff", "guest", "guests", "observers", "visitors"})
NO_ABSENCES = frozenset({"none", "nil", "no absences", "no apologies", "no regrets"})
LABEL_NEGATION = re.compile(
    r"\b(?:not|except|excluding|without)\b|n['\u2019]t\b", re.IGNORECASE
)
PRESENT_LABEL = re.compile(
    r"\b(?:was|were|is|are|as|under)\s+(?:the\s+)?present\b"
    r"|\bpresent\s+(?:members|directors|trustees|section)\b",
    re.IGNORECASE,
)
ROSTER_QUESTION = re.compile(
    rf"(?:please )?(?:name(?: and list)?|list|show|who (?:are|were)) (?:me )?(?:all |every )?(?:the |our )?(?:(?P<group>{GROUP}) )?"
    rf"(?P<role>{WORD}) (?:in |for |during )?(?:19|20)\d{{2}}"
    r"(?: (?:vs\.?|versus|and) (?:19|20)\d{2})?[?.]?",
    re.IGNORECASE,
)
ROSTER_PARAPHRASE = re.compile(
    rf"(?:who|which (?:people|members|individuals|directors|trustees)) "
    rf"(?:served|is|are|was|were|sat) (?:on|in) (?:the |our )?(?P<group>{GROUP}) "
    r"(?:in|for|during) (?:19|20)\d{2}[?.]?",
    re.IGNORECASE,
)
ATTENDANCE_SECTION = re.compile(
    r"^(present|attendees|regrets(?:/absent)?|absent|apologies|staff|guests?|observers|visitors)"
    r"(?:\s*:\s*(.*))?$",
    re.IGNORECASE,
)
DIRECTORY_TITLE = re.compile(
    r"\b(?:contact (?:list|directory)|roster|(?:membership|member) (?:list|directory|register))\b",
    re.IGNORECASE,
)
TENURE_TITLE = re.compile(
    r"\b(?:past|histor(?:y|ical))\b.*\b(?:list|roster|directory)\b|\b(?:tenure|register)\b",
    re.IGNORECASE,
)
TENURE_NAME = {"name", "full name", "member name", "director name", "trustee name"}
TENURE_START = {
    "year joined",
    "date joined",
    "year appointed",
    "date appointed",
    "start year",
}
TENURE_END = {
    "year retired",
    "date retired",
    "year left",
    "date left",
    "year resigned",
    "date resigned",
}
DIRECTORY_BOUNDARY = re.compile(
    r"^(?:(?:board|committee|council|administrative) )?(?:resources?|support|staff|guests?|advisors?|observers?)\s*:?$",
    re.IGNORECASE,
)
MEMBER_PREFIX = re.compile(
    r"^(?:(?:in the |board |committee |council |vice[- ]|past )+)", re.IGNORECASE
)
MEMBER_ROLE = re.compile(
    r"\b(?:chair\w*|director|member|officer|secretary|treasurer|trustee)\b",
    re.IGNORECASE,
)
OTHER_ROLE = re.compile(
    r"\b(?:admin\w*|candidate|consultant|ed|executive director|guest|incoming|nominee|observer|staff|visitor)\b",
    re.IGNORECASE,
)
ROLE_SUFFIX = re.compile(r"\s*\(([^()]*)\)\s*$|\s+-\s+(.+)$")
HONORIFIC = re.compile(r"^(?:dr|miss|mr|mrs|ms|prof)\.?\s+", re.IGNORECASE)


@dataclass(frozen=True)
class RollCallCount:
    present: int
    absent: int

    # Add the two disjoint member groups.
    @property
    def total(self) -> int:
        return self.present + self.absent


@dataclass(frozen=True)
class RollCall:
    group: str
    present: tuple[str, ...]
    absent: tuple[str, ...]

    # Count the same verified names that a roster reply will display.
    @property
    def count(self) -> RollCallCount:
        return RollCallCount(len(self.present), len(self.absent))


# Recognize concise year-scoped totals without capturing questions about attendance.
def dated_count_scope(question: str) -> tuple[str, int] | None:
    match = DATED_COUNT.fullmatch(" ".join(question.casefold().split()))
    if match is None or (match["group"] and match["where"]):
        return None
    group = match["group"] or match["where"]
    if not group and match["role"] == "directors":
        group = "board"
    if not group or GROUP_EXCLUSIONS.intersection(re.findall(WORD, group)):
        return None
    return group, int(match["year"])


# Apply an explicitly named attendance section using the same labels as the reader.
def named_attendance_section(question: str) -> Literal["present", "absent"] | None:
    labels = set(re.findall(WORD, question.casefold()))
    present, absent = bool(PRESENT_LABEL.search(question)), bool(ABSENT & labels)
    if present == absent or LABEL_NEGATION.search(question):
        return None
    return "present" if present else "absent"


# Accept only a simple membership count whose group appears in the source path.
def membership_group(question: str, source: str) -> str:
    normalized = " ".join(question.casefold().split())
    match = QUESTION.fullmatch(normalized)
    if match is not None:
        group, where = match.group("group", "where")
    elif (
        "present" in normalized
        and any(word in normalized for word in ABSENT)
        and "total" in normalized
        and (attendance := ATTENDANCE_COUNT.search(normalized))
    ):
        group, where = attendance["group"], None
    else:
        return ""
    if not (group or where) or (group and where):
        return ""
    words = re.findall(WORD, group or where or "")
    source_words = " ".join(re.findall(WORD, source.casefold()))
    if (
        GROUP_EXCLUSIONS.intersection(words)
        or f" {' '.join(words)} " not in f" {source_words} "
    ):
        return ""
    return " ".join(words)


# Parse plainly delimited names and normalize harmless differences before counting.
def listed_names(value: str, *, roles_required: bool = False) -> dict[str, str] | None:
    entries: dict[str, str] = {}
    for part in re.split(r"[,;](?![^()]*\))|\s+and\s+", normalize("NFKC", value)):
        name = " ".join(part.split())
        if not name:
            continue
        if suffix := ROLE_SUFFIX.search(name):
            role = (suffix[1] or suffix[2]).strip().casefold()
            if MEMBER_ROLE.fullmatch(MEMBER_PREFIX.sub("", role)):
                name = name[: suffix.start()].rstrip()
            elif roles_required and OTHER_ROLE.search(role):
                continue
            else:
                return None
        elif roles_required:
            return None
        if not NAME.fullmatch(name) or any(
            word.casefold() in NAME_EXCLUSIONS
            or (not word[0].isupper() and word.casefold() not in NAME_PARTICLES)
            for word in name.split()
        ):
            return None
        entries.setdefault(
            name.casefold()
            .replace("’", "'")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace(".", ""),
            name,
        )
    return entries


# Read role-labeled attendance, including wrapped lines and separate headings.
def role_labeled_roll_call(excerpt: str, group: str) -> RollCall | None:
    sections = attendance_sections(excerpt)
    present = [
        value for key, value in sections.items() if key in (PRESENT, "attendees")
    ]
    absent = [
        value for key, value in sections.items() if key in ABSENT | {"regrets/absent"}
    ]
    if len(present) != 1 or len(absent) != 1:
        return None
    attending = listed_names(present[0], roles_required=True)
    missing = (
        {}
        if absent[0].casefold() in NO_ABSENCES
        else listed_names(absent[0], roles_required=True)
    )
    if attending and missing is not None and not (attending.keys() & missing.keys()):
        return RollCall(group, tuple(attending.values()), tuple(missing.values()))
    return None


# Read a single adjacent roll call whose labels distinguish members from visitors.
def parse_roll_call(question: str, excerpt: str, *, source: str) -> RollCall | None:
    group = membership_group(question, source)
    if not group:
        return None
    group_words = set(group.split())

    sections: dict[str, tuple[int, str, bool]] = {}
    nonmembers: set[int] = set()
    for index, line in enumerate(line for line in excerpt.splitlines() if line.strip()):
        label, separator, value = line.partition(":")
        if not separator or not value.strip():
            continue
        label = label.strip().casefold()
        if label in NONMEMBERS:
            nonmembers.add(index)
            continue
        words = set(label.split())
        statuses = words & (ABSENT | {PRESENT})
        if len(statuses) != 1 or words - statuses - ROLES - group_words:
            continue
        key = PRESENT if PRESENT in statuses else "absent"
        if key in sections:
            return None
        sections[key] = (index, value.strip(), bool(words & ROLES))

    if sections.keys() != {PRESENT, "absent"}:
        return role_labeled_roll_call(excerpt, group)
    present_line, present_text, present_is_members = sections[PRESENT]
    absent_line, absent_text, absent_is_members = sections["absent"]
    first, last = sorted((present_line, absent_line))
    if set(range(first + 1, last)) - nonmembers:
        return None
    if not (present_is_members and absent_is_members) and not nonmembers.intersection(
        range(first - 1, last + 2)
    ):
        return role_labeled_roll_call(excerpt, group)

    present = listed_names(present_text)
    absent = {} if absent_text.casefold() in NO_ABSENCES else listed_names(absent_text)
    if not present or absent is None or present.keys() & absent.keys():
        return None
    return RollCall(group, tuple(present.values()), tuple(absent.values()))


# Count entries only after validating both membership sections.
def roll_call_count(
    question: str, excerpt: str, *, source: str
) -> RollCallCount | None:
    return (
        call.count
        if (call := parse_roll_call(question, excerpt, source=source))
        else None
    )


# Find dated membership list requests, tolerating minor spelling errors in the role.
def roster_group(question: str) -> str:
    request = re.split(r"[?!]\s+", " ".join(question.split()), maxsplit=1)[0]
    match = ROSTER_QUESTION.fullmatch(request) or ROSTER_PARAPHRASE.fullmatch(request)
    if match is None:
        return ""
    role = match.groupdict().get("role")
    if (
        role
        and role.casefold() not in ROLES
        and not get_close_matches(role.casefold(), ROLES, n=1, cutoff=0.85)
    ):
        return ""
    group = match["group"] or (
        "board" if role and role.casefold() == "directors" else ""
    )
    words = re.findall(WORD, group.casefold())
    return "" if GROUP_EXCLUSIONS.intersection(words) else " ".join(words)


# Read the opening attendance block and preserve wrapped entries under their label.
def attendance_sections(excerpt: str) -> dict[str, str]:
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    sections: dict[str, str] = {}
    label = ""
    for line in lines[:24]:
        if match := ATTENDANCE_SECTION.fullmatch(line):
            label = match[1].casefold()
            if label in sections:
                break
            sections[label] = match[2] or ""
        elif label:
            value = sections[label]
            if (
                not value
                or value.endswith((",", ";", "-"))
                or (
                    label in ABSENT | {"attendees", "present", "regrets/absent"}
                    and (" - " in line or value.count("(") > value.count(")"))
                )
            ):
                sections[label] = f"{value} {line}".lstrip()
            else:
                break
    return sections


# Read member headings in a declared contact directory, ending at the support section.
def contact_directory(text: str) -> tuple[str, tuple[str, ...]] | None:
    lines = text.splitlines()
    header = next(
        (
            (index, line.strip())
            for index, line in enumerate(lines[:6])
            if DIRECTORY_TITLE.search(line)
        ),
        None,
    )
    if header is None:
        return None
    index, heading = header
    members: dict[str, str] = {}
    for line in lines[index + 1 :]:
        if DIRECTORY_BOUNDARY.fullmatch(line.strip()):
            break
        name, _, role = line.split("\t", 1)[0].strip().partition(",")
        if role and not MEMBER_ROLE.fullmatch(MEMBER_PREFIX.sub("", role.strip())):
            continue
        if names := listed_names(name):
            members.update(names)
    return (heading, tuple(members.values())) if len(members) > 1 else None


# Keep dated periods and joining years without extending undated tenures forward.
def dated_tenure_names(tables: list[list[list[str]]], year: int) -> tuple[str, ...]:
    members: dict[str, str] = {}
    for rows in tables:
        if not rows:
            continue
        name, joined, retired = (
            next(
                (
                    i
                    for i, cell in enumerate(rows[0])
                    if " ".join(cell.casefold().split()) in labels
                ),
                -1,
            )
            for labels in (TENURE_NAME, TENURE_START, TENURE_END)
        )
        if min(name, joined, retired) < 0:
            continue
        for row in rows[1:]:
            if len(row) <= max(name, joined, retired):
                continue
            start = YEAR.match(row[joined])
            start_years = YEAR.findall(row[joined])
            end_years = YEAR.findall(row[retired])
            if start is None or len(start_years) != 1 or len(end_years) > 1:
                continue
            joined_year = int(start[0])
            retired_year = int(end_years[0]) if end_years else joined_year
            if not joined_year <= year <= retired_year:
                continue
            names = listed_names(
                re.sub(r"\s*\(?deceased\)?$", "", row[name], flags=re.IGNORECASE)
            )
            if names:
                members.update(names)
    return tuple(members.values())


# Keep names from member sections or names with an explicit membership role.
def roster_names(excerpt: str, group: str, source: str) -> list[str]:
    source_words = " ".join(re.findall(WORD, source.casefold()))
    if f" {group} " not in f" {source_words} ":
        return []
    return attendance_names(excerpt)


# Extract names from an attendance block that belongs to an identified meeting.
def attendance_names(excerpt: str) -> list[str]:
    sections = attendance_sections(excerpt)
    has_explicit_sections = (
        PRESENT in sections
        and bool(sections.keys() & (ABSENT | {"regrets/absent"}))
        and bool(sections.keys() & NONMEMBERS)
    )
    names = []
    for status, value in sections.items():
        if status in NONMEMBERS:
            continue
        for entry in re.split(r"[,;](?![^()]*\))|\s+and\s+", value):
            name = entry.strip().removesuffix(" NO QUORUM").strip()
            suffix = ROLE_SUFFIX.search(name)
            if suffix:
                role = suffix[1] or suffix[2]
                if not MEMBER_ROLE.search(role) or OTHER_ROLE.search(role):
                    continue
                name = name[: suffix.start()].strip()
            elif not has_explicit_sections or status == "attendees":
                continue
            name = HONORIFIC.sub("", name)
            if listed_names(name):
                names.append(name)
    return names


# Reconcile supported spelling variants that never occur together in a roll call.
def roster_aliases(roll_calls: list[list[str]]) -> dict[str, str]:
    frequency = Counter(name for call in roll_calls for name in set(call))
    calls_by_name: dict[str, set[int]] = {}
    for index, call in enumerate(roll_calls):
        for name in call:
            calls_by_name.setdefault(name, set()).add(index)

    # Keep full names ahead of shortened forms and repeated spellings ahead of typos.
    def preference(name: str) -> tuple[int, int, str]:
        return -len(re.findall(WORD, name)), -frequency[name], name.casefold()

    aliases: dict[str, str] = {}
    for name in sorted(frequency, key=preference):
        key = " ".join(re.findall(WORD, name.casefold()))
        for other, canonical in aliases.items():
            if calls_by_name[name] & calls_by_name[canonical]:
                continue
            candidate = " ".join(re.findall(WORD, other.casefold()))
            if (
                candidate.startswith(f"{key} ")
                or candidate == key
                or (
                    candidate.split()[0] == key.split()[0]
                    and abs(len(candidate) - len(key)) <= 1
                    and SequenceMatcher(None, candidate, key).ratio() >= 0.94
                )
            ):
                aliases[name] = canonical
                calls_by_name[canonical].update(calls_by_name[name])
                break
        else:
            aliases[name] = name
    return aliases
