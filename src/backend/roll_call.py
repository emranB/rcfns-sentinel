'''Count members accounted for by an unambiguous roll call.'''

import re
from dataclasses import dataclass
from unicodedata import normalize

WORD = r"[^\W\d_]+"
GROUP = rf"{WORD}(?:[ -]{WORD}){{0,3}}?"
QUESTION = re.compile(
    rf"how many (?:(?P<group>{GROUP}) )?(?:members|directors|trustees) "
    rf"(?:do we (?:currently )?have|are there|"
    rf"are (?:on|in) (?:the |our )?(?P<where>{GROUP}))"
    r"(?: currently| today| right now)?[?.]?",
    re.IGNORECASE,
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


@dataclass(frozen=True)
class RollCallCount:
    present: int
    absent: int

    # Add the two disjoint member groups.
    @property
    def total(self) -> int:
        return self.present + self.absent


# Accept only a simple membership count whose group appears in the source path.
def membership_group(question: str, source: str) -> set[str]:
    match = QUESTION.fullmatch(" ".join(question.casefold().split()))
    if match is None:
        return set()
    group, where = match.group("group", "where")
    if not (group or where) or (group and where):
        return set()
    words = re.findall(WORD, group or where)
    source_words = " ".join(re.findall(WORD, source.casefold()))
    if (
        GROUP_EXCLUSIONS.intersection(words)
        or f" {' '.join(words)} " not in f" {source_words} "
    ):
        return set()
    return set(words)


# Parse plainly delimited names and normalize harmless differences before counting.
def listed_names(value: str) -> set[str] | None:
    entries: set[str] = set()
    for part in re.split(r"[,;]", normalize("NFKC", value)):
        name = " ".join(part.split())
        if not NAME.fullmatch(name) or any(
            word.casefold() in NAME_EXCLUSIONS
            or (not word[0].isupper() and word.casefold() not in NAME_PARTICLES)
            for word in name.split()
        ):
            return None
        entries.add(
            name.casefold()
            .replace("’", "'")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace(".", "")
        )
    return entries


# Count a single adjacent roll call whose labels distinguish members from visitors.
def roll_call_count(
    question: str, excerpt: str, *, source: str
) -> RollCallCount | None:
    group = membership_group(question, source)
    if not group:
        return None

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
        if len(statuses) != 1 or words - statuses - ROLES - group:
            continue
        key = PRESENT if PRESENT in statuses else "absent"
        if key in sections:
            return None
        sections[key] = (index, value.strip(), bool(words & ROLES))

    if sections.keys() != {PRESENT, "absent"}:
        return None
    present_line, present_text, present_is_members = sections[PRESENT]
    absent_line, absent_text, absent_is_members = sections["absent"]
    first, last = sorted((present_line, absent_line))
    if set(range(first + 1, last)) - nonmembers:
        return None
    if not (present_is_members and absent_is_members) and not nonmembers.intersection(
        range(first - 1, last + 2)
    ):
        return None

    present = listed_names(present_text)
    absent = (
        set() if absent_text.casefold() in NO_ABSENCES else listed_names(absent_text)
    )
    if not present or absent is None or present & absent:
        return None
    return RollCallCount(present=len(present), absent=len(absent))
