"""Count members accounted for by an unambiguous roll call."""

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
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
ROSTER_QUESTION = re.compile(
    rf"(?:please )?(?:name(?: and list)?|list) (?:all |every )?(?:the )?(?P<group>{GROUP}) "
    rf"(?P<role>{WORD}) (?:in |for |during )?(?:19|20)\d{{2}}"
    r"(?: (?:vs\.?|versus|and) (?:19|20)\d{2})?[?.]?",
    re.IGNORECASE,
)
ROSTER_PARAPHRASE = re.compile(
    rf"who (?:served|was|were|sat) (?:on|in) (?:the |our )?(?P<group>{GROUP}) "
    r"(?:in|for|during) (?:19|20)\d{2}[?.]?",
    re.IGNORECASE,
)
ATTENDANCE_SECTION = re.compile(
    r"^(present|attendees|regrets(?:/absent)?|absent|apologies|staff|guests?|observers|visitors)"
    r"(?:\s*:\s*(.*))?$",
    re.IGNORECASE,
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
    words = re.findall(WORD, match["group"].casefold())
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
                    label == "attendees"
                    and (" - " in line or value.count("(") > value.count(")"))
                )
            ):
                sections[label] = f"{value} {line}".lstrip()
            else:
                break
    return sections


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
