'''Find strong references to named original documents in a question.'''

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from unicodedata import normalize

WORD = re.compile(r"[^\W_]+")
TITLE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
MINIMUM_TITLE_TERMS = 3
TITLE_CONTEXT_WORDS = 4
MAXIMUM_NAMED_DOCUMENTS = 3


@dataclass(frozen=True)
class TitleMatch:
    document_id: str
    start: int
    end: int
    score: tuple[int, float, int]


# Ignore case and punctuation when comparing filenames and question text.
def title_words(text: str) -> list[str]:
    return WORD.findall(normalize("NFKC", text).casefold())


# Rank contiguous title phrases and use nearby words to distinguish dated versions.
def named_document_ids(
    question: str, documents: Iterable[tuple[str, str]]
) -> list[str]:
    words = title_words(question)
    question_terms = set(words) - TITLE_STOP_WORDS
    if len(question_terms) < MINIMUM_TITLE_TERMS:
        return []

    matcher = SequenceMatcher(None, [], words, autojunk=False)
    matches: list[TitleMatch] = []
    for document_id, path in documents:
        filename = PurePosixPath(path.replace("\\", "/")).stem
        title = title_words(filename)
        terms = set(title) - TITLE_STOP_WORDS
        shared = len(terms & question_terms)
        if shared < MINIMUM_TITLE_TERMS or shared * 2 < len(terms):
            continue

        matcher.set_seq1(title)
        for _, start, length in matcher.get_matching_blocks():
            end = start + length
            phrase = set(words[start:end]) - TITLE_STOP_WORDS
            if len(phrase) < MINIMUM_TITLE_TERMS:
                continue
            context = set(
                words[max(0, start - TITLE_CONTEXT_WORDS) : end + TITLE_CONTEXT_WORDS]
            )
            score = len(phrase), len(terms & context) / len(terms), length
            matches.append(TitleMatch(document_id, start, end, score))

    # An ambiguous mention should fall back to content search instead of a guessed version.
    alternatives = Counter((match.start, match.end, match.score) for match in matches)
    claimed: list[tuple[int, int]] = []
    selected: dict[str, int] = {}
    for match in sorted(matches, key=lambda candidate: candidate.score, reverse=True):
        if match.document_id in selected or any(
            match.start < end and start < match.end for start, end in claimed
        ):
            continue
        claimed.append((match.start, match.end))
        if alternatives[match.start, match.end, match.score] == 1:
            selected[match.document_id] = match.start
        if len(selected) == MAXIMUM_NAMED_DOCUMENTS:
            break

    return [
        document_id
        for document_id, _ in sorted(selected.items(), key=lambda item: item[1])
    ]
