'''Summarize the configured library from its original file paths.'''

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath

GROUP_LIMIT = 6
EXAMPLE_LIMIT = 2
MARKDOWN_SYNTAX = re.compile(r"([\\`*_~\[\]<>])")


@dataclass(frozen=True)
class CollectionGroup:
    name: str
    file_count: int
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollectionOverview:
    original_count: int = 0
    groups: tuple[CollectionGroup, ...] = ()
    root_count: int = 0

    # Describe the inventory and distinguish it from the searchable index.
    def format(
        self, indexed_count: int, total_count: int, *, concise: bool = False
    ) -> str:
        original = "original file" if self.original_count == 1 else "original files"
        headline = f"The library contains {self.original_count:,} {original}"
        if concise:
            if self.groups:
                leading = " and ".join(
                    f"{display_name(group.name)} ({group.file_count:,})"
                    for group in self.groups[:2]
                )
                headline += f", led by {leading}"
            if total_count:
                return (
                    f"{headline}, with {indexed_count:,} of {total_count:,} "
                    "items selected for indexing now searchable."
                )
            return f"{headline}."

        answer = f"{headline}."
        if total_count:
            answer += (
                f" {indexed_count:,} of {total_count:,} items selected for indexing "
                "are searchable."
            )

        bullets = []
        for group in self.groups[:GROUP_LIMIT]:
            label = display_name(group.name)
            files = "file" if group.file_count == 1 else "files"
            line = f"- **{label}** has {group.file_count:,} {files}."
            if group.examples:
                examples = " and ".join(map(display_name, group.examples))
                line += f" Subfolders include {examples}."
            bullets.append(line)
        remaining = self.groups[GROUP_LIMIT:]
        if remaining:
            count = sum(group.file_count for group in remaining)
            files = "file" if count == 1 else "files"
            bullets.append(f"- **Other groups** account for {count:,} {files}.")
        if self.root_count:
            files = "file" if self.root_count == 1 else "files"
            bullets.append(f"- **Library root** holds {self.root_count:,} {files}.")
        return "\n\n".join((answer, "\n".join(bullets))) if bullets else answer


# Keep names as text even if a folder name contains Markdown syntax.
def display_name(name: str) -> str:
    return MARKDOWN_SYNTAX.sub(r"\\\1", name.replace("\r", " ").replace("\n", " "))


# Rank counts consistently even when discovery order or casing differs.
def ranked_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(
        counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
    )


# Remove only the directory prefix shared by every original.
def common_depth(originals: list[PurePosixPath]) -> int:
    first = originals[0].parent.parts
    depth = len(first)
    for original in originals[1:]:
        parts = original.parent.parts
        depth = min(depth, len(parts))
        while parts[:depth] != first[:depth]:
            depth -= 1
        if not depth:
            break
    return depth


# Group originals by their top-level folders, or by extension for a flat library.
def build_overview(paths: Iterable[str]) -> CollectionOverview:
    originals = list({path for path in map(PurePosixPath, paths) if path.name})
    if not originals:
        return CollectionOverview()

    depth = common_depth(originals)
    group_counts: Counter[str] = Counter()
    subfolders: dict[str, Counter[str]] = {}
    root_files = []
    for original in originals:
        parts = original.parts[depth:]
        if len(parts) == 1:
            root_files.append(original)
            continue
        folder = parts[0]
        group_counts[folder] += 1
        for child in parts[1:-1]:
            if any(character.isalpha() for character in child):
                subfolders.setdefault(folder, Counter())[child] += 1
                break

    if not group_counts:
        group_counts.update(
            file.suffix[1:].upper() or "No file extension" for file in root_files
        )
        root_files.clear()

    groups = tuple(
        CollectionGroup(
            name,
            count,
            tuple(
                example
                for example, _ in ranked_counts(subfolders.get(name, Counter()))[
                    :EXAMPLE_LIMIT
                ]
            ),
        )
        for name, count in ranked_counts(group_counts)
    )
    return CollectionOverview(len(originals), groups, len(root_files))
