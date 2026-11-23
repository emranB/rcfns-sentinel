'''Read complete fields from bounded table and workbook data.'''

import re
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from posixpath import normpath
from xml.etree.ElementTree import Element, ParseError
from zipfile import BadZipFile, ZipFile

from src.backend.office_text_recovery import read_parts, tag_name
from src.backend.record_lookup import mentioned_dates

WORKBOOK = "xl/workbook.xml"
RELATIONSHIPS = "xl/_rels/workbook.xml.rels"
STRINGS = "xl/sharedStrings.xml"
MONEY = re.compile(r"[$]?\s*(-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2})")
SHEET_QUESTION = re.compile(
    r"\b(?:sheet|worksheet)\s+(?:named\s+)?[\"']?(?P<sheet>[^\"'?\n]+?)[\"']?\s+"
    r"(?=(?:in|of|from)\s)",
    re.IGNORECASE,
)
COUNT_NAMES = re.compile(
    r"\b(?:count|how many)\b.*\b(?:named entries|names|named rows)\b", re.IGNORECASE
)
NAME_HEADERS = frozenset(
    {"name", "full name", "first name", "last name", "given name", "surname"}
)
NAME_LIST = re.compile(
    r"\b(?:give|list|identify|report)\b[^?!.\n]{0,60}\bnames\b", re.IGNORECASE
)
PERSON_FIELD = re.compile(r"[^\W\d_]+(?:[ .'\u2019-][^\W\d_]+)*\.?")


@dataclass(frozen=True)
class TableRow:
    name: str
    values: tuple[Decimal, ...]


# Join explicit given-name and family-name columns without copying adjacent details.
def table_person_names(question: str, text: str) -> list[str]:
    if not NAME_LIST.search(question):
        return []
    given = family = -1
    names: list[str] = []
    for line in text.splitlines():
        if not line.startswith("|") or not line.endswith("|"):
            if names:
                break
            given = family = -1
            continue
        cells = [part.strip() for part in line[1:-1].split("|")]
        if given < 0 or family < 0:
            fields = [set(re.findall(r"[^\W_]+", value.casefold())) for value in cells]
            given = next(
                (
                    index
                    for index, words in enumerate(fields)
                    if "name" in words and words & {"given", "first", "forename"}
                ),
                -1,
            )
            family = next(
                (
                    index
                    for index, words in enumerate(fields)
                    if "surname" in words
                    or ("name" in words and words & {"family", "last"})
                ),
                -1,
            )
        elif max(given, family) < len(cells) and all(
            PERSON_FIELD.fullmatch(cells[index]) for index in (given, family)
        ):
            names.append(f"{cells[given]} {cells[family]}")
    return list(dict.fromkeys(names))


# Restore a split label and numeric continuation without changing column order.
def table_rows(text: str) -> list[TableRow]:
    result = []
    pending = ""
    for line in text.splitlines():
        if not line.startswith("|") or not line.endswith("|"):
            pending = ""
            continue
        cells = [part.strip() for part in line[1:-1].split("|")]
        if len(cells) < 2:
            pending = ""
            continue
        name, *values = cells
        numbers = [MONEY.fullmatch(value) for value in values]
        if (name or pending) and all(numbers):
            result.append(
                TableRow(
                    name or pending,
                    tuple(Decimal(m[1].replace(",", "")) for m in numbers if m),
                )
            )
            pending = ""
        else:
            pending = name if name and not any(values) else ""
    return result


# Select date columns from an intact header and keep only explicitly mentioned rows.
def requested_table(
    question: str, excerpts: list[str]
) -> tuple[list[str], list[TableRow]] | None:
    normalized = " ".join(question.casefold().split())
    for text in excerpts:
        headings: list[str] = []
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            columns = line.strip("|").split("|")[1:]
            if len(columns) >= 2 and all(
                len(mentioned_dates(col)) == 1 for col in columns
            ):
                headings = [col.strip() for col in columns]
                break
        if not headings:
            continue
        rows = [
            row
            for row in table_rows(text)
            if len(row.values) == len(headings)
            and re.search(rf"\b{re.escape(row.name.casefold())}\b", normalized)
        ]
        selected = list(dict.fromkeys(rows))
        # Prefer the full row name where two labels match the same query span.
        selected = [
            row
            for row in selected
            if not any(
                row != other
                and row.name.casefold() in other.name.casefold()
                and len(
                    re.findall(rf"\b{re.escape(row.name.casefold())}\b", normalized)
                )
                == len(
                    re.findall(rf"\b{re.escape(other.name.casefold())}\b", normalized)
                )
                for other in selected
            )
        ]
        if selected and len({row.name for row in selected}) == len(selected):
            return headings, selected
    return None


# Recognize a request to count named rows on a specified worksheet.
def name_count_sheet(question: str) -> str:
    match = SHEET_QUESTION.search(question)
    return match["sheet"].strip() if match and COUNT_NAMES.search(question) else ""


# Read literal cell values, including inline and shared strings.
def cell_value(cell: Element, strings: list[str]) -> str:
    if cell.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.iter() if tag_name(t) == "t").strip()
    value = next((v.text or "" for v in cell if tag_name(v) == "v"), "")
    if cell.get("t") == "s":
        return strings[int(value)].strip()
    return value.strip()


# Open one named sheet through its workbook relationship and count populated name rows.
def count_named_rows(payload: bytes, name: str) -> int | None:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            workbook, links = read_parts(archive, [WORKBOOK, RELATIONSHIPS])
            matches = [
                sheet
                for sheet in workbook.iter()
                if tag_name(sheet) == "sheet"
                and sheet.get("name", "").casefold() == name.casefold()
            ]
            if len(matches) != 1:
                return None
            link_id = next(
                (
                    value
                    for key, value in matches[0].attrib.items()
                    if key.endswith("}id")
                ),
                "",
            )
            target = next(
                (
                    rel.get("Target", "")
                    for rel in links
                    if rel.get("Id") == link_id and rel.get("TargetMode") != "External"
                ),
                "",
            )
            sheet_path = normpath(
                target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            )
            if not sheet_path.startswith("xl/worksheets/"):
                return None
            names = (
                [sheet_path, STRINGS] if STRINGS in archive.namelist() else [sheet_path]
            )
            sheet, *shared = read_parts(archive, names)
        strings = [
            "".join(t.text or "" for t in item.iter() if tag_name(t) == "t")
            for root in shared
            for item in root
            if tag_name(item) == "si"
        ]
        rows = [
            {
                re.sub(r"\d+$", "", cell.get("r", "")): cell_value(cell, strings)
                for cell in row
                if tag_name(cell) == "c"
            }
            for row in sheet.iter()
            if tag_name(row) == "row"
        ]
    except (
        BadZipFile,
        IndexError,
        KeyError,
        OSError,
        ParseError,
        RuntimeError,
        ValueError,
    ):
        return None

    for index, row in enumerate(rows[:12]):
        columns = {
            key for key, value in row.items() if value.casefold() in NAME_HEADERS
        }
        if columns:
            # A name cell is an entry when it contains a name rather than a prose note.
            return sum(
                any(
                    re.fullmatch(r"[^\W\d_][^\d.!?\n]{0,99}", entry.get(key, ""))
                    for key in columns
                )
                for entry in rows[index + 1 :]
            )
    return None
