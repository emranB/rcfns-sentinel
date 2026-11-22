'''Preserve table alignment when preparing retrieved document excerpts.'''

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)


@dataclass
class TableCell:
    columns: int
    rows: int
    text: list[str] = field(default_factory=list)


# Read a common HTML table without interpreting the meaning of individual cells.
class TableParser(HTMLParser):
    # Track explicit cells in document order.
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[TableCell]] = []
        self.cell: TableCell | None = None
        self.caption: list[str] = []
        self.in_caption = False

    # Start each row or read the dimensions of its next cell.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "caption":
            self.in_caption = True
        elif tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"} and self.rows:
            spans = dict(attrs)
            self.cell = TableCell(
                columns=span_size(spans.get("colspan")),
                rows=span_size(spans.get("rowspan")),
            )
            self.rows[-1].append(self.cell)
        elif tag == "br":
            self.handle_data(" ")

    # End the current cell or caption.
    def handle_endtag(self, tag: str) -> None:
        if tag == "caption":
            self.in_caption = False
        elif tag in {"td", "th", "tr"}:
            self.cell = None

    # Collect visible cell and caption text without embedding HTML markup.
    def handle_data(self, data: str) -> None:
        if self.in_caption:
            self.caption.append(data)
        elif self.cell:
            self.cell.text.append(data)


# Bound ambiguous or oversized spans before expanding a grid.
def span_size(value: str | None) -> int:
    return (
        min(64, max(1, int(value)))
        if value and value.isascii() and value.isdecimal() and len(value) < 5
        else 1
    )


# Reconstruct the column grid before rendering. Merged positions refer to the
# original cell; invalid spans fall back to the original table instead of shifting values.
def render_table(match: re.Match[str]) -> str:
    table = match[0]
    if len(re.findall(r"<table\b", table, re.IGNORECASE)) != 1:
        return table

    parser = TableParser()
    parser.feed(table)
    parser.close()
    if not any(parser.rows):
        return table

    active_spans: dict[int, int] = {}
    rendered = []
    for row_index, cells in enumerate(parser.rows):
        active_spans = {
            column: end for column, end in active_spans.items() if row_index < end
        }
        row = dict.fromkeys(active_spans, "(above)")
        column = 0
        for cell in cells:
            while column in row:
                column += 1
            value = " ".join("".join(cell.text).split()).replace("|", r"\|")
            for offset in range(cell.columns):
                slot = column + offset
                if slot in row:
                    return table
                row[slot] = "(left)" if offset else value
                if cell.rows > 1:
                    active_spans[slot] = row_index + cell.rows
            column += cell.columns
        if any(row.values()):
            rendered.append(
                "| " + " | ".join(row.get(i, "") for i in range(max(row) + 1)) + " |"
            )
    if not rendered:
        return table
    caption = " ".join("".join(parser.caption).split())
    return (
        (f"\n{caption}\n" if caption else "\n")
        + "Table (left-to-right columns, parentheses mark merged cells):\n"
        + "\n".join(rendered)
        + "\n"
    )


# Format complete HTML tables and leave other retrieved text as extracted.
def format_excerpt(text: str) -> str:
    return TABLE.sub(render_table, text).strip()
