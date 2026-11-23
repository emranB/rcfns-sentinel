'''Represent cited numeric inputs and evaluate bounded arithmetic over them.'''

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Literal

from pydantic import BaseModel, Field

from src.backend.current_sources import source_terms
from src.backend.grounding import EvidenceText, normalized
from src.backend.tabular import Worksheet, worksheet_tables

CALCULATION_REQUEST = re.compile(
    r"\b(?:calculat\w*|comput\w*|subtract\w*|divid\w*|multiply|ratio|working capital|"
    r"variance|percentage (?:change|of|increase|decrease)|percent change|difference|"
    r"how much [^?!.]{0,45}\b(?:higher|lower|more|less|change[ds]?|increas\w*|decreas\w*)|"
    r"add(?:s|ed)? up|tie(?:s)? out|reconcil(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
CALCULATION_REFERENCE = re.compile(
    r"\b(?:same|that|this|those)\s+(?:\w+\s+){0,2}(?:reports?|sheets?|workbooks?|files?|documents?)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"\(?(?:-?[$\u00a3\u20ac]?|[$\u00a3\u20ac] ?-)\d[\d,]*(?:\.\d+)?%?\)?"
)
CELL = re.compile(r"\S.*?(?= {2,}|$)")
CELL_REFERENCE = re.compile(r"([A-Z]{1,3})([1-9]\d*)")
OPERATORS = {
    "sum": "+",
    "difference": "-",
    "product": "x",
    "ratio": "/",
    "share": "/",
    "percent_change": "-",
}


class OperandSource(BaseModel):
    title: str = Field(max_length=160)
    year: int | None = Field(ge=1900, le=2099)
    month: int | None = Field(ge=1, le=12)
    day: int | None = Field(ge=1, le=31)
    period: Literal["any", "monthly", "cumulative"]
    query: str = Field(min_length=1, max_length=350)


class QuantityPlan(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    reuse_sources: bool
    sources: list[OperandSource] = Field(min_length=1, max_length=5)


class NumericInput(BaseModel):
    source: int = Field(ge=1)
    quote: str = Field(min_length=4, max_length=400)
    amount: str = Field(min_length=1, max_length=40)


class Equation(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    operation: Literal[
        "sum", "difference", "product", "ratio", "share", "percent_change"
    ]
    inputs: list[int] = Field(min_length=2, max_length=8)
    precision: int = Field(ge=0, le=4)


class QuantityAnswer(BaseModel):
    inputs: list[NumericInput] = Field(max_length=12)
    equations: list[Equation] = Field(max_length=5)
    missing: list[str] = Field(max_length=6)


@dataclass(frozen=True)
class CitedAmount:
    number: Decimal
    unit: str
    source: int
    line: str

    # Keep a currency prefix or a percent suffix with its value.
    def display(self) -> str:
        exponent = self.number.as_tuple().exponent
        precision = max(0, -exponent) if isinstance(exponent, int) else 0
        digits = f"{abs(self.number):,.{precision}f}"
        sign = "-" if self.number < 0 else ""
        if self.unit == "%":
            return f"{sign}{digits}%"
        return f"{sign}{self.unit}{digits}"


# Read a single printed numeric token without changing its sign or unit.
def amount_value(text: str) -> tuple[Decimal, str] | None:
    if len(text) > 40 or not NUMBER.fullmatch(text):
        return None
    unit = next((symbol for symbol in "$\u00a3\u20ac%" if symbol in text), "")
    number = Decimal(
        text.strip("()").replace(unit, "").replace(",", "").replace(" ", "")
    )
    if number.adjusted() > 23:
        return None
    return (-abs(number) if text.startswith("(") else number), unit


# Preserve the original row label and bind each recovered value to its column heading.
def numeric_layout(text: str, question: str, limit: int = 36) -> str:
    heading = [line.strip() for line in text.splitlines() if line.strip()][:3]
    terms = source_terms(question)
    rows: dict[str, int] = {}
    for page in text.split("\f"):
        columns: list[tuple[int, str]] = []
        data_started = False
        for line in page.splitlines():
            cells = list(CELL.finditer(line))
            if not cells:
                continue
            values = [(cell, amount_value(cell[0])) for cell in cells[1:]]
            numbers = [(cell, value) for cell, value in values if value is not None]
            if not numbers:
                if not data_started and (
                    len(cells) > 1 or cells[0][0].casefold() == "total"
                ):
                    columns = [
                        (cell.end(), cell[0]) for cell in cells if cell.start() > 15
                    ]
                continue
            data_started = True
            label = cells[0][0]
            score = 4 * len(source_terms(label) & terms) + int(
                "total" in source_terms(label)
            )
            for cell, (value, unit) in numbers:
                choices = sorted((abs(end - cell.end()), name) for end, name in columns)
                column = (
                    choices[0][1]
                    if choices
                    and (
                        len(choices) == 1
                        or choices[0][0] <= 4
                        and choices[0][0] < choices[1][0]
                    )
                    else ""
                )
                if not column and len(numbers) > 1:
                    continue
                printed = CitedAmount(value, unit, 0, "").display()
                entry = (
                    f"{label} | {column} | {printed}"
                    if column
                    else f"{label} | {printed}"
                )
                rows[entry] = score + len(source_terms(column) & terms)
    selected = sorted(rows, key=rows.__getitem__, reverse=True)[:limit]
    return "\n".join([*heading, *selected]) if selected else ""


# Resolve spreadsheet column numbers independent of their letter length.
def column_number(letters: str) -> int:
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - ord("A") + 1
    return number


# Read the first intact table header and any merged group immediately above it.
def workbook_headers(sheet: Worksheet) -> dict[str, str]:
    heading: dict[str, str] = {}
    earlier: dict[str, str] = {}
    for row in sheet.rows[:20]:
        if len(row) < 2:
            earlier.update(row)
            continue
        if any(amount_value(value) is not None for value in list(row.values())[1:]):
            break
        earlier.update(heading)
        heading = row
    result = {}
    for cell, label in heading.items():
        if not (header := CELL_REFERENCE.fullmatch(cell)):
            continue
        text = label
        column = column_number(header[1])
        for span in sheet.merges:
            start, _, finish = span.partition(":")
            left, right = (
                CELL_REFERENCE.fullmatch(start),
                CELL_REFERENCE.fullmatch(finish),
            )
            if left is None or right is None or start not in earlier:
                continue
            if int(left[2]) + 1 == int(header[2]) and column_number(
                left[1]
            ) <= column <= column_number(right[1]):
                text = f"{earlier[start]} / {label}"
                break
        result[header[1]] = text
    return result


# Expose cached numeric cells with their sheet, row and column identities.
def numeric_workbook(payload: bytes, question: str, limit: int = 36) -> str:
    terms = source_terms(question)
    selected: dict[str, int] = {}
    for sheet in worksheet_tables(payload):
        headers = workbook_headers(sheet)
        if not headers:
            continue
        for row in sheet.rows:
            if len(row) < 2:
                continue
            label = next(iter(row.values()))
            for cell, text in list(row.items())[1:]:
                if not (reference := CELL_REFERENCE.fullmatch(cell)) or not (
                    header := headers.get(reference[1])
                ):
                    continue
                if not (amount := amount_value(text)):
                    continue
                number, unit = amount
                if "%" in header and not unit:
                    number *= 100
                    unit = "%"
                number = round(number, 4 if unit == "%" else 2)
                printed = CitedAmount(number, unit, 0, "").display()
                quote = f"{label} | {header} | {sheet.name}!{cell} | {printed}"
                selected[quote] = 4 * len(source_terms(label) & terms) + 2 * len(
                    source_terms(header) & terms
                )
    return "\n".join(sorted(selected, key=selected.__getitem__, reverse=True)[:limit])


# Accept an input only when the cited line fixes its context and numeric value.
def cited_amount(
    fact: NumericInput, evidence: list[EvidenceText]
) -> CitedAmount | None:
    if fact.source > len(evidence) or (expected := amount_value(fact.amount)) is None:
        return None
    quote = normalized(fact.quote)
    for line in evidence[fact.source - 1].content.splitlines():
        if normalized(line) != quote or not any(
            character.isalpha() for character in line
        ):
            continue
        if " | " in line:
            matches = [
                part.strip()
                for part in line.split(" | ")
                if NUMBER.fullmatch(part.strip())
            ]
        else:
            matches = [
                match[0]
                for match in re.finditer(rf"(?<!\w){NUMBER.pattern}(?!\w)", line)
            ]
        if len(matches) != 1 or (actual := amount_value(matches[0])) is None:
            continue
        if actual[0] == expected[0] and (not expected[1] or expected[1] == actual[1]):
            return CitedAmount(*actual, fact.source, line)
    return None


# Evaluate only the declared operation on vetted inputs with compatible units.
def calculate(equation: Equation, amounts: list[CitedAmount | None]) -> str:
    if not all(1 <= index <= len(amounts) for index in equation.inputs):
        return ""
    selected = [amounts[index - 1] for index in equation.inputs]
    known = [item for item in selected if item is not None]
    if len(known) != len(selected) or len(set(equation.inputs)) != len(selected):
        return ""
    operation = equation.operation
    units = {item.unit for item in known if item.unit}
    if (
        len(units) > 1
        or operation in {"ratio", "share", "percent_change"}
        and len(known) != 2
    ):
        return ""
    values = [item.number for item in known]
    unit = next(iter(units), "")
    with localcontext() as context:
        context.prec = 80
        first, second = values[:2]
        if operation in {"ratio", "share", "percent_change"} and not second:
            return ""
        if operation == "difference":
            result = first - sum(values[1:], Decimal(0))
        elif operation == "sum":
            result = sum(values, Decimal(0))
        elif operation == "product":
            result = Decimal(1)
            for value in values:
                result *= value
        elif operation == "percent_change":
            result = (first - second) / abs(second) * 100
        else:
            result = first / second * (100 if operation == "share" else 1)
        if result.adjusted() > 36:
            return ""
        result = result.quantize(
            Decimal(1).scaleb(-equation.precision), rounding=ROUND_HALF_UP
        )
    rendered = [item.display() for item in known]
    formula = f" {OPERATORS[operation]} ".join(rendered)
    if operation in {"share", "percent_change"}:
        formula = (
            f"({formula}) x 100"
            if operation == "share"
            else f"({formula}) / {abs(second):,f} x 100"
        )
        unit = "%"
    elif operation == "ratio":
        unit = ""
    answer = CitedAmount(result, unit, 0, "").display()
    references = " ".join(
        f"[{ref}]" for ref in dict.fromkeys(item.source for item in known)
    )
    label = re.sub(r"\[\d+\]|[\r\n]", "", equation.label).strip()
    return f"**{label}** = **{answer}** ({formula}) {references}"
