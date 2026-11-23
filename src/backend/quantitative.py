'''Represent cited numeric inputs and evaluate bounded arithmetic over them.'''

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Literal

from pydantic import BaseModel, Field

from src.backend.current_sources import MONTH_HINT, source_terms
from src.backend.tabular import Worksheet, worksheet_tables

CALCULATION_REQUEST = re.compile(
    r"\b(?:calculat\w*|comput\w*|subtract\w*|divid\w*|multiply|ratio|working capital|"
    r"variance|percent(?:age)? (?:change|of|increase|decrease|for the same|for that same)|difference|"
    r"(?:what|as a) (?:share|proportion|fraction) of|"
    r"how much [^?!.]{0,45}\b(?:higher|lower|more|less|change[ds]?|increas\w*|decreas\w*)|"
    r"add(?:s|ed)? up|tie(?:s)? out|reconcil(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
CALCULATION_REFERENCE = re.compile(
    r"\b(?:same|that|this|those)\s+(?:\w+\s+){0,2}(?:reports?|sheets?|workbooks?|files?|documents?|dates?)\b",
    re.IGNORECASE,
)
PRIOR_CALCULATION_PERIOD = re.compile(
    r"\bthe (?P<period>full.year|annual|monthly|month.only|year.to.date|cumulative) "
    r"(?:report|statement|sheet|original)\b",
    re.IGNORECASE,
)
PERIOD_CHANGE = re.compile(
    r"\b(?:year.over.year|change in|difference in|how much (?:did|has) .{1,90}"
    r"(?:increas\w*|decreas\w*|chang\w*))\b",
    re.IGNORECASE,
)
SNAPSHOT_METRIC = re.compile(r"\b(?:working capital|current ratio)\b", re.IGNORECASE)
FROM_FIRST_TO_LAST = re.compile(
    r"\b(?:from|between) (?:the )?(?:first|earliest|oldest)\b[^?.]{0,60}"
    r"\b(?:last|latest|most recent)\b",
    re.IGNORECASE,
)
RECONCILE_SUM = re.compile(
    r"\b(?:do|does)\s+(?P<terms>[^?]+?\s+plus\s+[^?]+?)\s+"
    r"(?:reconcile to|equal|match)\s+(?P<total>[^?.]+?)(?=\s+(?:on|for|in)\s+(?:both|each|all)\b|[?.]|$)",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"\(?(?:-?[$\u00a3\u20ac]?|[$\u00a3\u20ac] ?-)\d[\d,]*(?:\.\d+)?%?\)?"
)
CELL = re.compile(r"\S.*?(?= {2,}|$)")
CELL_REFERENCE = re.compile(r"([A-Z]{1,3})([1-9]\d*)")
NET_RESULT = frozenset(
    {
        "profit",
        "net profit",
        "net income",
        "net loss",
        "net income loss",
        "profit or loss",
        "net profit or loss",
        "net profit loss",
    }
)
OPERATORS = {
    "sum": "+",
    "difference": "-",
    "remainder": "-",
    "product": "x",
    "ratio": "/",
    "share": "/",
    "percent_change": "-",
}


# Preserve an explicitly requested operation when the question asks for one calculation.
def requested_operation(question: str) -> Literal["share", "ratio", "difference", ""]:
    if re.search(
        r"\b(?:percent(?:age)? change|percent(?:age)? (?:increase|decrease))\b",
        question,
        re.IGNORECASE,
    ):
        return ""  # A request can ask for both the amount and percent change.
    if re.search(
        r"\b(?:percentage|percent|share|fraction|proportion) of\b",
        question,
        re.IGNORECASE,
    ) and not re.search(
        r"\b(?:variance|above budget|below budget|in dollars)\b",
        question,
        re.IGNORECASE,
    ):
        return "share"
    if re.search(r"\b(?:divide|ratio)\b", question, re.IGNORECASE):
        return "ratio"
    if re.search(
        r"\b(?:variance|difference|how much (?:more|less|higher|lower)|subtract)\b",
        question,
        re.IGNORECASE,
    ):
        return "difference"
    return ""


class OperandSource(BaseModel):
    title: str = Field(max_length=160)
    year: int | None = Field(ge=1900, le=2099)
    month: int | None = Field(ge=1, le=12)
    day: int | None = Field(ge=1, le=31)
    period: Literal["any", "monthly", "cumulative"]
    query: str = Field(min_length=1, max_length=350)


class Measure(BaseModel):
    row: str = Field(min_length=1, max_length=120)
    column: str = Field(max_length=120)
    record: int | None = Field(ge=1, le=5)


class Equation(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    operation: Literal[
        "sum",
        "difference",
        "remainder",
        "product",
        "ratio",
        "share",
        "percent_change",
        "minimum",
        "maximum",
    ]
    inputs: list[int] = Field(min_length=2, max_length=8)
    precision: int = Field(ge=0, le=4)


class QuantityPlan(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    reuse_sources: bool
    sources: list[OperandSource] = Field(max_length=5)
    measures: list[Measure] = Field(min_length=1, max_length=12)
    equations: list[Equation] = Field(max_length=5)


class RowChoices(BaseModel):
    rows: list[int | None] = Field(max_length=12)


class Equations(BaseModel):
    equations: list[Equation] = Field(max_length=5)


# Apply a requested snapshot formula to every verified date before comparing results.
def snapshot_calculation(
    question: str, days: list[date]
) -> tuple[list[Measure], list[Equation]] | None:
    matches = {match[0].casefold() for match in SNAPSHOT_METRIC.finditer(question)}
    if len(matches) != 1 or not 2 <= len(days) <= 5 or len(set(days)) != len(days):
        return None
    metric = matches.pop()
    operation: Literal["difference", "ratio"] = (
        "ratio" if metric == "current ratio" else "difference"
    )
    measures = [
        Measure(row=field, column="", record=record)
        for record in range(1, len(days) + 1)
        for field in ("Total Current Assets", "Total Current Liabilities")
    ]
    equations = [
        Equation(
            label=f"{metric.title()} on {day:%Y-%m-%d}",
            operation=operation,
            inputs=[index * 2 + 1, index * 2 + 2],
            precision=2,
        )
        for index, day in enumerate(days)
    ]
    for direction, wording in (
        ("minimum", r"lowest|smallest|minim(?:um|al)"),
        ("maximum", r"highest|largest|maxim(?:um|al)"),
    ):
        if re.search(rf"\b(?:{wording})\b", question, re.IGNORECASE):
            equations.append(
                Equation(
                    label=f"{'Lowest' if direction == 'minimum' else 'Highest'} {metric}",
                    operation="minimum" if direction == "minimum" else "maximum",
                    inputs=[-index for index in range(1, len(days) + 1)],
                    precision=2,
                )
            )
    if FROM_FIRST_TO_LAST.search(question):
        equations.append(
            Equation(
                label=f"Change in {metric} from {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}",
                operation="difference",
                inputs=[-len(days), -1],
                precision=2,
            )
        )
    return measures, equations


# Build each dated comparison from the same metric in each verified original.
def period_calculation(
    question: str, days: list[date], plan: QuantityPlan
) -> tuple[list[Measure], list[Equation]] | None:
    if snapshot := snapshot_calculation(question, days):
        return snapshot
    if len(days) != 2 or days[0] == days[1]:
        return None
    if reconciliation := RECONCILE_SUM.search(question):
        addends = re.split(r"\s+plus\s+", reconciliation["terms"], flags=re.IGNORECASE)
        terms = [reconciliation["total"], *addends]
        if len(terms) > 6:
            return None
        fields = [f"Total {term.removeprefix('total ').strip()}" for term in terms]
        measures = [
            Measure(row=field, column="", record=record)
            for record in (1, 2)
            for field in fields
        ]
        equations = [
            Equation(
                label=f"Reconciliation difference on {day:%Y-%m-%d}",
                operation="remainder",
                inputs=list(
                    range(index * len(fields) + 1, (index + 1) * len(fields) + 1)
                ),
                precision=2,
            )
            for index, day in enumerate(days)
        ]
        return measures, equations
    if not PERIOD_CHANGE.search(question):
        return None
    unique = {
        (
            metric_phrase(field.row).removeprefix("total "),
            field.column.casefold(),
        ): field
        for field in plan.measures
        if metric_phrase(field.row).removeprefix("total ") in metric_phrase(question)
    }
    if not 1 <= len(unique) <= 3:
        return None
    measures = [
        field.model_copy(update={"record": record})
        for field in unique.values()
        for record in (1, 2)
    ]
    equations = [
        Equation(
            label=f"Change in {field.row} from {days[0]:%Y-%m-%d} to {days[1]:%Y-%m-%d}",
            operation="difference",
            inputs=[index + 2, index + 1],
            precision=2,
        )
        for index, field in zip(
            range(0, len(measures), 2), unique.values(), strict=True
        )
    ]
    if re.search(r"\bpercent(?:age)?\b", question, re.IGNORECASE):
        percent = [
            item.model_copy(
                update={"operation": "percent_change", "label": f"{item.label} (%)"}
            )
            for item in equations
        ]
        if not re.search(r"\b(?:dollars?|amount|absolute)\b", question, re.IGNORECASE):
            return measures, percent
        equations = [*equations, *percent]
    return measures, equations


# Validate model-selected arithmetic against explicit comparison language and dimensions.
def requested_equations(question: str, plan: QuantityPlan) -> list[Equation]:
    if len(plan.measures) == 2:
        roles = [
            source_terms(f"{measure.row} {measure.column}") & {"actual", "budget"}
            for measure in plan.measures
        ]
        if {frozenset(role) for role in roles} == {
            frozenset({"actual"}),
            frozenset({"budget"}),
        }:
            inputs = [roles.index({"actual"}) + 1, roles.index({"budget"}) + 1]
            decimal = Equation(
                label="Actual minus budget",
                operation="difference",
                inputs=inputs,
                precision=2,
            )
            if re.search(
                r"\b(?:above|below|under|over|variance|minus|difference)\b",
                question,
                re.IGNORECASE,
            ):
                if re.search(r"\bpercent(?:age)?\b", question, re.IGNORECASE):
                    percent = Equation(
                        label="Variance as a percentage of budget",
                        operation="percent_change",
                        inputs=inputs,
                        precision=2,
                    )
                    return (
                        [decimal, percent]
                        if re.search(
                            r"\b(?:dollars?|amount|absolute)\b", question, re.IGNORECASE
                        )
                        else [percent]
                    )
                return [decimal]

    if len(plan.measures) >= 3 and re.search(
        r"\bminus\b[^?!.]{1,90}\band\b", question, re.IGNORECASE
    ):
        lowered = question.casefold()
        positions = [
            (lowered.find(measure.row.casefold()), index)
            for index, measure in enumerate(plan.measures, 1)
        ]
        if all(position >= 0 for position, _ in positions) and len(
            {position for position, _ in positions}
        ) == len(positions):
            return [
                Equation(
                    label="Remaining difference",
                    operation="remainder",
                    inputs=[index for _, index in sorted(positions)],
                    precision=2,
                )
            ]

    if len(plan.measures) == 2 and (operation := requested_operation(question)):
        equation = next(
            (item for item in plan.equations if len(item.inputs) == 2), None
        )
        wording = metric_phrase(question)
        operand_positions = [
            wording.find(metric_phrase(item.row).removeprefix("total "))
            for item in plan.measures
        ]
        explicit_order = re.search(
            r"\bas (?:a )?(?:share|percent(?:age)?|proportion|fraction) of\b",
            question,
            re.IGNORECASE,
        )
        if (
            (explicit_order or operation == "ratio")
            and min(operand_positions) >= 0
            and operand_positions[0] != operand_positions[1]
        ):
            order = [1, 2] if operand_positions[0] < operand_positions[1] else [2, 1]
            return [
                Equation(
                    label=equation.label if equation else operation.title(),
                    operation=operation,
                    inputs=order,
                    precision=equation.precision if equation else 2,
                )
            ]
        if equation is not None:
            return [equation.model_copy(update={"operation": operation})]
    return plan.equations


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


@dataclass(frozen=True)
class Calculation:
    amount: CitedAmount
    sources: tuple[int, ...]
    display: str
    text: str


# Bind requested metric and dimension words to their own cells before ranking.
def measure_rows(
    measure: Measure,
    rows: list[CitedAmount],
    source: int | None,
    titles: list[str],
    question: str = "",
) -> list[int]:
    ignored = {
        "total",
        "for",
        "the",
        "of",
        "amount",
        "figure",
        "reported",
        "quarterly",
        "monthly",
        "ytd",
    }
    column = " ".join(measure.column.casefold().split())
    column_terms = source_terms(column)
    role = source_terms(measure.row) & {"actual", "budget"}
    metric = re.sub(r"\byear[ -]to[ -]date\b", "", measure.row, flags=re.IGNORECASE)
    requested = (
        source_terms(metric) - ignored - column_terms - MONTH_HINT.keys() - {"only"}
    )
    if not requested:
        return []
    net_result = (
        metric_phrase(re.sub(r"\breported\b", "", metric, flags=re.IGNORECASE))
        in NET_RESULT
    )
    queries = source_terms(question)
    request_words = f" {metric_phrase(question)} "
    dimensions: dict[int, set[str]] = {}
    parsed: list[tuple[int, CitedAmount, str, str]] = []
    for index, item in enumerate(rows, 1):
        if source is not None and source != item.source:
            continue
        fields = [value.strip() for value in item.line.strip("|").split("|")]
        heading = fields[1].casefold() if len(fields) > 2 else ""
        dimensions.setdefault(item.source, set()).update(source_terms(heading))
        parsed.append((index, item, fields[0], heading))

    named_totals = {
        phrase
        for _, _, label, _ in parsed
        if (phrase := metric_phrase(label)).startswith("total ")
        and f" {phrase} " in request_words
    }
    budget_comparison = {"budget", "variance"} & queries and (
        source_terms(measure.row) | column_terms
    ) & {"actual", "budget"}
    if budget_comparison and len(named_totals) == 1:
        requested = source_terms(next(iter(named_totals))) - ignored

    candidates: list[tuple[tuple[int, int, int, int, int, int], int]] = []
    for index, amount, label, heading in parsed:
        headings = source_terms(heading)
        qualifier = column_terms
        if len(role) == 1:
            qualifier = (qualifier - {"actual", "budget"}) | role
        if (
            qualifier == {"actual"}
            and not {"actual", "budget"} & dimensions[amount.source]
        ):
            qualifier = set()
        wanted = requested - headings
        terms = source_terms(label) - ignored
        if net_result and metric_phrase(label) in NET_RESULT:
            terms = wanted
        compatible_total = (
            "expense" in wanted
            and bool(terms)
            and (wanted - {"expense"} == terms)
            and label.casefold().startswith("total ")
        )
        if not wanted or (not wanted <= terms and not compatible_total):
            continue
        if (
            qualifier
            and not qualifier <= headings
            and not (
                heading == "total"
                and qualifier <= source_terms(titles[amount.source - 1])
            )
        ):
            continue
        named_axis = headings - ignored - {"actual", "budget", "current", "prior"}
        exact_context = (
            bool(named_axis)
            and named_axis <= queries
            and not heading.startswith("total ")
        )
        axes = {part.strip() for part in heading.split("/")}
        axis = " ".join(sorted(qualifier))
        exact_column = (
            2 if column and column == heading else int(bool(axis) and axis in axes)
        )
        score = (
            exact_column,
            int(wanted == terms),
            int(exact_context),
            int(heading == "total"),
            -len(terms - wanted),
            int(label.casefold().startswith("total ")),
        )
        candidates.append((score, index))
    highest = max((score for score, _ in candidates), default=None)
    return [index for score, index in candidates if score == highest][:6]


# Match literal rows across harmless punctuation, pluralization, and separator differences.
def metric_phrase(text: str) -> str:
    words = (next(iter(source_terms(word))) for word in re.findall(r"[^\W\d_]+", text))
    return " ".join(word for word in words if word not in {"for", "of", "the"})


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


# Assign IDs only to original rows that identify one unambiguous numeric amount.
def quantity_rows(text: str, source: int) -> list[CitedAmount]:
    rows = []
    for line in dict.fromkeys(text.splitlines()):
        if not any(character.isalpha() for character in line) or (
            "|" not in line
            and re.fullmatch(r"[A-Za-z -]+ (?:19|20)\d{2}", line.strip())
        ):
            continue
        if "|" in line:
            matches = [
                part.strip()
                for part in line.split("|")
                if NUMBER.fullmatch(part.strip())
            ]
        else:
            matches = [
                match[0]
                for match in re.finditer(rf"(?<!\w){NUMBER.pattern}(?!\w)", line)
            ]
        if len(matches) == 1 and (amount := amount_value(matches[0])) is not None:
            rows.append(CitedAmount(*amount, source, line))
    return rows


# Evaluate a formula over verified original rows and earlier calculated results.
def evaluate_equation(
    equation: Equation,
    amounts: list[CitedAmount],
    earlier: dict[int, Calculation],
) -> Calculation | None:
    inputs = equation.inputs
    if len(set(inputs)) != len(inputs) or not all(
        1 <= key <= len(amounts) if key > 0 else -key in earlier for key in inputs
    ):
        return None
    known = [amounts[key - 1] if key > 0 else earlier[-key].amount for key in inputs]
    sources = tuple(
        dict.fromkeys(
            source
            for key in inputs
            for source in (
                (amounts[key - 1].source,) if key > 0 else earlier[-key].sources
            )
        )
    )
    operation = equation.operation
    units = {item.unit for item in known if item.unit}
    if len(units) > 1 or (
        operation not in {"sum", "product", "remainder", "minimum", "maximum"}
        and len(known) != 2
    ):
        return None
    values = [item.number for item in known]
    unit = next(iter(units), "")
    formula = ""
    with localcontext() as context:
        context.prec = 80
        first, second = values[:2]
        if operation in {"ratio", "share", "percent_change"} and not second:
            return None
        if operation in {"difference", "remainder"}:
            result = first - sum(values[1:], Decimal(0))
        elif operation == "sum":
            result = sum(values, Decimal(0))
        elif operation == "product":
            result = Decimal(1)
            for value in values:
                result *= value
        elif operation == "percent_change":
            result = (first - second) / second * 100
        elif operation in {"minimum", "maximum"}:
            result = min(values) if operation == "minimum" else max(values)
            formula = ", ".join(
                amount.line for amount in known if amount.number == result
            )
        else:
            result = first / second * (100 if operation == "share" else 1)
        if result.adjusted() > 36:
            return None
        displayed = result.quantize(
            Decimal(1).scaleb(-equation.precision), rounding=ROUND_HALF_UP
        )
    rendered = [
        amounts[key - 1].display() if key > 0 else earlier[-key].display
        for key in inputs
    ]
    if not formula and operation in OPERATORS:
        formula = f" {OPERATORS[operation]} ".join(rendered)
    if operation in {"share", "percent_change"}:
        formula = (
            f"({formula}) x 100"
            if operation == "share"
            else f"({formula}) / {rendered[1]} x 100"
        )
        unit = "%"
    elif operation == "ratio":
        unit = ""
    label = re.sub(r"\[\d+\]|[\r\n]", "", equation.label).strip()
    answer = CitedAmount(displayed, unit, 0, label).display()
    references = " ".join(f"[{ref}]" for ref in sources)
    explanation = f" ({formula})" if formula else ""
    return Calculation(
        CitedAmount(result, unit, 0, label),
        sources,
        answer,
        f"**{label}** = **{answer}**{explanation} {references}",
    )


# Render arithmetic on original rows without intermediate calculations.
def calculate(equation: Equation, amounts: list[CitedAmount]) -> str:
    result = evaluate_equation(equation, amounts, {})
    return result.text if result else ""
