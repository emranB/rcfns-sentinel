'''Exercise calculations grounded in independently scoped original values.'''

import unittest
from decimal import Decimal
from unittest.mock import patch

from src.backend.quantitative import (
    Equation,
    Measure,
    calculate,
    measure_rows,
    numeric_layout,
    numeric_workbook,
    quantity_rows,
)
from src.backend.tabular import Worksheet


class QuantityTests(unittest.TestCase):
    # Select the exact field and column; an unrelated amount cannot fill a missing operand.
    def test_measure_identity(self) -> None:
        rows = quantity_rows(
            "Total Transportation | Total for East | $215.00\n"
            "Total Transportation | East | $160.00\n"
            "Total Transportation | West | $335.00\n"
            "Transportation Permits | East | $45.00\n"
            "Cash Balance | East | $650.00",
            1,
        )
        field = Measure(row="Transportation", column="East", record=1)
        self.assertEqual(measure_rows(field, rows, 1, ["Transit Summary"]), [2])
        self.assertEqual(
            measure_rows(
                field.model_copy(update={"column": "Total for East"}),
                rows,
                1,
                ["Transit Summary"],
            ),
            [1],
        )
        self.assertEqual(
            measure_rows(
                Measure(row="Cash Payments", column="East", record=1),
                rows,
                1,
                ["Transit Summary"],
            ),
            [],
        )

    # Preserve merged worksheet headers and distinguish cached zero from a blank.
    def test_workbook_uses_group_and_row_labels(self) -> None:
        sheet = Worksheet(
            "Estimate",
            [
                {"B4": "North", "D4": "South"},
                {"B5": "Actual", "C5": "Budget", "D5": "Actual", "E5": "Budget"},
                {"A6": "Labor", "B6": "0", "C6": "950.6", "E6": "600.0"},
            ],
            ("B4:C4", "D4:E4"),
        )
        with patch("src.backend.quantitative.worksheet_tables", return_value=[sheet]):
            result = numeric_workbook(b"", "North labor actual compared with budget")
        self.assertIn("Labor | North / Actual | Estimate!B6 | 0.00", result)
        self.assertIn("Labor | North / Budget | Estimate!C6 | 950.60", result)
        self.assertNotIn("Labor | South / Actual", result)

    # Keep the requested program separate from its own subtotal and the overall total.
    def test_layout_preserves_column_identity(self) -> None:
        headline = "Contract Spending\nAs of October 2027\n"
        header = f"{'Distribution':25}{'West':>14}{'East':>14}{'Total':>14}"
        row = f"{'Total spend':25}{'12.73':>14}{'96.22':>14}{'$108.95':>14}"
        normalized = numeric_layout(f"{headline}{header}\n{row}", "total spend West")
        self.assertIn("Total spend | West | 12.73", normalized)
        self.assertIn("Total spend | East | 96.22", normalized)
        self.assertIn("Total spend | Total | $108.95", normalized)
        verified = quantity_rows(normalized, 1)
        self.assertEqual(
            next(row.number for row in verified if "West" in row.line), Decimal("12.73")
        )
        self.assertEqual(quantity_rows("Total spend | 12.73 | 96.22", 1), [])

    # Compute fresh differences and percentages while rejecting incomplete or mismatched inputs.
    def test_computation_uses_only_literal_inputs(self) -> None:
        amounts = [
            *quantity_rows("Fuel spend | Actual | $1,000.00", 1),
            *quantity_rows("Fuel spend | Actual | $1,250.00", 2),
        ]
        self.assertEqual(amounts[1].number, Decimal("1250.00"))
        difference = Equation(
            label="Increase", operation="difference", inputs=[2, 1], precision=2
        )
        change = Equation(
            label="Percent change",
            operation="percent_change",
            inputs=[2, 1],
            precision=2,
        )
        self.assertIn("$250.00", calculate(difference, amounts))
        self.assertIn("25.00%", calculate(change, amounts))
        self.assertIn("[2] [1]", calculate(change, amounts))
        self.assertEqual(calculate(difference, amounts[1:]), "")
        self.assertEqual(
            calculate(difference.model_copy(update={"inputs": [2, 2]}), amounts), ""
        )
