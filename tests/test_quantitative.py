'''Exercise calculations grounded in independently scoped original values.'''

import unittest
from decimal import Decimal
from unittest.mock import patch

from src.backend.grounding import EvidenceText
from src.backend.quantitative import (
    Equation,
    NumericInput,
    calculate,
    cited_amount,
    numeric_layout,
    numeric_workbook,
)
from src.backend.tabular import Worksheet


class QuantityTests(unittest.TestCase):
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
        verified = [EvidenceText("2027 contract summary", normalized)]
        self.assertIsNone(
            cited_amount(
                NumericInput(
                    source=1, quote="Total spend | West | 12.73", amount="96.22"
                ),
                verified,
            )
        )

    # Compute fresh differences and percentages while rejecting incomplete or mismatched inputs.
    def test_computation_uses_only_literal_inputs(self) -> None:
        evidence = [
            EvidenceText("before", "Fuel spend | Actual | $1,000.00"),
            EvidenceText("after", "Fuel spend | Actual | $1,250.00"),
        ]
        amounts = [
            cited_amount(
                NumericInput(source=i, quote=item.content, amount=amount), evidence
            )
            for i, item, amount in zip(
                (1, 2), evidence, ("1000.00", "1250.00"), strict=True
            )
        ]
        assert amounts[1] is not None
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
        self.assertEqual(calculate(difference, [None, amounts[1]]), "")
        self.assertEqual(
            calculate(difference.model_copy(update={"inputs": [2, 2]}), amounts), ""
        )
