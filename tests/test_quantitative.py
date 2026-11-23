'''Exercise calculations grounded in independently scoped original values.'''

import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import patch

from src.backend.quantitative import (
    Equation,
    Measure,
    QuantityPlan,
    calculate,
    measure_rows,
    numeric_layout,
    numeric_workbook,
    period_calculation,
    quantity_rows,
    requested_equations,
    requested_operation,
)
from src.backend.tabular import Worksheet


class QuantityTests(unittest.TestCase):
    # A model that finds one period cannot collapse a comparison across two periods.
    def test_period_change_expands_metrics_over_verified_dates(self) -> None:
        days = [date(2014, 10, 15), date(2017, 10, 15)]
        plan = QuantityPlan(
            question="Calculate changes",
            reuse_sources=True,
            sources=[],
            measures=[
                Measure(row="Total Tickets", column="", record=1),
                Measure(row="Overtime", column="", record=1),
            ],
            equations=[],
        )
        result = period_calculation(
            "Calculate the year-over-year change in total tickets and the change in overtime.",
            days,
            plan,
        )
        assert result is not None
        measures, equations = result
        self.assertEqual([measure.record for measure in measures], [1, 2, 1, 2])
        self.assertEqual([item.inputs for item in equations], [[2, 1], [4, 3]])

    # Reconcile the requested components against the total separately in each dated record.
    def test_period_reconciliation_checks_both_originals(self) -> None:
        plan = QuantityPlan(
            question="Reconcile the records",
            reuse_sources=True,
            sources=[],
            measures=[Measure(row="Total Participants", column="", record=1)],
            equations=[],
        )
        result = period_calculation(
            "Do adults plus children reconcile to total participants on both dates?",
            [date(2014, 10, 15), date(2017, 10, 15)],
            plan,
        )
        assert result is not None
        measures, equations = result
        self.assertEqual([measure.record for measure in measures], [1, 1, 1, 2, 2, 2])
        self.assertEqual([item.inputs for item in equations], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(measures[0].row, "Total participants")

    # A malformed model equation cannot override the requested budget variance or reconciliation.
    def test_plan_cannot_change_explicit_arithmetic(self) -> None:
        wrong = Equation(label="Combined", operation="sum", inputs=[1, 2], precision=2)
        plan = QuantityPlan(
            question="Calculate payroll variance",
            reuse_sources=False,
            sources=[],
            measures=[
                Measure(row="Payroll Budget", column="", record=None),
                Measure(row="Payroll Actual", column="", record=None),
            ],
            equations=[wrong],
        )
        chosen = requested_equations(
            "Give payroll above budget in dollars and as a percentage of budget.", plan
        )
        self.assertEqual(
            [(item.operation, item.inputs) for item in chosen],
            [("difference", [2, 1]), ("percent_change", [2, 1])],
        )
        plan.measures = [
            Measure(row=label, column="", record=None)
            for label in ("Costs", "Revenue", "Adjustments")
        ]
        (remaining,) = requested_equations(
            "Calculate revenue minus costs and adjustments.", plan
        )
        self.assertEqual(
            (remaining.operation, remaining.inputs), ("remainder", [2, 1, 3])
        )

    # Select the exact field and column; an unrelated amount cannot fill a missing operand.
    def test_measure_identity(self) -> None:
        rows = quantity_rows(
            "Total Transportation | Total for East | $215.00\n"
            "Total Transportation | East | $160.00\n"
            "Total Transportation | West | $335.00\n"
            "Transportation Permits | East | $45.00\n"
            "Cash Balance | East | $650.00\n"
            "Total for Travel | East | $50.00",
            1,
        )
        field = Measure(row="Transportation", column="East", record=1)
        self.assertEqual(measure_rows(field, rows, 1, ["Transit Summary"]), [2])
        travel = field.model_copy(update={"row": "Total Travel Expenses"})
        self.assertEqual(measure_rows(travel, rows, 1, ["Transit Summary"]), [6])
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
        unspecified = field.model_copy(update={"column": "Actual"})
        self.assertEqual(
            measure_rows(
                unspecified, rows, 1, ["Transit Summary"], "East transportation"
            ),
            [2],
        )
        workbook = quantity_rows(
            "Total Payroll | West / Actual | Costs!D18 | 415.00\n"
            "Total Payroll | East / Actual | Costs!H18 | 997.00\n"
            "Total Payroll | East / Budget | Costs!I18 | 600.00\n"
            "Total Payroll | East / % of Budget | Costs!J18 | 166.17%\n"
            "Total Expenses | East / Actual | Costs!H42 | 1,311.00\n"
            "Total Expenses | East / Budget | Costs!I42 | 1,440.00",
            1,
        )
        payroll = Measure(row="East Actual Payroll", column="Actual", record=1)
        self.assertEqual(measure_rows(payroll, workbook, 1, ["Employee Costs"]), [2])
        budget = Measure(row="East Payroll Budget", column="", record=1)
        self.assertEqual(measure_rows(budget, workbook, 1, ["Employee Costs"]), [3])
        follow_up = "In the same workbook, what was the East total expense variance?"
        self.assertEqual(
            measure_rows(budget, workbook, 1, ["Employee Costs"], follow_up), [6]
        )

    # Keep a percentage request distinct from a ratio or a compound variance question.
    def test_preserves_explicit_operation(self) -> None:
        self.assertEqual(
            requested_operation("Express attended visits as a share of all visits."),
            "share",
        )
        self.assertEqual(
            requested_operation(
                "What was the ratio of tendered contracts to signed contracts?"
            ),
            "ratio",
        )
        self.assertEqual(
            requested_operation(
                "Give the change in dollars and as a percentage of budget."
            ),
            "",
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
