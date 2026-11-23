'''Verify report coverage and arithmetic without relying on generated claims.'''

import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx

from src.backend import server
from src.backend.ragflow import RagflowSession
from src.backend.statement_checks import parse_statement, statement_years


# Build a report with an exact printed label and a right-aligned grand total column.
def report(period: str, rows: dict[str, str]) -> str:
    return "\n".join(
        [
            "Profit and Loss by Class",
            period,
            f"{'Distribution Account':30}{'Class A':>12}{'Total':>16}",
            *[f"{label:30}{'$5.00':>12}{amount:>16}" for label, amount in rows.items()],
        ]
    )


class StatementTests(unittest.IsolatedAsyncioTestCase):
    # Read actual report dates and total columns while ignoring expense subtotals.
    def test_grand_total_with_cost_and_other_adjustments(self) -> None:
        text = report(
            "July 1, 2023 - June 30, 2024",
            {
                "Total Income": "$600.00",
                "Total Cost of Sales": "$20.00",
                "Gross Profit": "$580.00",
                "Total for Payroll Expenses": "$45.00",
                "Total Expenses": "$500.00",
                "Net Operating Income": "$80.00",
                "Total Other Income": "($25.00)",
                "Total Other Expenses": "$10.00",
                "Net Other Income": "($35.00)",
                "Profit": "$45.00",
            },
        )
        parsed = parse_statement(text)
        assert parsed is not None
        self.assertEqual(parsed.start, date(2023, 7, 1))
        self.assertEqual(parsed.end, date(2024, 6, 30))
        self.assertEqual(parsed.rows["expenses"], Decimal("500.00"))
        self.assertEqual(parsed.adjustment(), Decimal("-55.00"))
        self.assertEqual(parsed.differences(), {})

    # Reconcile original reports across folders and flag a real discrepancy and a missing total.
    async def test_same_annual_check_in_both_modes(self) -> None:
        question = "For 2024, do the numbers in the P/L reports add up?"
        self.assertEqual(statement_years(question), (2024,))
        files = {
            "parent": "Reports/2030/P&#x3a;L by Class.pdf",
            "semantic": "Another folder/AR-09.pdf",
            "missing": "Archived/Income Statement.pdf",
            "wrong_period": "Reports/2024/Profit and Loss.pdf",
            "other_type": "Public/Profit Sharing Notes.pdf",
        }
        rows = {"Total for Income": "$20.00", "Total Expenses": "$2.10"}
        texts = {
            "parent": report("January 2024", {**rows, "Profit": "$17.90"}),
            "semantic": report("January-February, 2024", {**rows, "Profit": "$18.00"}),
            "missing": report("March 2024", rows),
            "wrong_period": report("December 2023", {**rows, "Profit": "$17.90"}),
            "other_type": "Profit Sharing Notes\nSeptember 2024\n",
        }

        # Treat each synthetic original as the plain native text after extraction.
        async def archive(
            _state: RagflowSession, document_id: str, **_options: str | int
        ) -> bytes:
            return texts[document_id].encode()

        async with httpx.AsyncClient() as client:
            state = RagflowSession(client, documents=files)
            with (
                patch.object(server, "original_archive", side_effect=archive),
                patch.object(server, "pdf_text", bytes.decode),
                patch.object(
                    server,
                    "retrieve",
                    AsyncMock(
                        return_value=[
                            server.RetrievedChunk(document_id="semantic", content="P/L")
                        ]
                    ),
                ),
                patch.object(server, "generate", new_callable=AsyncMock) as generate,
            ):
                for mode in ("standard", "agentic"):
                    payload = server.Question.model_validate(
                        {"question": question, "mode": mode}
                    )
                    response = await server.respond_to_question(
                        state, question, payload
                    )
                    self.assertEqual(response.mode_used, mode)
                    self.assertIn("1 of 2 reports checked", response.answer)
                    self.assertIn("net +0.10", response.answer)
                    self.assertIn("missing grand totals for net", response.answer)
                    self.assertEqual(
                        [source.path for source in response.sources],
                        [files[key] for key in ("parent", "semantic", "missing")],
                    )
                    self.assertEqual(
                        response.period_subject,
                        "do the numbers in the P/L reports add up",
                    )
                generate.assert_not_awaited()
