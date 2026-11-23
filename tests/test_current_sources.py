'''Verify that dated source suggestions preserve topic and time scope.'''

import unittest
from datetime import date

from src.backend.current_sources import comparison_document_ids, latest_document_ids


class CurrentSourceTests(unittest.TestCase):
    # Keep records from both years, including older entries that fall outside a recent lookup.
    def test_comparison_covers_each_year_and_prefers_a_consistent_series(self) -> None:
        documents = [
            ("recent", "Warehouse/Inventory/2025/20251202InventoryReport.pdf"),
            ("early", "Warehouse/Inventory/2025/Inventory report 01.10.2025.pdf"),
            ("previous", "Warehouse/Inventory/2024/20241128InventoryReport.pdf"),
            ("older", "Warehouse/Inventory/2024/20240303InventoryReport.pdf"),
            ("agenda", "Warehouse/Inventory/2025/20251212InventoryAgenda.docx"),
            ("template", "Warehouse/Inventory/2024/20240601InventoryTemplate.docx"),
            ("unrelated", "Contacts/2024/20240109AdvisoryReport.docx"),
        ]
        result = comparison_document_ids(
            "List inventory totals in 2025 vs 2024", documents, today=date(2026, 1, 1)
        )
        self.assertEqual(list(result), [2025, 2024])
        self.assertEqual(
            result, {2025: ["early", "recent"], 2024: ["older", "previous"]}
        )

    # Match the meeting folder when the filename uses an abbreviation.
    def test_finds_current_records_by_original_path(self) -> None:
        folder = "Board/Board Meeting Records/2026/November/"
        documents = [
            ("new-minutes", folder + "20261116_Minutes_BoD.docx"),
            ("older-record", "Board/Records/2026-10-24 board roster.xlsx"),
            ("future", folder + "2026-11-23 minutes.docx"),
            ("agenda", folder + "20261122 agenda.docx"),
            ("draft", folder + "20261122 draft minutes.docx"),
            ("template", folder + "20261122 record template.docx"),
            ("archived", "Board/Archives/20261122 board membership.pdf"),
            ("invalid", folder + "20261132 minutes.docx"),
            ("undated", folder + "Board records.docx"),
            ("unrelated", "Operations/20261122 inventory.pdf"),
        ]
        self.assertEqual(
            latest_document_ids(
                "How many board members do we have?",
                documents,
                today=date(2026, 11, 22),
            ),
            ["new-minutes", "older-record"],
        )

    # Preserve explicit historical scope in cross-folder questions.
    def test_preserves_questions_with_an_explicit_or_past_period(self) -> None:
        cross_folder = (
            "In the August 2025 Visitor Services website outline, where did the draft put Parking? "
            "Using the Updated Visitor Parking Policy, explain the requirements for day passes. "
            "Cite both originals and keep the answer under 90 words."
        )
        documents = [("record", "Board/20261116 Board minutes.txt")]
        for question in (
            cross_folder,
            "What are the latest board totals for 2025?",
            "How many board members do we have as of March 15?",
            "What was the board count last month?",
            "How many board members did we have?",
            "Explain how the board works.",
        ):
            with self.subTest(question=question):
                self.assertEqual(latest_document_ids(question, documents), [])

    # Include a distinct subject even when one subject has many newer records.
    def test_limits_and_orders_results_with_distinct_topic_coverage(self) -> None:
        documents = [
            (f"warehouse-{day}", f"Warehouse/202611{day:02d} warehouse report.txt")
            for day in range(5, 10)
        ] + [
            ("scheduling", r"HR\Scheduling\2026-11-01 schedule.txt"),
            ("warehouse-9", "Warehouse/20261109 warehouse report.txt"),
            ("tie-9", "Warehouse/20261109 warehouse update.txt"),
        ]
        question = "What are the current warehouse and scheduling updates?"
        as_of = date(2026, 11, 22)
        expected = ["warehouse-9", "scheduling", "tie-9", "warehouse-8"]
        self.assertEqual(
            latest_document_ids(question, documents, today=as_of), expected
        )
        self.assertEqual(
            latest_document_ids(question, reversed(documents), today=as_of), expected
        )
