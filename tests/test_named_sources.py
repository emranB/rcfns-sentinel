'''Verify that explicit document names reliably narrow independent searches.'''

import unittest

from src.backend.named_sources import named_document_ids


class NamedSourceTests(unittest.TestCase):
    # Match distinct original titles and select the stated date despite reordered words.
    def test_matches_titles_from_multiple_folders(self) -> None:
        documents = [
            (
                "current",
                r"Operations\Website\Regional Website Menu Tree - May 2026.docx",
            ),
            ("draft", r"Operations\Website\Regional Website Menu Tree - May 2025.docx"),
            (
                "plan",
                "Fundraising/Community Partnership Strategy for Local Schools.pdf",
            ),
            ("partial", "Fundraising/Local Schools.pdf"),
        ]
        result = named_document_ids(
            "In the May 2025 Regional Website menu-tree, where do partner links go? "
            "Using Community Partnership Strategy for Local Schools, give two examples.",
            documents,
        )
        self.assertEqual(result, ["draft", "plan"])

    # Abstain when a phrase fits equally well across copies or versions.
    def test_leaves_ambiguous_versions_to_content_search(self) -> None:
        documents = [
            ("draft", "Internal/Rural Investment Impact Report Draft.pdf"),
            ("final", "Public/Rural Investment Impact Report Final.pdf"),
            ("fallback", "Reference/2024 Investment Impact Report Notes.pdf"),
        ]
        self.assertEqual(
            named_document_ids("Summarize Rural Investment Impact Report.", documents),
            [],
        )

    # Prefer the exact title when its copy is named separately.
    def test_prefers_the_exact_title_to_a_copy(self) -> None:
        documents = [
            ("copy", "Backup/Regional Energy Costs Forecast - Copy.xlsx"),
            ("original", "Finance/Regional Energy Costs Forecast.xlsx"),
        ]
        self.assertEqual(
            named_document_ids("What is Regional Energy Costs Forecast?", documents),
            ["original"],
        )

    # Leave short topic mentions to regular content search.
    def test_ignores_generic_topic_fragments(self) -> None:
        documents = [("manual", "Board/Board Members and Committee Policy.pdf")]
        self.assertEqual(
            named_document_ids("How many board members are there?", documents), []
        )

    # Bound the number of targeted searches and retain the question's source order.
    def test_limits_the_number_of_targeted_sources(self) -> None:
        documents = [
            ("one", "Rural Energy Operating Cost.xlsx"),
            ("two", "Public Transport Passenger Survey.pdf"),
            ("three", "Urban Tree Canopy Monitoring Plan.pdf"),
            ("four", "Annual Finance Overview.pdf"),
        ]
        self.assertEqual(
            named_document_ids(
                "Compare Annual Finance Overview and Public Transport Passenger Survey "
                "with Rural Energy Operating Cost and Urban Tree Canopy Monitoring Plan.",
                documents,
            ),
            ["two", "one", "three"],
        )
