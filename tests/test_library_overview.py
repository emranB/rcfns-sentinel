'''Check library summaries against known file and index counts.'''

import unittest

from src.backend.library_overview import build_overview


class LibraryOverviewTests(unittest.TestCase):
    # Group real inventory paths beyond their shared wrapper and preserve root files.
    def test_groups_folders_and_reports_separate_index_counts(self) -> None:
        paths = (
            "data/backup/Finance/2025/Reconciliation Reports/report.pdf",
            "data/backup/Finance/2024/Reconciliation Reports/report.pdf",
            "data/backup/Operations/plan.docx",
            "data/backup/training.mov",
        )
        overview = build_overview([*paths, paths[0]])

        self.assertEqual(overview.original_count, 4)
        self.assertEqual(overview.root_count, 1)
        answer = overview.format(2, 5)
        self.assertIn("2 of 5 items selected for indexing are searchable", answer)
        self.assertIn("**Finance** has 2 files", answer)
        self.assertIn("Subfolders include Reconciliation Reports", answer)
        self.assertIn("**Operations** has 1 file", answer)
        self.assertIn("**Library root** holds 1 file", answer)
        self.assertNotIn("backup", answer)
        self.assertIn(
            "Finance (2) and Operations (1)", overview.format(2, 5, concise=True)
        )
        self.assertEqual(overview, build_overview(reversed(paths)))

    # Summarize a flat collection by extension and support a one-line reply.
    def test_flat_library_and_concise_reply(self) -> None:
        overview = build_overview(
            [
                "source/Evidence.PDF",
                "source/Notes.pdf",
                "source/Index.txt",
                "source/LICENSE",
            ]
        )
        self.assertIn("**PDF** has 2 files", overview.format(4, 4))
        self.assertIn("**No file extension** has 1 file", overview.format(4, 4))
        concise = overview.format(4, 4, concise=True)
        self.assertNotIn("\n", concise)
        self.assertIn("4 original files", concise)
        self.assertIn("4 of 4 items", concise)

    # Keep untrusted directory punctuation from becoming formatting or a link.
    def test_escapes_folder_names(self) -> None:
        overview = build_overview(["source/[Ops]/plan.txt", "source/mail.txt"])
        self.assertIn(r"**\[Ops\]**", overview.format(1, 2))
