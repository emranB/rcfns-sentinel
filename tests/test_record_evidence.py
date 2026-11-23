'''Exercise source selection and proof checks with unrelated synthetic records.'''

import unittest
from datetime import date
from io import BytesIO
from zipfile import ZipFile

from src.backend.grounding import ClaimAnswer, EvidenceText, values, verified_answer
from src.backend.record_lookup import dated_references, explicit_ids, has_record_date
from src.backend.tabular import count_named_rows, requested_table


class RecordEvidenceTests(unittest.TestCase):
    # A real quotation from one file cannot be attributed to another type of source.
    def test_rejects_proof_credited_to_a_different_document_type(self) -> None:
        line = "The new field kit weighs 8 kilograms."
        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {
                        "text": "The presentation describes a field kit weighing 8 kilograms.",
                        "proofs": [{"source": 1, "quote": line}],
                    }
                ],
                "incomplete": False,
            }
        )
        self.assertEqual(
            verified_answer(draft, [EvidenceText("Field minutes.docx", line)]), ""
        )
        self.assertIn("[1]", verified_answer(draft, [EvidenceText("Field.pptx", line)]))

    # A shortened currency amount and its expanded value refer to the same quantity.
    def test_verifies_scaled_currency_without_converting_ordinary_units(self) -> None:
        self.assertEqual(values("about $18.2K"), values("$18,200.00"))
        self.assertNotEqual(values("18.2m of rope"), values("18,200,000 metres"))

    # Scope each explicit date to the right type and ignore a similar agenda.
    def test_dated_minutes_keep_each_requested_meeting(self) -> None:
        originals = [
            ("jan", "Wetlands Council/2022/20220112_CouncilMinutes.docx"),
            ("feb", "Wetlands Council/2022/20220215_CouncilMinutes.docx"),
            ("agenda", "Wetlands Council/2022/20220215_CouncilAgenda.docx"),
        ]
        references = dated_references(
            "Compare January 12 and February 15, 2022 Wetlands Council minutes.",
            originals,
        )
        self.assertEqual(
            [(ref.day, ref.exact) for ref in references],
            [
                (date(2022, 1, 12), ("jan",)),
                (date(2022, 2, 15), ("feb",)),
            ],
        )

    # Do not mistake the year in a reporting date for part of another filename.
    def test_exact_titles_exclude_a_different_record_date(self) -> None:
        originals = [
            ("other", "Warehouse/2024 Balance Sheet.pdf"),
            ("plan", "Warehouse/InventoryPlanV3 (1).docx"),
        ]
        self.assertEqual(
            explicit_ids(
                "Compare the March 3, 2024 Balance Sheet and InventoryPlanV3", originals
            ),
            ["plan"],
        )
        self.assertTrue(has_record_date("As of March 3, 2024", date(2024, 3, 3)))

    # Rejoin a broken row while preserving the two distinct source periods.
    def test_table_uses_headers_and_split_numeric_row(self) -> None:
        rows = requested_table(
            "Compare Stock Value in both date columns",
            [
                (
                    "| | AS OF OCTOBER 1, 2023 | AS OF OCTOBER 1, 2022 |\n"
                    "| Stock Value | | |\n| | $2,199.60 | $1,205.20 |"
                ),
            ],
        )
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(
            [str(amount) for amount in rows[1][0].values], ["2199.60", "1205.20"]
        )

    # Resolve the named worksheet and count entries beyond an arbitrary chunk boundary.
    def test_counts_all_name_rows_in_named_sheet(self) -> None:
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                '<workbook xmlns:r="urn:r"><sheets><sheet name="Equipment" r:id="r1"/></sheets></workbook>',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<Relationships><Relationship Id="r1" Target="worksheets/sheet9.xml"/></Relationships>',
            )
            values = [
                "Name",
                "Trail Beacon",
                "Hill Radio",
                "Depot Rover",
                "Last checked in 2023.",
            ]
            rows = "".join(
                f'<row><c r="B{i}" t="inlineStr"><is><t>{value}</t></is></c></row>'
                for i, value in enumerate(values, 1)
            )
            archive.writestr(
                "xl/worksheets/sheet9.xml",
                f"<worksheet><sheetData>{rows}</sheetData></worksheet>",
            )
        self.assertEqual(count_named_rows(output.getvalue(), "Equipment"), 3)
        self.assertIsNone(count_named_rows(output.getvalue(), "Other"))

    # Reject invented numbers, missing quotations and departures inferred from regrets.
    def test_rejects_unproven_claims_without_discarding_supported_facts(self) -> None:
        proof = "Regrets: Ari Green. Approved 16 emergency kits."
        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {
                        "text": "Ari Green resigned.",
                        "proofs": [{"source": 1, "quote": proof}],
                    },
                    {
                        "text": "The council approved 60 emergency kits.",
                        "proofs": [{"source": 1, "quote": proof}],
                    },
                    {
                        "text": "The council approved 16 emergency kits.",
                        "proofs": [{"source": 1, "quote": proof}],
                    },
                ],
                "incomplete": False,
            }
        )
        result = verified_answer(draft, [EvidenceText("Council Minutes", proof)])
        self.assertIn("16 emergency kits. [1]", result)
        self.assertNotIn("resigned", result)
        self.assertNotIn("60 emergency kits", result)
