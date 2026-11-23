'''Exercise source selection and proof checks with unrelated synthetic records.'''

import unittest
from datetime import date
from io import BytesIO
from zipfile import ZipFile

from src.backend.grounding import (
    ClaimAnswer,
    EvidenceText,
    field_evidence,
    labeled_passage,
    quantitative_passage,
    requested_fields,
    requested_part_score,
    requested_parts,
    values,
    verified_answer,
)
from src.backend.record_lookup import dated_references, explicit_ids, has_record_date
from src.backend.tabular import count_named_rows, requested_table, table_person_names


class RecordEvidenceTests(unittest.TestCase):
    # Approval evidence must match the action's object and any explicit threshold.
    def test_authority_uses_the_requested_action_and_money_boundary(self) -> None:
        recommendation = "The panel approves the facility recommendations."
        self.assertEqual(
            labeled_passage(
                "Who approves the facility recommendations in the 2021 handbook?",
                f"The 2021 handbook is approved by the trustees.\n{recommendation}",
            ),
            recommendation,
        )
        lower = "Purchases under $3,500: Services Committee"
        self.assertEqual(
            labeled_passage(
                "Who reviews purchases under $3,500 in the handbook?",
                "Purchases over $3,500: Finance Committee approval.\n" + lower,
            ),
            lower,
        )

    # Recognize shorthand amounts and preserve the requested cadence subject.
    def test_spaced_values_and_annual_review_are_grounded(self) -> None:
        line = "Premium value of 2.5 k"
        self.assertEqual(values(line), values("$2,500"))
        self.assertEqual(
            labeled_passage("What value is listed for Premium?", line), line
        )
        cadence = "Annual review of external suppliers."
        self.assertEqual(
            field_evidence(
                "How often are external suppliers reviewed?",
                f"Quarterly review of the procurement policy.\n{cadence}",
            ),
            cadence,
        )

    # Match a duration to its unit and keep financial phases separate.
    def test_field_evidence_binds_requested_units_and_payment_status(self) -> None:
        self.assertEqual(
            field_evidence(
                "annual hours cap for volunteers",
                "Appointments last two years.\nA volunteer may serve up to 26 hours per year.",
            ),
            "A volunteer may serve up to 26 hours per year.",
        )
        records = "The applicant requested $8,500.\nMaximum awards may be $9,000."
        self.assertEqual(
            field_evidence("How much was ultimately awarded?", records), ""
        )
        self.assertEqual(
            field_evidence(
                "How much was ultimately awarded?",
                records + "\nThe panel awarded $7,200.",
            ),
            "The panel awarded $7,200.",
        )
        self.assertEqual(
            field_evidence(
                "What amount does the application request?",
                "Total cost\n$9,200\nAmount requested\n$7,200",
            ),
            "Amount requested\n$7,200",
        )
        self.assertEqual(
            field_evidence(
                "The review interval for that policy",
                "Archives are kept for six years.\nReview the policy every three years.",
            ),
            "Review the policy every three years.",
        )

    # Literal proof for a number cannot support a claim with a different unit.
    def test_verified_quantity_preserves_unit(self) -> None:
        proof = "The audit took 12 hours after a two-day preparation period."
        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {
                        "text": "The audit took 12 days.",
                        "proofs": [{"source": 1, "quote": proof}],
                    }
                ],
                "incomplete": False,
            }
        )
        self.assertEqual(verified_answer(draft, [EvidenceText("Audit", proof)]), "")

    # A full filename next to another filename keeps priority over its shorter suffix.
    def test_adjacent_filenames_do_not_select_a_suffix(self) -> None:
        originals = [
            ("notes", "Operations/Service notes.pdf"),
            ("full", "Policy/Warehouse-Safety-Guidelines.docx"),
            ("short", "Policy/Guidelines.docx"),
        ]
        self.assertEqual(
            explicit_ids(
                "Use Service notes.pdf, Warehouse-Safety-Guidelines.docx.", originals
            ),
            ["notes", "full"],
        )

    # Keep the evidence for an event separate from nearby values about different events.
    def test_requested_field_requires_the_event_and_value(self) -> None:
        fields = requested_fields(
            "Give the planning decisions and the exact workshop date "
            "and what dollar amount is printed beside supplies?"
        )
        self.assertEqual(len(fields), 3)
        record = "Board meeting: May 4, 2026\nSelect workshop date\nTravel: $320"
        self.assertEqual(field_evidence(fields[1], record), "")
        self.assertEqual(field_evidence(fields[2], record), "")
        self.assertEqual(
            field_evidence(fields[1], record + "\nWorkshop date\nJune 8, 2026"),
            "Workshop date\nJune 8, 2026",
        )
        self.assertEqual(
            field_evidence(fields[2], record + "\nSupplies: $475"),
            "Supplies: $475",
        )

    # A carried decision and a different reporting period leave the requested values open.
    def test_qualified_tallies_and_periods_require_explicit_values(self) -> None:
        self.assertEqual(
            field_evidence(
                "How many votes against the motion?",
                "Motion to pay $800 for rails. Carried.",
            ),
            "",
        )
        self.assertEqual(
            field_evidence(
                "How many votes against the motion?", "Motion vote: 4 votes in favour."
            ),
            "",
        )
        self.assertEqual(
            field_evidence(
                "What percentage was set for the end of 2026?",
                "40% by mid-2026. 60% by end of 2027.",
            ),
            "",
        )
        self.assertEqual(
            field_evidence(
                "What percentage was set for the end of 2026?", "70% by end of 2026."
            ),
            "70% by end of 2026.",
        )

    # Use the literal acceptance record even when nomination and offer lines are nearby.
    def test_role_acceptance_requires_the_requested_role(self) -> None:
        question = requested_fields(
            "Give who accepted the safety lead role at the meeting."
        )[0]
        line = "Rin Patel accepted the safety lead offer."
        self.assertEqual(
            labeled_passage(question, f"Ren Lee accepts the logistics role.\n{line}"),
            line,
        )

    # Keep related fields together and preserve commas in dates and amounts.
    def test_independent_requests_preserve_related_facts(self) -> None:
        parts = requested_parts(
            "Use the three originals. Report the sales total for March 3, 2024, "
            "the booking count and fee, and the two inventory amounts. "
            "Who approves spending over $2,001?"
        )
        self.assertEqual(len(parts), 4)
        self.assertIn("March 3, 2024", parts[0])
        self.assertIn("booking count and fee", parts[1])
        self.assertIn("$2,001", parts[3])

    # Join explicit first and last name columns and preserve only the names requested.
    def test_name_columns_exclude_unrequested_contacts(self) -> None:
        table = (
            "| Candidate First Name | Candidate Last Name | Extension |\n"
            "| Kelly | Reed | 123 |\n"
            "| Marc | Rivers | 456 |\n"
        )
        self.assertEqual(
            table_person_names("Give the names listed.", table),
            ["Kelly Reed", "Marc Rivers"],
        )
        self.assertEqual(table_person_names("Give the extensions.", table), [])

    # Prefer a literal section heading over scattered matches in another record.
    def test_requested_heading_distinguishes_sources_with_equal_topic_overlap(
        self,
    ) -> None:
        question = "Give the decisions listed on the Field Readiness slide."
        slide = EvidenceText("Briefing.pptx", "FIELD READINESS\nFit safety rails.")
        memo = EvidenceText(
            "Notes.docx", "Field inspections assess equipment readiness."
        )
        self.assertEqual(requested_part_score(question, slide), (0, 2, 2))
        self.assertEqual(requested_part_score(question, memo), (0, 2, 0))

    # Keep a version comparison focused on the requested part of the revision.
    def test_discards_supported_but_unrelated_changes(self) -> None:
        training = "The medical training goal increased to 100%."
        events = "Event publications increased to 6 per year."
        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {"text": text, "proofs": [{"source": 1, "quote": text}]}
                    for text in (training, events)
                ],
                "incomplete": False,
            }
        )
        answer = verified_answer(
            draft,
            [EvidenceText("Operations plan", f"{training}\n{events}")],
            "What did V3 add to the medical training goals?",
        )
        self.assertIn(training, answer)
        self.assertNotIn(events, answer)

    # Recover the relevant fact from a literal source passage when a model omitted it.
    def test_numeric_passages_keep_topic_and_quantity_together(self) -> None:
        answer = quantitative_passage(
            "What does the March 7, 2022 report say about the 12 insulated equipment sheds?",
            "Approved 19 staff passes.\nApproved twelve kits.\n"
            "The committee funded all 12 insulated equipment sheds for $31,400.",
        )
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("$31,400", answer[1])
        self.assertIsNone(
            quantitative_passage(
                "What does the report say about 12 insulated equipment sheds?",
                "Approved 19 staff passes.",
            )
        )

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
