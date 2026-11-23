'''Exercise source selection and proof checks with unrelated synthetic records.'''

import unittest
from datetime import date
from io import BytesIO
from zipfile import ZipFile

from src.backend.grounding import (
    ClaimAnswer,
    EvidenceText,
    field_context,
    field_evidence,
    labeled_passage,
    primary_passages,
    quantitative_passage,
    rate_passages,
    requested_fields,
    requested_part_score,
    requested_parts,
    start_passage,
    threshold_passage,
    values,
    verified_answer,
)
from src.backend.record_lookup import (
    dated_references,
    document_period,
    explicit_ids,
    has_record_date,
)
from src.backend.tabular import count_named_rows, requested_table, table_person_names


class RecordEvidenceTests(unittest.TestCase):
    # Quote each fee and its own base, even when other rates appear nearby.
    def test_fee_rates_keep_their_separate_bases(self) -> None:
        question = "Compare the booking fee percentages and what each rate applies to."
        first = "The booking fee is 4% of on-site ticket receipts."
        second = "The booking fee is 8% of online sales."
        record = f"{first} A separate tax is 10%.\n{second}"
        self.assertEqual(rate_passages(question, record).splitlines(), [first, second])
        parts = requested_parts(question + " Include every rate.")
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(requested_fields(parts[0])), 1)

    # A planned finish and a preparation cadence must match their explicit topics.
    def test_planned_date_and_active_voice_cadence(self) -> None:
        date = "The building is to be fully inspected by August 4, 2018."
        report = "Prepare a monthly incident log for supervisor review."
        evidence = "The hiring process started July 4, 2018.\n" + date + "\n" + report
        self.assertEqual(
            field_evidence(
                "What date is the building intended to be fully inspected", evidence
            ),
            date,
        )
        self.assertEqual(
            field_evidence(
                "How often should they prepare an incident log for supervisor review",
                evidence,
            ),
            report,
        )
        amount = "Receipts apply to contributions over $35.00."
        self.assertEqual(
            field_evidence(
                "What minimum contribution amount applies to a receipt?", amount
            ),
            amount,
        )

    # A follow-on instruction and a coordinated yes-or-no question remain separate.
    def test_preserves_following_fact_requests(self) -> None:
        first, second = requested_parts(
            "Who approves a service change? Include notice requirements and say whether the panel can cancel it."
        )
        self.assertEqual(first, "Who approves a service change?")
        self.assertEqual(
            requested_fields(second),
            ["Include notice requirements", "whether the panel can cancel it"],
        )
        question = "Who can end the arrangement? Include notice requirements and say whether an extension is allowed."
        parent = field_context(question, "Include notice requirements")
        self.assertEqual(parent, "Who can end the arrangement")
        evidence = "The meeting notice is shared with observers.\nTermination takes effect on written notice to the custodian."
        self.assertTrue(
            primary_passages(
                "Include notice requirements", evidence, parent
            ).startswith("Termination takes effect")
        )

    # Each requested time estimate must stay with its own activity.
    def test_time_commitment_list_preserves_qualified_totals(self) -> None:
        fields = requested_fields(
            "How much annual time is expected for touring, rehearsals, and overall?"
        )
        record = "Touring: 20 hours.\nRehearsals: 7-9 hours.\nOverall: 31 hours."
        self.assertEqual(
            [field_evidence(field, record) for field in fields], record.splitlines()
        )

    # A recommendation and an unrelated permission are not proof of authority.
    def test_permission_preserves_negative_and_requested_object(self) -> None:
        question = "Could the panel sign service contracts?"
        rule = "The panel has no delegated authority to approve service contracts."
        proposal = "The panel can recommend approval of service contracts."
        unrelated = "The panel can sign travel forms."
        self.assertEqual(
            labeled_passage(question, f"{proposal}\n{unrelated}\n{rule}"), rule
        )
        self.assertEqual(labeled_passage(question, f"{proposal}\n{unrelated}"), "")

    # A dated payout schedule is distinct from the due date for distributing income.
    def test_payout_date_keeps_actual_schedule_label(self) -> None:
        q = "Is there a calendar day for paying individual recipients?"
        schedule = "Funds awarded to recipient's account: Nov 23, 2019"
        income = "Fund income must be distributed by December 10 of each year."
        self.assertEqual(labeled_passage(q, f"{income}\n{schedule}"), schedule)
        self.assertEqual(labeled_passage(q, income), "")

    # Reimbursement eligibility never fills the missing dollar cap.
    def test_missing_bound_stays_missing_when_other_facts_are_present(self) -> None:
        q = "What maximum travel reimbursement amount is listed?"
        policy = (
            "Travel reimbursement is available.\nSoftware purchases are capped at $200."
        )
        self.assertEqual(field_evidence(q, policy), "")
        self.assertEqual(
            field_evidence(q, f"{policy}\nTravel reimbursement is capped at $80."),
            "Travel reimbursement is capped at $80.",
        )
        self.assertEqual(
            field_evidence(q, "There is no cap for travel reimbursement."),
            "There is no cap for travel reimbursement.",
        )

    # Distinguish a requested due date from the separate ceiling on each payment.
    def test_coordinated_deadline_and_amount_each_get_a_verification(self) -> None:
        fields = requested_fields(
            "Does the policy specify an application deadline and a maximum amount per grant?"
        )
        self.assertEqual(len(fields), 2)
        original = (
            "The application deadline is August 15, 2019.\nA grant can be up to $350."
        )
        self.assertEqual(
            [field_evidence(field, original) for field in fields], original.splitlines()
        )
        self.assertEqual(
            [field_evidence(field, "Grants are available.") for field in fields],
            ["", ""],
        )

    # A start date inherits the subject rather than choosing a different application milestone.
    def test_start_date_passage_preserves_the_continuation(self) -> None:
        question = "What is the pilot award amount, when does it start?"
        detail = "when does it start"
        context = field_context(question, detail)
        headings = (
            "Applications open October 1, 2015.\n"
            "One pilot award annually, beginning in 2016 at £400 per\n"
            "award, subject to budget approval."
        )
        self.assertIn(
            "beginning in 2016 at £400 per award, subject to budget approval.",
            primary_passages(detail, headings, context),
        )
        self.assertEqual(
            start_passage(detail, headings, context),
            "One pilot award annually, beginning in 2016 at £400 per award, subject to budget approval.",
        )

    # Approval evidence must address the requested subject, even beside similar clauses.
    def test_approval_evidence_requires_the_actual_subject(self) -> None:
        note = "The work group can recommend schedule changes for approval."
        rule = "Any change in mandate requires approval by both authorities."
        q = "Who must approve changing the work group's mandate?"
        self.assertEqual(labeled_passage(q, f"{note}\n{rule}"), rule)
        self.assertEqual(labeled_passage(q, note), "")

    # Thresholds remain tied to their triggering condition and distinct from ordinary prices.
    def test_threshold_keeps_its_condition(self) -> None:
        condition = (
            "Once a threshold of $1,250 is reached, the reserve can be drawn down."
        )
        question = "What balance allows the reserve to be drawn down?"
        self.assertEqual(
            threshold_passage(question, f"The monthly fee is $80.\n{condition}"),
            condition,
        )
        self.assertEqual(
            threshold_passage(
                question, condition.replace(" $1,250", " $1,250 or $3,400")
            ),
            "",
        )

    # Resolve older day-first headings with a century established by the original.
    def test_reporting_date_supports_legacy_headings_without_guessing_century(
        self,
    ) -> None:
        self.assertEqual(
            document_period("Audit2012Report", "30 Nov 12\nInventory"),
            date(2012, 11, 30),
        )
        self.assertEqual(
            document_period("Inspection report", "As of 1 March 1988"), date(1988, 3, 1)
        )
        self.assertIsNone(document_period("Undated report", "30 Nov 12\nInventory"))

    # Compare tied facts only within the requested event, subject, and value type.
    def test_approved_amount_and_form_count_require_their_own_evidence(self) -> None:
        proposal = (
            "Previous program total funding: $61,000.\n"
            "The proposed budget is $18,000.\n"
            "The panel approved $9,000 for tree planting.\n"
            "How many endorsements are required to submit an application?: 2"
        )
        self.assertEqual(
            field_evidence(
                "What dollar amount was actually approved for trail repairs?", proposal
            ),
            "",
        )
        self.assertEqual(
            field_evidence(
                "What dollar amount was actually approved for tree planting?", proposal
            ),
            "The panel approved $9,000 for tree planting.",
        )
        self.assertEqual(
            field_evidence(
                "How many endorsements does the form say are required?", proposal
            ),
            "How many endorsements are required to submit an application?: 2",
        )

    # Preserve prerequisites from each numbered stage and the original day unit.
    def test_prerequisites_and_relative_deadlines_are_complete(self) -> None:
        policy = (
            "Inspection 1: April\nFollowing receipt of the submitted drawings.\n"
            "Inspection 2: May\nOnce the final checklist has been submitted.\n"
            "Inspection forms must be filed within 5 business days."
        )
        self.assertEqual(
            labeled_passage("What must be submitted before each inspection?", policy),
            "\n".join(policy.splitlines()[:4]),
        )
        self.assertEqual(
            labeled_passage("Explain the inspection form filing deadline", policy),
            "Inspection forms must be filed within 5 business days.",
        )

    # Each named duration must use its own amount and its own evidence.
    def test_coordinated_time_fields_are_verified_separately(self) -> None:
        first, second = requested_fields(
            "How many hours does the handbook allow for installation and for inspection?"
        )
        instructions = (
            "Installation: 20 hours.\nInspection: 4-6 hours.\nTotal: 26 hours."
        )
        self.assertEqual(field_evidence(first, instructions), "Installation: 20 hours.")
        self.assertEqual(field_evidence(second, instructions), "Inspection: 4-6 hours.")

    # Preserve the event, method, and number written in a scoped original.
    def test_workflow_and_count_passages(self) -> None:
        record = (
            "When auditions close\nDisable the registration form and send a confirmation.\n"
            "Once performers are selected through a panel vote, notify staff.\n"
            "At least 3 independent recommendations are recorded."
        )
        self.assertEqual(
            labeled_passage(
                "What happens to the registration form when auditions close?", record
            ),
            "When auditions close\nDisable the registration form and send a confirmation.",
        )
        self.assertEqual(
            labeled_passage("How are performers selected?", record),
            "Once performers are selected through a panel vote, notify staff.",
        )
        self.assertEqual(
            field_evidence(
                "How many independent recommendations are required?", record
            ),
            "At least 3 independent recommendations are recorded.",
        )
        self.assertEqual(
            field_evidence(
                "When was the form completed?",
                "Start time: Oct 1, 2025\nCompletion time: Oct 2, 2025",
            ),
            "Completion time: Oct 2, 2025",
        )

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
