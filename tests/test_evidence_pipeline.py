'''Check the shared evidence pipeline when follow-up search helps or fails.'''

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.backend import server
from src.backend.evidence import format_excerpt
from src.backend.grounding import (
    INCOMPLETE_TEXT,
    ClaimAnswer,
    CoverageReview,
    EvidenceText,
)
from src.backend.library_overview import build_overview
from src.backend.ragflow import IndexProgress, RagflowSession


# Build a deterministic model reply from exact supporting test passages.
def literal_claims(*passages: str) -> str:
    return ClaimAnswer.model_validate(
        {
            "claims": [
                {"text": text, "proofs": [{"source": index, "quote": text}]}
                for index, text in enumerate(passages, 1)
            ],
            "incomplete": False,
        }
    ).model_dump_json()


class EvidencePipelineTests(unittest.IsolatedAsyncioTestCase):
    # Preserve a low-ranking field passage within the document budget.
    def test_requested_field_survives_chunk_budget(self) -> None:
        noise = "The project pilot duration will be specialized later. " * 400
        reporting = "The group provides quarterly written updates to both committees."
        chunks = [
            server.Excerpt(
                server.RetrievedChunk(document_id="record", content=text), text
            )
            for text in (noise, reporting)
        ]
        retained = server.precise_evidence(
            "What is the pilot duration and reporting cadence?",
            chunks,
        )
        self.assertTrue(any(reporting in item.text for item in retained))
        self.assertLessEqual(
            sum(len(item.text) for item in retained), server.PERIOD_EVIDENCE_BUDGET
        )

    # Carry the last cited originals across multiple user turns and ignore draft answer claims.
    async def test_record_follow_up_keeps_only_the_latest_request(self) -> None:
        subject = "Use Maintenance Handbook.docx. How long are service records kept?"
        part = "And what is the effective date for that same policy?"
        history = [
            server.ChatMessage(role="user", content=subject),
            server.ChatMessage(role="assistant", content="Use Employee Manual.docx."),
            server.ChatMessage(role="user", content="What about the review interval?"),
            server.ChatMessage(role="assistant", content="The date was August 21."),
        ]
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client, documents={"maintenance": "Maintenance Handbook.docx"}
            )
            with (
                patch.object(
                    server,
                    "select_records",
                    AsyncMock(
                        side_effect=[
                            server.RecordSelection(),
                            server.RecordSelection(requested=True),
                        ]
                    ),
                ) as select,
                patch.object(
                    server,
                    "answer_records",
                    AsyncMock(
                        return_value=server.ChatResponse(
                            answer="checked", sources=[], mode_used="standard"
                        )
                    ),
                ) as answer,
            ):
                await server.respond_to_question(
                    state, part, server.Question(question=part, history=history)
                )
        self.assertEqual(
            [call.args[1] for call in select.await_args_list], [part, subject]
        )
        self.assertEqual(answer.await_args_list[0].args[1], part)

    # Clear a draft warning only after checking the actual requested details.
    async def test_scoped_coverage_preserves_genuine_gaps(self) -> None:
        decisions = "Replace the aging pump and inspect the outlet valve."
        record = EvidenceText("Maintenance briefing", f"Next Work\n{decisions}")
        draft = ClaimAnswer.model_validate_json(literal_claims(decisions))
        draft.incomplete = True
        good = CoverageReview(covered_lines=[1], missing=[])
        cases = (
            ("List the decisions under Next Work.", good.model_dump_json(), ""),
            (
                "List the decisions under Next Work and give their approved cost.",
                good.model_copy(
                    update={"missing": ["approved cost"]}
                ).model_dump_json(),
                "**Could not verify**\n\n- approved cost",
            ),
            (
                "List the decisions under Next Work.",
                good.model_copy(update={"covered_lines": [3]}).model_dump_json(),
                INCOMPLETE_TEXT,
            ),
            (
                "List the decisions under Next Work.",
                good.model_copy(
                    update={"missing": ["staff assignments"]}
                ).model_dump_json(),
                INCOMPLETE_TEXT,
            ),
            ("List the decisions under Next Work.", "{}", INCOMPLETE_TEXT),
        )
        async with httpx.AsyncClient() as client:
            state = RagflowSession(client)
            for question, review, warning in cases:
                with (
                    self.subTest(question=question, review=review),
                    patch.object(
                        server,
                        "generate",
                        AsyncMock(side_effect=[draft.model_dump_json(), review]),
                    ) as generate,
                ):
                    answer = await server.read_claims(
                        state, question, [record], [0], coverage="partial"
                    )
                    self.assertEqual(answer.text, f"{decisions} [1]")
                    self.assertEqual(answer.warning(), warning)
                    self.assertEqual(generate.await_count, 2)
                    self.assertIn(question, generate.await_args_list[-1].args[2])

    # Combine independent details into one warning while keeping the verified text.
    def test_record_warning_deduplicates_missing_details(self) -> None:
        combined = server.RecordAnswer(
            "The pump was replaced. [1]", ("the delivery date",)
        ).combined_with(
            server.RecordAnswer(
                "The outlet passed inspection. [2]",
                ("the delivery date.", "the inspection cost"),
            )
        )
        self.assertEqual(
            combined.text,
            "The pump was replaced. [1]\n\nThe outlet passed inspection. [2]",
        )
        self.assertEqual(
            combined.warning(),
            "**Could not verify**\n\n- the delivery date\n- the inspection cost",
        )

    # A fully supported draft can be returned without a separate coverage request.
    async def test_complete_scoped_draft_needs_no_coverage_review(self) -> None:
        line = "The delivery arrived on Friday."
        async with httpx.AsyncClient() as client:
            with patch.object(
                server, "generate", AsyncMock(return_value=literal_claims(line))
            ) as generate:
                answer = await server.read_claims(
                    RagflowSession(client),
                    "When did the delivery arrive?",
                    [EvidenceText("Delivery log", line)],
                    [0],
                    coverage="partial",
                )
                self.assertEqual(answer, server.RecordAnswer(f"{line} [1]"))
                generate.assert_awaited_once()

    # A complete draft flag cannot hide a verified gap on a specifically requested slide.
    async def test_named_slide_review_catches_a_silent_omission(self) -> None:
        question = "Give the decisions and the approved cost on the Next Work slide."
        line = "Inspect the outlet valve."
        check = CoverageReview(covered_lines=[1], missing=["the approved cost"])
        draft = ClaimAnswer.model_validate_json(literal_claims(line))
        draft.claims[0].proofs[0].source = 2
        async with httpx.AsyncClient() as client:
            with patch.object(
                server,
                "generate",
                AsyncMock(
                    side_effect=[draft.model_dump_json(), check.model_dump_json()]
                ),
            ) as generate:
                answer = await server.read_claims(
                    RagflowSession(client),
                    question,
                    [
                        EvidenceText("Earlier slides", "Approved cost: $8,000."),
                        EvidenceText("Next Work", line),
                    ],
                    [1],
                    coverage="always",
                )
                self.assertIn(f"{line} [2]", answer.text)
                self.assertNotIn("$8,000", generate.await_args_list[-1].args[2])
                self.assertEqual(
                    answer.warning(), "**Could not verify**\n\n- the approved cost"
                )

    # Recheck the answer after restoring a requested number from a literal passage.
    async def test_numeric_recovery_precedes_coverage_review(self) -> None:
        line = "A $6,000 disbursement will buy 24 new safety guards."
        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {
                        "text": "The disbursement will buy 24 new safety guards.",
                        "proofs": [{"source": 1, "quote": line}],
                    }
                ],
                "incomplete": True,
            }
        )
        review = CoverageReview(covered_lines=[2], missing=[])
        async with httpx.AsyncClient() as client:
            with patch.object(
                server,
                "generate",
                AsyncMock(
                    side_effect=[draft.model_dump_json(), review.model_dump_json()]
                ),
            ) as generate:
                answer = await server.read_claims(
                    RagflowSession(client),
                    "Report the disbursement amount and number of safety guards.",
                    [EvidenceText("Purchase record", line)],
                    [0],
                    coverage="partial",
                    literal_passage=f"> {line} [1]",
                )
                review_prompt = generate.await_args_list[-1].args[2]
                self.assertIn(f"Line 2: > {line}", review_prompt)
                self.assertIn(f"> {line} [1]", answer.text)
                self.assertEqual(answer.warning(), "")

    # Verify that comparison facts survive an extra Agentic evidence pass.
    async def test_preserves_original_and_new_milestones_in_both_modes(self) -> None:
        question = (
            "Compare the two operations plans. What did the revised plan add "
            "to the forklift inspection goals?"
        )
        baseline = "| Inspection | Inspect 40 forklift engines before June. | Notes |"
        updated = "| Inspection | Inspect 40 forklift engines before June. Inspect 60 by October. | Notes |"
        selection = server.RecordSelection(
            evidence=[
                server.Excerpt(
                    server.RetrievedChunk(document_id=doc, content=text), text
                )
                for doc, text in (("previous", baseline), ("revised", updated))
            ],
            requested=True,
        )
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="library",
                documents={
                    "previous": "Original plan.docx",
                    "revised": "Revised plan.docx",
                },
            )
            with (
                patch.object(
                    server,
                    "investigate",
                    AsyncMock(return_value=server.Investigation(question, [])),
                ),
                patch.object(server, "read_claims", AsyncMock()) as generate,
            ):
                for mode in ("standard", "agentic"):
                    payload = server.Question.model_validate(
                        {"question": question, "mode": mode}
                    )
                    result = await server.answer_records(
                        state, question, payload, selection
                    )
                    self.assertEqual(result.mode_used, mode)
                    self.assertIn("40 forklift engines before June", result.answer)
                    self.assertIn("60 by October", result.answer)
                    self.assertEqual(
                        [s.url for s in result.sources],
                        ["/sources/previous", "/sources/revised"],
                    )
                generate.assert_not_awaited()

    # Recover missing comparison facts even when the model already cited their originals.
    async def test_multisource_answer_checks_claim_coverage_separately(self) -> None:
        question = (
            "Compare the two operations plans. What did the revised plan add to the forklift "
            "inspection goals? How long does the onboarding seminar last and where is it held?"
        )
        input_text = [
            "Inspect 40 forklift engines before June.",
            "Inspect 40 forklift engines before June. Inspect 60 by October.",
            "The onboarding seminar lasts 16 hours.",
        ]
        evidence = [
            server.Excerpt(
                server.RetrievedChunk(document_id=str(index), content=text), text
            )
            for index, text in enumerate(input_text)
        ]
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="library",
                documents={
                    "0": "Original plan.txt",
                    "1": "Revised plan.txt",
                    "2": "Seminar.txt",
                },
            )
            sources = [server.citation(state, item.chunk) for item in evidence]
            with patch.object(
                server,
                "read_claims",
                AsyncMock(
                    side_effect=[
                        server.RecordAnswer(
                            "The onboarding seminar lasts 16 hours. [3]"
                        ),
                        server.RecordAnswer(
                            missing=("the location of the onboarding seminar",)
                        ),
                    ]
                ),
            ) as model:
                answer = await server.record_claims(
                    state, question, evidence, sources, {}
                )
                self.assertEqual(model.await_count, 2)
                self.assertTrue(
                    all(call.args[3] == [2] for call in model.await_args_list)
                )
            self.assertIn("40 forklift engines before June", answer.text)
            self.assertIn("60 by October", answer.text)
            self.assertIn("16 hours. [3]", answer.text)
            self.assertIn("the location of the onboarding seminar", answer.warning())

    # Recover facts the draft skipped and remove a value borrowed from a different event.
    async def test_coverage_recovers_supported_omissions(self) -> None:
        actual = "The completion date was April 8, 2025."
        inspection = "The inspection date was April 6, 2025."
        draft = ClaimAnswer.model_validate_json(literal_claims(inspection))
        draft.claims[0].text = "The completion date was April 6, 2025."
        request = "Give the completion date and the signing date."
        review = CoverageReview(
            covered_lines=[],
            missing=["the signing date"],
            recovered=ClaimAnswer.model_validate_json(literal_claims(actual)).claims,
            unsupported_lines=[1],
        )
        async with httpx.AsyncClient() as client:
            with patch.object(
                server,
                "generate",
                AsyncMock(
                    side_effect=[draft.model_dump_json(), review.model_dump_json()]
                ),
            ):
                answer = await server.read_claims(
                    RagflowSession(client),
                    request,
                    [
                        EvidenceText(
                            "Completion and inspection log", f"{actual}\n{inspection}"
                        )
                    ],
                    [0],
                    coverage="always",
                )
        self.assertEqual(answer.text, f"{actual} [1]")
        self.assertEqual(answer.missing, ("the signing date",))

    # A scoped comparison keeps the record opening when search ranks later text first.
    async def test_period_search_preserves_openings_and_ranked_chunks(self) -> None:
        opening = server.RetrievedChunk(
            id="opening", document_id="early", content="Opening inventory counts."
        )
        ranked = server.RetrievedChunk(
            id="ranked", document_id="early", content="An inventory correction."
        )
        later = server.RetrievedChunk(
            id="later", document_id="late", content="Later inventory counts."
        )
        async with httpx.AsyncClient() as client:
            state = RagflowSession(client)
            with (
                patch.object(
                    server, "record_opening", AsyncMock(side_effect=[later, opening])
                ),
                patch.object(
                    server, "retrieve", AsyncMock(side_effect=[[later], [ranked]])
                ) as retrieve,
            ):
                evidence = await server.period_evidence(
                    state,
                    "Compare inventory in 2024 vs 2022",
                    {2024: ["late"], 2022: ["early"]},
                    180,
                )
        self.assertEqual(
            [(item.period, item.chunk.id) for item in evidence],
            [(2024, "later"), (2022, "opening"), (2022, "ranked")],
        )
        self.assertEqual(
            [call.kwargs["document_ids"] for call in retrieve.await_args_list],
            [["late"], ["early"]],
        )

    # Count a clear roll call from the dated original in either requested mode.
    async def test_membership_total_uses_both_roll_call_groups(self) -> None:
        question = "How many council members do we have?"
        path = "data/Council/20250120_Meeting_Minutes.txt"
        chunk = server.RetrievedChunk(
            document_id="current",
            content=(
                "Present: Jamie Green, Sam Fields\n"
                "Regrets: Ari King\n"
                "Staff: Morgan West\n"
                "Guest: Jordan Stone"
            ),
        )
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="library",
                documents={"current": "Meeting_Minutes.txt"},
                source_paths={"Meeting_Minutes.txt": path},
            )
            with (
                patch.object(
                    server, "retrieve_recent", AsyncMock(return_value=[chunk])
                ),
                patch.object(server, "generate", AsyncMock()) as generate,
            ):
                for mode in ("standard", "agentic"):
                    payload = server.Question.model_validate(
                        {"question": question, "mode": mode}
                    )
                    result = await server.respond_to_question(state, question, payload)
                    self.assertIn("**3 members**", result.answer)
                    self.assertIn("2 present and 1 listed as absent", result.answer)
                    self.assertEqual([source.path for source in result.sources], [path])
                generate.assert_not_awaited()

    # Put recent applicable records ahead of stale keyword hits.
    async def test_present_question_gets_recent_indexed_record(self) -> None:
        question = "How many volunteers do we have?"
        current_path = "data/Volunteers/Meeting Notes/20250120_Volunteer_Status.txt"
        current = server.RetrievedChunk(
            id="current-hit", document_id="recent", content="Six active volunteers."
        )
        archived = server.RetrievedChunk(
            id="archived-hit", document_id="archive", content="Ten interviews planned."
        )

        # Return an out-of-scope hit as well to exercise defensive filtering.
        async def search(
            _state: RagflowSession, method: str, path: str, *, json: dict[str, object]
        ) -> object:
            self.assertEqual((method, path), ("POST", "/retrieval"))
            self.assertEqual(json["document_ids"], ["recent"])
            return {"chunks": [current.model_dump(), archived.model_dump()]}

        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="library",
                documents={
                    "recent": "Volunteer Status.txt",
                    "archive": "Feasibility Study.txt",
                },
                source_paths={
                    "Volunteer Status.txt": current_path,
                    "Feasibility Study.txt": "data/Archived/Feasibility Study.txt",
                },
            )
            with (
                patch.object(server, "request", side_effect=search),
                patch.object(
                    server,
                    "generate",
                    AsyncMock(return_value=literal_claims(current.content)),
                ) as generate,
            ):
                result = await server.respond_to_question(
                    state, question, server.Question(question=question)
                )
                self.assertEqual(
                    [source.path for source in result.sources], [current_path]
                )
                args = generate.await_args
                assert args is not None
                prompt = str(args.args[2])
                self.assertIn(current.content, prompt)
                self.assertNotIn(archived.content, prompt)

    # Put both explicitly named originals in the answer even when broad search misses them.
    async def test_both_modes_reserve_evidence_from_named_originals(self) -> None:
        question = (
            "Using the Municipal Equipment Lease Terms and the Coastal Radio Renewal Plan, "
            "when are the equipment payments and radio renewals due?"
        )
        lease = server.RetrievedChunk(
            document_id="lease", content="Payment is due in March."
        )
        plan = server.RetrievedChunk(
            document_id="plan", content="Radio renewal is due in May."
        )

        # Resolve the actual originals without a library-wide retrieval.
        async def search(
            _state: RagflowSession,
            document_id: str,
            _limit: int,
        ) -> list[server.RetrievedChunk]:
            return [{"lease": lease, "plan": plan}[document_id]]

        draft = ClaimAnswer.model_validate(
            {
                "claims": [
                    {
                        "text": lease.content,
                        "proofs": [{"source": 1, "quote": lease.content}],
                    },
                    {
                        "text": plan.content,
                        "proofs": [{"source": 2, "quote": plan.content}],
                    },
                ],
                "incomplete": False,
            }
        )

        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="library",
                documents={
                    "lease": "Municipal Equipment Lease Terms.txt",
                    "plan": "Coastal Radio Renewal Plan.txt",
                    "other": "Expense Policy.txt",
                },
            )
            with (
                patch.object(server, "record_chunks", side_effect=search),
                patch.object(
                    server,
                    "investigate",
                    AsyncMock(return_value=server.Investigation(question, [])),
                ),
                patch.object(
                    server, "generate", AsyncMock(return_value=draft.model_dump_json())
                ) as generate,
            ):
                for mode in ("standard", "agentic"):
                    payload = server.Question.model_validate(
                        {"question": question, "mode": mode}
                    )
                    result = await server.respond_to_question(state, question, payload)
                    self.assertEqual(
                        [source.url for source in result.sources],
                        ["/sources/lease", "/sources/plan"],
                    )
                    assert generate.await_args is not None
                    prompt = str(generate.await_args.args[2])
                    self.assertIn(lease.content, prompt)
                    self.assertIn(plan.content, prompt)

    # Answer collection orientation in either mode using the complete file inventory.
    async def test_library_summary_uses_inventory_without_model_calls(self) -> None:
        library = server.Library(
            native=IndexProgress(indexed_count=1, total_count=2),
            overview=build_overview(
                ["backup/Finance/plan.pdf", "backup/Programs/calendar.docx"]
            ),
        )
        overview = "just a high level summary of everything that exists"
        one_line = "give me a short 1 lines summary with all the important numbers"

        async with httpx.AsyncClient() as client:
            state = RagflowSession(client)
            with (
                patch.object(server, "retrieve", AsyncMock()) as retrieve,
                patch.object(server, "generate", AsyncMock()) as generate,
            ):
                for mode in ("standard", "agentic"):
                    for question in (overview, one_line):
                        payload = server.Question.model_validate(
                            {
                                "question": question,
                                "mode": mode,
                                "history": [{"role": "user", "content": overview}]
                                if question == one_line
                                else [],
                            }
                        )
                        result = await server.respond_to_question(
                            state, question, payload, library
                        )
                        self.assertIn("Finance", result.answer)
                        self.assertEqual(result.sources, [])
                        self.assertEqual("\n" in result.answer, question == overview)
                retrieve.assert_not_awaited()
                generate.assert_not_awaited()

    # Ask for scope in either mode without treating previous assistant guesses as context.
    async def test_unscoped_summary_needs_no_search_or_generation(self) -> None:
        question = "give me a short 1 lines summary with all the important numbers"
        history = [
            server.ChatMessage(role="user", content=question),
            server.ChatMessage(
                role="assistant", content="I found a reconciliation report."
            ),
        ]
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="current",
                documents={"fund": "Reconciliation.txt", "policy": "Policy.txt"},
                source_paths={
                    "Reconciliation.txt": "finance/Reconciliation.txt",
                    "Policy.txt": "operations/Policy.txt",
                },
            )
            with (
                patch.object(server, "retrieve", new_callable=AsyncMock) as retrieve,
                patch.object(
                    server, "investigate", new_callable=AsyncMock
                ) as investigate,
                patch.object(server, "generate", new_callable=AsyncMock) as generate,
            ):
                for payload in (
                    server.Question(question=question, history=history),
                    server.Question(question=question, history=history, mode="agentic"),
                ):
                    with self.subTest(mode=payload.mode):
                        result = await server.respond_to_question(
                            state, question, payload
                        )
                        self.assertIn(
                            "Which document, topic, or time period", result.answer
                        )
                        self.assertEqual(result.sources, [])
                retrieve.assert_not_awaited()
                investigate.assert_not_awaited()
                generate.assert_not_awaited()

    # Carry user-provided scope into retrieval while keeping the latest answer format.
    async def test_summary_follow_up_keeps_user_subject_and_format_in_both_modes(
        self,
    ) -> None:
        subject = "What were the 2025 operating grant award totals?"
        question = "Summarize the important figures in 2 sentences."
        history = [
            server.ChatMessage(role="user", content=subject),
            server.ChatMessage(role="assistant", content="The building budget was $5."),
        ]
        award = server.RetrievedChunk(
            document_id="awards",
            content="The 2025 operating grants total came to $100.",
        )
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client, dataset_id="current", documents={"awards": "Award totals.txt"}
            )
            with (
                patch.object(
                    server, "retrieve", AsyncMock(return_value=[award])
                ) as retrieve,
                patch.object(
                    server,
                    "investigate",
                    AsyncMock(
                        return_value=server.Investigation(question=subject, chunks=[])
                    ),
                ),
                patch.object(
                    server,
                    "generate",
                    AsyncMock(return_value=literal_claims(award.content)),
                ) as generate,
            ):
                for payload in (
                    server.Question(question=question, history=history),
                    server.Question(question=question, history=history, mode="agentic"),
                ):
                    with self.subTest(mode=payload.mode):
                        result = await server.respond_to_question(
                            state, question, payload
                        )
                        assert (
                            retrieve.await_args is not None
                            and generate.await_args is not None
                        )
                        search = str(retrieve.await_args.args[1])
                        answer_question = str(generate.await_args.args[2]).rsplit(
                            "\nQuestion:\n", 1
                        )[-1]
                        self.assertTrue(search.startswith(question))
                        self.assertIn(subject, search)
                        self.assertNotIn("building budget", search)
                        self.assertEqual(answer_question, search)
                        self.assertEqual(
                            [source.url for source in result.sources],
                            ["/sources/awards"],
                        )

    # Use the configured original as scope in the one-file pilot.
    async def test_summary_can_use_the_only_configured_source(self) -> None:
        question = "Summarize the key numbers in 1 line."
        filename = "data/Annual Accounts.pdf"
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client, source_paths={"Annual Accounts.pdf": filename}
            )
            with patch.object(
                server, "retrieve", AsyncMock(return_value=[])
            ) as retrieve:
                await server.respond_to_question(
                    state, question, server.Question(question=question)
                )
                retrieve.assert_awaited_once()
                assert retrieve.await_args is not None
                search = str(retrieve.await_args.args[1])
                self.assertTrue(search.startswith(question))
                self.assertIn(filename, search)

    # Verify that iterative search keeps the baseline and adds each new source once.
    async def test_agentic_adds_distinct_evidence(self) -> None:
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client,
                dataset_id="current",
                documents={"guide": "Guide.txt", "notice": "Notice.md"},
            )
            baseline = server.RetrievedChunk(
                id="rules",
                document_id="guide",
                content="Bring a photo ID to pick up a replacement card.",
            )
            additional = server.RetrievedChunk(
                id="hours",
                document_id="notice",
                content="Replacement card pickup is open until 6 p.m. on Thursdays.",
            )
            with (
                patch.object(server, "retrieve", AsyncMock(return_value=[baseline])),
                patch.object(
                    server,
                    "investigate",
                    AsyncMock(
                        return_value=server.Investigation(
                            question="When and how can I pick up my replacement card?",
                            chunks=[baseline, additional],
                        )
                    ),
                ),
                patch.object(
                    server,
                    "generate",
                    AsyncMock(
                        return_value=literal_claims(
                            baseline.content, additional.content
                        )
                    ),
                ) as generate,
            ):
                question = "What do I need to bring, and when can I go?"
                earlier = server.ChatMessage(
                    role="user", content="I need a replacement card."
                )
                for history in ([], [earlier]):
                    result = await server.respond_to_question(
                        state,
                        question,
                        server.Question(
                            question=question, mode="agentic", history=history
                        ),
                    )
                    self.assertEqual(result.mode_used, "agentic")
                    self.assertEqual(
                        [source.url for source in result.sources],
                        ["/sources/guide", "/sources/notice"],
                    )
                    args = generate.await_args
                    assert args is not None
                    prompt = str(args.args[2])
                    self.assertTrue(prompt.endswith(f"\nQuestion:\n{question}"))
                    self.assertEqual(
                        "When and how can I pick up my replacement card?" in prompt,
                        bool(history),
                    )
                    self.assertLess(
                        prompt.index(baseline.content),
                        prompt.index(additional.content),
                    )

    # Verify that an unavailable agent generates the same prompt and sources as Standard.
    async def test_agentic_failure_preserves_standard_answer(self) -> None:
        async with httpx.AsyncClient() as client:
            state = RagflowSession(
                client, dataset_id="current", documents={"doc": "Policy.md"}
            )
            chunk = server.RetrievedChunk(
                document_id="doc", content="You can request a replacement online."
            )
            question = "How do I request one?"
            history = [server.ChatMessage(role="user", content="I lost my card.")]
            with (
                patch.object(server, "retrieve", AsyncMock(return_value=[chunk])),
                patch.object(
                    server, "investigate", AsyncMock(side_effect=TimeoutError)
                ),
                patch.object(
                    server,
                    "generate",
                    AsyncMock(return_value=literal_claims(chunk.content)),
                ) as generate,
                self.assertLogs(server.LOGGER, level="WARNING"),
            ):
                standard = await server.respond_to_question(
                    state, question, server.Question(question=question, history=history)
                )
                agentic = await server.respond_to_question(
                    state,
                    question,
                    server.Question(question=question, history=history, mode="agentic"),
                )
                self.assertEqual(agentic, standard)
                self.assertEqual(
                    generate.await_args_list[0], generate.await_args_list[1]
                )


class EvidenceRankingTests(unittest.TestCase):
    # Resolve a year-only follow-up from the user's wording and ignore prior answers.
    def test_year_follow_up_carries_subject_into_requested_year(self) -> None:
        history = [
            server.ChatMessage(
                role="user", content="List all council memebers for 2024"
            ),
            server.ChatMessage(
                role="assistant", content="Here are nominees from 2021."
            ),
        ]
        self.assertEqual(
            server.scoped_question("and for 2022?", history),
            "List all council memebers for 2022",
        )
        self.assertIsNone(server.scoped_question("and for 2022?", []))

    # Retain only sources the answer cites and preserve repeated links when renumbering.
    def test_display_citations_follow_the_order_of_first_use(self) -> None:
        sources = [
            server.Citation(name=name, path=name, url=f"/sources/{index}")
            for index, name in enumerate(("Policy.txt", "Uncited.txt", "Notice.txt"), 1)
        ]
        answer, used = server.cited_answer("See [3, 1], then [3] again.[9]", sources)
        self.assertEqual(answer, "See [1] [2], then [1] again.")
        self.assertEqual(used, [sources[2], sources[0]])
        self.assertEqual(server.cited_answer("No supported finding.", sources)[1], [])

    # Merge matching quotes across fields while preserving distinct source references.
    def test_repeated_passages_merge_before_citation_numbering(self) -> None:
        sources = [
            server.Citation(name=name, path=name, url=url)
            for name, url in (
                ("Schedule.docx", "/sources/schedule"),
                ("Schedule.docx", "/sources/schedule"),
                ("Minutes.docx", "/sources/minutes"),
            )
        ]
        passage = "Training is quarterly, with up to ten hours per participant."
        answer, used = server.cited_answer(
            f"> {passage} [1]\n\nA separate condition applies. [1]\n\n"
            f"> {passage} [2, 3]\n\n> Training hours vary by role. [3]",
            sources,
        )
        self.assertEqual(answer.count(passage), 1)
        self.assertEqual(
            answer,
            f"> {passage} [1] [2]\n\nA separate condition applies. [1]\n\n"
            "> Training hours vary by role. [2]",
        )
        self.assertEqual(used, [sources[0], sources[2]])

    # Keep both subjects of a standalone cross-folder question.
    def test_cross_folder_prompt_preserves_both_requested_subjects(self) -> None:
        question = (
            "In the August 2025 Visitor Services website outline, where did the draft put Parking? "
            "Using the Updated Visitor Parking Policy, explain the requirements for day passes. "
            "Cite both originals and keep the answer under 90 words."
        )
        self.assertEqual(server.scoped_question(question, []), question)

    # Preserve two leading excerpts from the same document under a tight budget.
    def test_keeps_strong_supporting_excerpt(self) -> None:
        chunks = [
            server.RetrievedChunk(
                id="limits", document_id="guide", content="Award limits"
            ),
            server.RetrievedChunk(
                id="match", document_id="guide", content="Matching rules"
            ),
            server.RetrievedChunk(
                id="steps", document_id="workflow", content="Staff steps"
            ),
            server.RetrievedChunk(
                id="other", document_id="other", content="Related only"
            ),
        ]
        selected = server.select_evidence(chunks, 39)
        self.assertEqual(
            [item.chunk.id for item in selected], ["limits", "match", "steps"]
        )


class EvidenceFormatTests(unittest.TestCase):
    # Verify that arbitrary text and merged table values retain their column positions.
    def test_table_preserves_layout(self) -> None:
        source = (
            "Site inventory"
            "<table><caption>Regional <b>inventory</b><br>February 2025</caption>"
            "<tr><th>Site</th><th colspan='2'>Equipment</th></tr>"
            "<tr><td rowspan='2'>Harbour</td><td>Radio</td><td>4</td></tr>"
            "<tr><td>Charger</td><td>7</td></tr></table>"
            "Inventory is checked weekly."
        )
        excerpt = format_excerpt(source)
        self.assertIn("Site inventory", excerpt)
        self.assertIn("Regional inventory February 2025", excerpt)
        self.assertIn("| Site | Equipment | (left) |", excerpt)
        self.assertIn("| Harbour | Radio | 4 |", excerpt)
        self.assertIn("| (above) | Charger | 7 |", excerpt)
        self.assertTrue(excerpt.endswith("Inventory is checked weekly."))


if __name__ == "__main__":
    unittest.main()
