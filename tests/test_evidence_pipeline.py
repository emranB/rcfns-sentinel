'''Check the shared evidence pipeline when follow-up search helps or fails.'''

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.backend import server
from src.backend.evidence import format_excerpt
from src.backend.library_overview import build_overview
from src.backend.ragflow import IndexProgress, RagflowSession


class EvidencePipelineTests(unittest.IsolatedAsyncioTestCase):
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
                    server, "generate", AsyncMock(return_value="Six volunteers [1].")
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
        other = server.RetrievedChunk(
            document_id="other", content="Submit expenses in July."
        )

        # Give general retrieval an unrelated result and scope exact requests by ID.
        async def search(
            _state: RagflowSession,
            _question: str,
            _limit: int = 8,
            *,
            document_ids: list[str] | None = None,
        ) -> list[server.RetrievedChunk]:
            by_document = {"lease": lease, "plan": plan}
            return [by_document[document_ids[0]]] if document_ids else [other]

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
                patch.object(server, "retrieve", side_effect=search),
                patch.object(
                    server,
                    "investigate",
                    AsyncMock(return_value=server.Investigation(question, [])),
                ),
                patch.object(
                    server, "generate", AsyncMock(return_value="March [1] and May [2].")
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
                    AsyncMock(return_value="Awards totaled $100 [1]."),
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
                    server, "generate", AsyncMock(return_value="See [1] and [2].")
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
                    server, "generate", AsyncMock(return_value="Request it online [1].")
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
    # Retain only sources the answer cites and preserve repeated links when renumbering.
    def test_display_citations_follow_the_order_of_first_use(self) -> None:
        sources = [
            server.Citation(name=name, path=name, url=f"/sources/{index}")
            for index, name in enumerate(("Policy.txt", "Uncited.txt", "Notice.txt"), 1)
        ]
        answer, used = server.cited_answer(
            "See [3], then [1], then [3] again. [9] is not a valid source.", sources
        )
        self.assertEqual(
            answer, "See [1], then [2], then [1] again. [9] is not a valid source."
        )
        self.assertEqual(used, [sources[2], sources[0]])
        self.assertEqual(server.cited_answer("No supported finding.", sources)[1], [])

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
