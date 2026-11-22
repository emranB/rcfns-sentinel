'''Regression checks for bulk indexing and resumable uploads.'''

import asyncio
from collections.abc import Iterator
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import get_ident
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch

import httpx

from src.backend import ragflow
from src.backend.index_cache import prepare_index_cache


class UploadBatchTests(TestCase):
    # Prefer smaller sources while enforcing the combined upload limit.
    def test_smallest_files_first(self) -> None:
        with TemporaryDirectory() as directory:
            files = {
                name: Path(directory) / name for name in ("large", "medium", "small")
            }
            for size, file in zip((15, 4, 1), files.values(), strict=True):
                file.write_bytes(b"x" * size)

            with patch.object(ragflow, "BATCH_BYTES", 8):
                batches = list(ragflow.upload_batches(files))

            self.assertEqual(
                [list(batch) for batch in batches], [["small", "medium"], ["large"]]
            )

    # Reject an oversized source before allocating its upload buffer.
    def test_reject_oversized_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(b"oversized")
            with (
                patch.object(ragflow, "MAX_DOCUMENT_UPLOAD_SIZE", 4),
                patch.object(
                    Path, "open", side_effect=AssertionError("read attempted")
                ),
                self.assertRaisesRegex(ValueError, "exceeds the upload limit"),
            ):
                ragflow.read_upload_files({path.name: path})

    # Preserve both classes in the first batches so cheaper files do not wait behind OCR.
    def test_interleave_lightweight_and_layout_files(self) -> None:
        sizes = {
            "later.pdf": 7,
            "policy.docx": 5,
            "instructions.txt": 4,
            "first.PNG": 3,
            "outline.md": 2,
            "small.csv": 1,
        }
        with TemporaryDirectory() as directory:
            files = {name: Path(directory) / name for name in sizes}
            for name, file in files.items():
                file.write_bytes(b"x" * sizes[name])
            order = [name for _, name, _ in ragflow.prioritized_files(files)]

        self.assertEqual(
            order,
            [
                "small.csv",
                "outline.md",
                "instructions.txt",
                "first.PNG",
                "policy.docx",
                "later.pdf",
            ],
        )


class IngestionTests(IsolatedAsyncioTestCase):
    # Retry a document listing after a timeout and 503 without replaying writes.
    async def test_retries_transient_read_errors_without_replaying_writes(self) -> None:
        calls: list[str] = []
        documents_path = "/datasets/library/documents"

        # Simulate a provider going briefly offline and recovering its document list.
        def provider(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, f"/api/v1{documents_path}")
            calls.append(request.method)
            if request.method != "GET" or len(calls) == 1:
                raise httpx.ReadTimeout("Provider is restarting", request=request)
            if len(calls) == 2:
                return httpx.Response(503, text="Provider is restarting")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "docs": [{"id": "original-id", "name": "original.txt"}],
                        "total": 1,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
            state = ragflow.RagflowSession(client, authorization="test-token")
            with patch.object(ragflow, "READ_RETRY_DELAYS", (0, 0, 0, 0)):
                documents = await ragflow.list_documents(state, documents_path)
                for method in ("POST", "DELETE"):
                    with self.assertRaises(httpx.ReadTimeout):
                        await ragflow.request(state, method, documents_path)

        self.assertEqual(documents["original.txt"].id, "original-id")
        self.assertEqual(calls, ["GET", "GET", "GET", "POST", "DELETE"])

    # Preserve the existing index and request ingestion only for the added source.
    async def test_cached_index_submits_only_the_added_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = {name: root / name for name in ("existing.txt", "pending.txt")}
            for path in files.values():
                path.write_text(path.stem, encoding="utf-8")
            existing = ragflow.Document(
                id="existing-id",
                name="existing(1).txt",
                ingestion_status="COMPLETED",
                chunk_count=1,
            )
            pending = ragflow.Document(
                id="pending-id", name="pending.txt", ingestion_status="SCHEDULED"
            )
            remote = {item.name: item for item in (existing, pending)}
            first = prepare_index_cache(root, "collection", "", files, {})
            ragflow.record_cached_documents(
                first, {"existing.txt": existing, "pending.txt": pending}
            )
            first.save()

            addition = root / "new.txt"
            addition.write_text("new source bytes", encoding="utf-8")
            files[addition.name] = addition
            cache = prepare_index_cache(root, "collection", "", files, {})
            new_name = cache.names[addition.name]
            new = ragflow.Document(
                id="new-id", name=new_name, ingestion_status="PENDING"
            )
            writes: list[tuple[str, str]] = []

            # Simulate a provider that already has the deployed dataset and jobs.
            def provider(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if request.method == "GET" and path == "/api/v1/datasets":
                    self.assertEqual(request.url.params["name"], first.manifest.dataset)
                    data: object = [
                        {"id": "existing-dataset", "name": first.manifest.dataset}
                    ]
                elif request.method == "GET" and path.endswith("/documents"):
                    self.assertEqual(
                        path, "/api/v1/datasets/existing-dataset/documents"
                    )
                    data = {
                        "docs": [item.model_dump() for item in remote.values()],
                        "total": len(remote),
                    }
                elif request.method == "POST" and path.endswith("/documents"):
                    body = request.read()
                    self.assertEqual(body.count(b'name="file"'), 1)
                    self.assertIn(f'filename="{new_name}"'.encode(), body)
                    self.assertIn(b"new source bytes", body)
                    writes.append((request.method, path))
                    remote[new_name] = new
                    data = [new.model_dump()]
                elif request.method == "POST" and path.endswith("/chunks"):
                    self.assertEqual(
                        loads(request.read()), {"document_ids": ["new-id"]}
                    )
                    writes.append((request.method, path))
                    for item in (new, pending):
                        remote[item.name] = item.model_copy(
                            update={"ingestion_status": "COMPLETED", "chunk_count": 1}
                        )
                    data = {}
                else:
                    self.fail(f"Unexpected provider call {request.method} {path}")
                return httpx.Response(200, json={"code": 0, "data": data})

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(provider)
            ) as client:
                state = ragflow.RagflowSession(client, authorization="test-token")
                progress = ragflow.IndexProgress()
                await ragflow.index_sources(
                    state,
                    {cache.names[name]: path for name, path in files.items()},
                    progress,
                    cache=cache,
                )

            self.assertTrue(progress.complete)
            self.assertEqual(progress.indexed_count, 3)
            self.assertEqual(state.dataset_id, "existing-dataset")
            self.assertEqual(
                state.documents,
                {
                    "existing-id": "existing.txt",
                    "pending-id": "pending.txt",
                    "new-id": new_name,
                },
            )
            self.assertEqual(
                writes,
                [
                    ("POST", "/api/v1/datasets/existing-dataset/documents"),
                    ("POST", "/api/v1/datasets/existing-dataset/chunks"),
                ],
            )
            restarted = prepare_index_cache(root, "collection", "", files, {})
            self.assertEqual(restarted.manifest.files["new.txt"].remote_id, "new-id")

    # Verify retirement candidates by identity and do not repeat a committed delete.
    async def test_retirement_deletes_only_the_verified_document(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = {name: root / name for name in ("obsolete.txt", "collision.txt")}
            for path in files.values():
                path.write_text(path.stem, encoding="utf-8")
            first = prepare_index_cache(root, "collection", "", files, {})
            old = ragflow.Document(id="old-id", name="obsolete(1).txt")
            unrelated = ragflow.Document(id="unrelated-id", name="collision.txt")
            remote = {item.name: item for item in (old, unrelated)}
            ragflow.record_cached_documents(first, {"obsolete.txt": old})
            first.save()
            cache = prepare_index_cache(root, "collection", "", {}, {})
            cache.save()
            deleted: list[str] = []
            documents_path = "/datasets/existing-dataset/documents"

            # Leave a same-named unrelated document whose bytes differ from the source.
            def provider(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                collection = f"/api/v1{documents_path}"
                if request.method == "GET" and path == f"{collection}/{unrelated.id}":
                    return httpx.Response(200, content=b"unrelated bytes")
                if request.method == "GET" and path == collection:
                    data: object = {
                        "docs": [item.model_dump() for item in remote.values()],
                        "total": len(remote),
                    }
                elif request.method == "DELETE" and path == collection:
                    self.assertEqual(loads(request.read()), {"ids": [old.id]})
                    deleted.append(remote.pop(old.name).id)
                    data = {"deleted": 1}
                else:
                    self.fail(f"Unexpected provider call {request.method} {path}")
                return httpx.Response(200, json={"code": 0, "data": data})

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(provider)
            ) as client:
                state = ragflow.RagflowSession(client, authorization="test-token")
                remaining = await ragflow.retire_cached_documents(
                    state, documents_path, remote.copy(), cache
                )
                restarted = prepare_index_cache(root, "collection", "", {}, {})
                remaining_after_retry = await ragflow.retire_cached_documents(
                    state, documents_path, remaining, restarted
                )

            self.assertEqual(deleted, [old.id])
            self.assertEqual(remaining_after_retry, {unrelated.name: unrelated})
            self.assertFalse(restarted.manifest.retired)

    # Recover a committed document before sending a second upload.
    async def test_retry_only_missing_documents(self) -> None:
        files = {"saved.txt": Path("saved.txt"), "missing.txt": Path("missing.txt")}
        saved = ragflow.Document(id="saved", name="saved.txt")
        missing = ragflow.Document(id="retry", name="missing.txt")
        async with httpx.AsyncClient() as client:
            state = ragflow.RagflowSession(client)
            with (
                patch.object(
                    ragflow,
                    "upload_documents",
                    new_callable=AsyncMock,
                    side_effect=[
                        ragflow.RagflowError(102, "Partial upload"),
                        {"missing.txt": missing},
                    ],
                ) as upload,
                patch.object(
                    ragflow,
                    "find_document",
                    new_callable=AsyncMock,
                    side_effect=lambda state, path, name: (
                        saved if name == "saved.txt" else None
                    ),
                ),
            ):
                uploaded = await ragflow.upload_batch(state, "/documents", files)

            self.assertEqual(uploaded, {"saved.txt": saved, "missing.txt": missing})
            self.assertEqual(upload.await_count, 2)
            self.assertEqual(
                upload.await_args_list[1].args[2], {"missing.txt": files["missing.txt"]}
            )

    # Keep directory metadata reads off the event loop during startup.
    async def test_scan_batches_off_event_loop(self) -> None:
        files = {"source.txt": Path("source.txt")}
        source = ragflow.Document(
            id="done", name="source.txt", ingestion_status="COMPLETED", chunk_count=1
        )
        event_loop_thread = get_ident()

        # Record where the filesystem work would be performed.
        def batches(paths: dict[str, Path]) -> Iterator[dict[str, Path]]:
            self.assertNotEqual(get_ident(), event_loop_thread)
            yield paths

        async with httpx.AsyncClient() as client:
            state = ragflow.RagflowSession(client)
            progress = ragflow.IndexProgress()
            with (
                patch.object(ragflow, "dataset_name", return_value="dataset"),
                patch.object(
                    ragflow,
                    "ensure_dataset",
                    new_callable=AsyncMock,
                    return_value="dataset",
                ),
                patch.object(
                    ragflow,
                    "list_documents",
                    new_callable=AsyncMock,
                    side_effect=[{}, {"source.txt": source}],
                ),
                patch.object(ragflow, "upload_batches", side_effect=batches),
                patch.object(
                    ragflow,
                    "upload_batch",
                    new_callable=AsyncMock,
                    return_value={"source.txt": source},
                ),
                patch.object(ragflow, "queue_documents", new_callable=AsyncMock),
            ):
                await ragflow.index_sources(state, files, progress)

        self.assertTrue(progress.complete)
        self.assertEqual(state.documents, {"done": "source.txt"})

    # Keep the recovery queue alive while the primary queue is making progress.
    async def test_recovery_uses_shared_ingestion_activity(self) -> None:
        name = "recovery.txt"
        pending = ragflow.Document(
            id="recovery", name=name, ingestion_status="SCHEDULED"
        )
        indexed = pending.model_copy(
            update={"ingestion_status": "COMPLETED", "chunk_count": 1}
        )
        queued = {name: pending}
        elapsed = ragflow.STALL_TIMEOUT_SECONDS + 2.0

        # Poll an unchanged recovery queue until it finishes or the worker stalls.
        async def poll(primary_advances: bool) -> ragflow.IndexProgress:
            async with httpx.AsyncClient() as client:
                state = ragflow.RagflowSession(client)
                progress = ragflow.IndexProgress()

                # Advance time and account for activity in the primary queue.
                async def advance_clock(_: float) -> None:
                    clock.return_value = elapsed
                    if primary_advances:
                        state.last_ingestion_activity = elapsed - 1.0

                with (
                    patch.object(ragflow, "dataset_name", return_value="dataset"),
                    patch.object(ragflow, "ensure_dataset", return_value="recovery"),
                    patch.object(
                        ragflow,
                        "list_documents",
                        side_effect=[queued, queued, queued, {name: indexed}],
                    ),
                    patch.object(ragflow, "monotonic", return_value=0.0) as clock,
                    patch.object(asyncio, "sleep", side_effect=advance_clock),
                    patch.object(
                        ragflow,
                        "request",
                        side_effect=AssertionError("unexpected network request"),
                    ),
                ):
                    await ragflow.index_sources(
                        state, {name: Path(name)}, progress, supplemental="recovery"
                    )
                return progress

        progress = await poll(primary_advances=True)
        self.assertTrue(progress.complete)
        self.assertEqual(progress.indexed_count, 1)
        with self.assertRaisesRegex(RuntimeError, "ingestion made no progress"):
            await poll(primary_advances=False)

    # Count zero-chunk completions as failures and leave scheduled files out of search.
    async def test_publish_only_searchable_documents(self) -> None:
        files = {
            name: Path(name) for name in ("indexed.txt", "blank.pdf", "queued.pdf")
        }
        documents = {
            "indexed.txt": ragflow.Document(
                id="indexed",
                name="indexed.txt",
                ingestion_status="COMPLETED",
                chunk_count=3,
            ),
            "blank.pdf": ragflow.Document(
                id="blank",
                name="blank.pdf",
                ingestion_status="COMPLETED",
                chunk_count=0,
            ),
            "queued.pdf": ragflow.Document(
                id="queued",
                name="queued.pdf",
                ingestion_status="SCHEDULED",
            ),
        }
        progress = ragflow.IndexProgress()
        sync = ragflow.ImportState(progress, files, documents)
        async with httpx.AsyncClient() as client:
            state = ragflow.RagflowSession(client)
            ragflow.report_index(state, sync)

        self.assertEqual(state.documents, {"indexed": "indexed.txt"})
        self.assertEqual(
            (progress.uploaded_count, progress.indexed_count, progress.failed_count),
            (3, 1, 1),
        )


if __name__ == "__main__":
    main()
