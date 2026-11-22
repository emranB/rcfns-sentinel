'''Verify recovery keeps exact originals and honors the inventory.'''

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch
from zipfile import ZipFile

import httpx

from src.backend import ragflow, server
from src.backend.source_recovery import (
    expand_archive,
    prepare_supplemental,
    recovery_folder,
)
from src.backend.sources import local_sources


class ArchiveTests(TestCase):
    # Keep a normalized upload tied to the original document path.
    def test_pdf_upload_uses_derivative_and_keeps_original(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            original = root / "Agreement.pdf"
            original.write_bytes(b"original bytes")
            modified = root / "Compatible.pdf"
            modified.write_bytes(b"normalized bytes")
            notes = root / "Notes.txt"
            files = {original.name: original, notes.name: notes}
            with patch.object(
                server, "normalize_pdf", return_value=modified
            ) as normalize:
                originals = server.prepare_pdf_sources(files)
            normalize.assert_called_once()
            self.assertEqual(files, {original.name: modified, notes.name: notes})
            self.assertEqual(originals, {original.name: original})
            self.assertEqual(original.read_bytes(), b"original bytes")

    # Stage documents and keep direct and archived recordings as reference paths.
    def test_archive_and_references(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            data = root / "data"
            data.mkdir()
            archive = data / "reports.zip"
            with ZipFile(archive, "w") as source:
                source.writestr("board/minutes.txt", "Approved operating plan")
                source.writestr("board/meeting.MOV", "video")
            (data / "cloud.url").write_text("[InternetShortcut]", encoding="utf-8")
            (data / "recording.m4a").write_bytes(b"audio")
            (data / "event.mp4").write_bytes(b"video")

            prepared = prepare_supplemental(
                local_sources(data), root, root / "scratch", root / "cache"
            )
            name, staged = next(iter(prepared.files.items()))
            self.assertEqual(staged.read_bytes(), b"Approved operating plan")
            self.assertEqual(
                prepared.labels[name], "data/reports.zip!/board/minutes.txt"
            )
            self.assertEqual(
                {item.path: item.kind for item in prepared.inventory},
                {
                    "data/cloud.url": "reference",
                    "data/event.mp4": "reference",
                    "data/recording.m4a": "reference",
                    "data/reports.zip!/board/meeting.MOV": "reference",
                },
            )
            self.assertEqual(list(staged.parent.iterdir()), [staged])
            self.assertEqual(prepared.unresolved, [])

    # Reject parent traversal and leave the files outside scratch untouched.
    def test_reject_unsafe_archive_paths(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            archive = root / "unsafe.zip"
            with ZipFile(archive, "w") as source:
                source.writestr("../escape.txt", "Should never leave scratch")
            with self.assertRaises(ValueError):
                expand_archive(archive, root / "staged")
            self.assertFalse((root / "escape.txt").exists())


class AliasTests(IsolatedAsyncioTestCase):
    # Reconcile only the alias whose archived bytes match the exact local original.
    async def test_alias_selects_verified_original(self) -> None:
        with TemporaryDirectory() as work:
            path = Path(work) / "Original.txt"
            path.write_bytes(b"original local content")
            size, expected = ragflow.source_digest(path)
            candidates = {
                "Original(1).txt": ragflow.Document(
                    id="one", name="Original(1).txt", size=size
                ),
                "Original(2).txt": ragflow.Document(
                    id="two", name="Original(2).txt", size=size
                ),
            }
            async with httpx.AsyncClient() as client:
                state = ragflow.RagflowSession(client)
                with patch.object(
                    ragflow,
                    "stored_digest",
                    AsyncMock(side_effect=[b"not the original", expected]),
                ):
                    matched = await ragflow.match_documents(
                        state, "/documents", {path.name: path}, candidates
                    )
            self.assertEqual(matched[path.name].id, "two")


class ImageRecoveryTests(IsolatedAsyncioTestCase):
    # Count prepared image content as queued and retain genuinely unresolved files.
    async def test_prepared_image_leaves_recovery_inventory(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            photo = root / "photo.heic"
            photo.write_bytes(b"image")
            source = local_sources(root)
            cached = recovery_folder(photo, root / "recovery", "media-v1")
            (cached / "content.md").write_text("Image text", encoding="utf-8")
            library = server.Library()
            library.inventory.needs_recovery = ["photo.heic", "unsupported.ext"]
            library.native.skipped_count = 2

            async with httpx.AsyncClient() as client:
                state = ragflow.RagflowSession(client)
                with (
                    patch.object(server, "STATE_DIR", root),
                    patch.object(
                        server, "index_sources", new_callable=AsyncMock
                    ) as queue,
                ):
                    await server.index_recovered_images(library, state, source, root)
                queue.assert_awaited_once()

            self.assertEqual(library.inventory.needs_recovery, ["unsupported.ext"])
            self.assertEqual(library.native.skipped_count, 1)
