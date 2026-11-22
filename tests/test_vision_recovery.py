'''Verify bounded local image recovery and PowerPoint slide order.'''

import asyncio
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, call, patch
from zipfile import ZipFile

import httpx
from PIL import Image

from src.backend import server, vision_recovery
from src.backend.ragflow import IndexProgress, RagflowSession
from src.backend.source_recovery import PreparedFiles
from src.backend.sources import SourceCollection
from src.backend.vision_recovery import (
    VisionRecoveryError,
    load_images,
    presentation_images,
    recover_vision,
    validate_local_model,
)


class VisionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    # Keep two image recoveries in flight and write each chunk once after failover.
    async def test_native_recovery_overlaps_and_falls_back(self) -> None:
        primary = "http://localhost:11434"
        secondary = "http://localhost:25134"
        primary_started = asyncio.Event()
        secondary_started = asyncio.Event()
        inferences: list[tuple[str, str, int | None]] = []

        # Hold the first inference until the secondary starts, then fail the secondary.
        async def infer(
            _client: httpx.AsyncClient,
            source: Path,
            destination: Path,
            *,
            ollama_url: str,
            model: str,
            threads: int | None = None,
        ) -> Path:
            inferences.append((source.stem, ollama_url, threads))
            if ollama_url == secondary:
                await primary_started.wait()
                secondary_started.set()
                raise httpx.ConnectError("Secondary endpoint unavailable")
            if source.stem == "first":
                primary_started.set()
                await secondary_started.wait()
            destination.write_text(source.stem, encoding="utf-8")
            return destination

        documents = {"one": "first.png", "two": "second.png"}
        progress = IndexProgress(total_count=2, failed_count=2, complete=True)
        with TemporaryDirectory() as directory:
            scratch = Path(directory)
            sources = SourceCollection(
                {name: scratch / name for name in documents.values()}
            )
            async with httpx.AsyncClient() as client:
                state = RagflowSession(
                    client, dataset_id="native", without_chunks=documents
                )
                with (
                    patch.multiple(
                        server,
                        OLLAMA_URL=primary,
                        OLLAMA_SECONDARY_VISION_URL=secondary,
                        OLLAMA_SECONDARY_VISION_THREADS=2,
                        recover_vision=infer,
                    ),
                    patch.object(server, "request", new_callable=AsyncMock) as request,
                    self.assertLogs(server.LOGGER, "WARNING"),
                ):
                    await asyncio.wait_for(
                        server.recover_native(
                            server.Library(native=progress),
                            state,
                            sources,
                            PreparedFiles(),
                            scratch,
                        ),
                        timeout=3,
                    )
                self.assertCountEqual(
                    inferences,
                    [
                        ("first", primary, None),
                        ("second", secondary, 2),
                        ("second", primary, None),
                    ],
                )
                self.assertCountEqual(
                    request.await_args_list,
                    [
                        call(
                            state,
                            "POST",
                            "/datasets/native/documents/one/chunks",
                            json={"content": "first"},
                        ),
                        call(
                            state,
                            "POST",
                            "/datasets/native/documents/two/chunks",
                            json={"content": "second"},
                        ),
                    ],
                )
                self.assertEqual(state.documents, documents)
                self.assertEqual(
                    (progress.indexed_count, progress.failed_count), (2, 0)
                )

    # Recover after a temporary local model error without changing the source image.
    async def test_retries_temporary_error_and_writes_derived_markdown(self) -> None:
        calls: list[str] = []

        # Fail the first inference and succeed on retry.
        def send(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            self.assertEqual(request.url.host, "host.docker.internal")
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["vision"]})
            self.assertIn(b'"images"', request.content)
            if len(calls) == 2:
                return httpx.Response(500, json={"error": "Temporary local error"})
            return httpx.Response(
                200, json={"message": {"content": "Visible text: Library hours"}}
            )

        with TemporaryDirectory() as directory:
            source = Path(directory) / "hours.png"
            output = Path(directory) / "recovered.md"
            Image.new("RGBA", (40, 24), (12, 24, 50, 0)).save(source)
            before = source.read_bytes()
            async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
                with patch.object(vision_recovery, "VISION_RETRY_DELAYS", (0, 0)):
                    path = await recover_vision(
                        client,
                        source,
                        output,
                        ollama_url="http://host.docker.internal:11434",
                        model="local:4b",
                    )
            self.assertEqual(path, output)
            self.assertEqual(calls, ["/api/show", "/api/chat", "/api/chat"])
            self.assertIn(
                "Visible text: Library hours", output.read_text(encoding="utf-8")
            )
            self.assertEqual(source.read_bytes(), before)

    # Surface a rejected inference immediately without writing a partial result.
    async def test_does_not_retry_bad_request(self) -> None:
        calls: list[str] = []

        # Return a non-retryable error once the model has been validated.
        def send(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"capabilities": ["vision"]})
            return httpx.Response(400, json={"error": "Invalid image request"})

        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.png"
            output = Path(directory) / "recovered.md"
            Image.new("RGB", (20, 20), "white").save(source)
            async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
                with (
                    patch.object(vision_recovery, "VISION_RETRY_DELAYS", (0, 0)),
                    self.assertRaises(httpx.HTTPStatusError) as raised,
                ):
                    await recover_vision(
                        client,
                        source,
                        output,
                        ollama_url="http://localhost:11434",
                        model="local:4b",
                    )
            self.assertEqual(raised.exception.response.status_code, 400)
            self.assertEqual(calls, ["/api/show", "/api/chat"])
            self.assertFalse(output.exists())

    # Make sure an explicitly configured remote endpoint cannot receive the image.
    def test_rejects_remote_endpoint(self) -> None:
        with self.assertRaisesRegex(VisionRecoveryError, "local Ollama"):
            validate_local_model("https://example.test:11434", "local:4b")

    # Preserve PDF page order and reject overflow instead of indexing a partial document.
    def test_pdf_rasterizes_in_order_with_an_overflow_page(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scan.pdf"
            original = b"%PDF sample input"
            source.write_bytes(original)

            # Simulate a renderer that writes up to the configured overflow page.
            def render(command: list[str], **options: object) -> None:
                self.assertIn("-dLastPage=3", command)
                self.assertIn("-dPDFFitPage", command)
                self.assertEqual(
                    options["timeout"], vision_recovery.PDF_RENDER_TIMEOUT_SECONDS
                )
                output = next(
                    item.removeprefix("-sOutputFile=")
                    for item in command
                    if item.startswith("-sOutputFile=")
                )
                for number in reversed(range(1, page_count + 1)):
                    color = "red" if number == 1 else "blue"
                    Image.new("RGB", (20, 20), color).save(output % number)

            with (
                patch.object(vision_recovery, "MAX_PACKAGE_IMAGES", 2),
                patch("src.backend.vision_recovery.subprocess.run", side_effect=render),
            ):
                page_count = 2
                images = load_images(source)
                self.assertEqual(
                    [image.section for image in images], ["Page 1", "Page 2"]
                )
                with Image.open(BytesIO(images[0].jpeg)) as first:
                    pixel = first.getpixel((0, 0))
                    assert isinstance(pixel, tuple)
                    self.assertGreater(pixel[0], 240)
                page_count = 3
                with self.assertRaisesRegex(VisionRecoveryError, "page limit"):
                    load_images(source)
            self.assertEqual(source.read_bytes(), original)

    # Use presentation relationships for ordering instead of slide filename numbers.
    def test_follows_presentation_slide_order(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "slides.pptx"
            relationships = (
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
            )
            presentation = (
                '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                f'xmlns:r="{relationships[:-1]}">'
                '<p:sldIdLst><p:sldId r:id="s2"/><p:sldId r:id="s1"/></p:sldIdLst>'
                "</p:presentation>"
            )
            slide = (
                '<a:blip xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                f'xmlns:r="{relationships[:-1]}" r:embed="picture"/>'
            )
            with ZipFile(source, "w") as archive:
                archive.writestr("ppt/presentation.xml", presentation)
                archive.writestr(
                    "ppt/_rels/presentation.xml.rels",
                    '<Relationships>'
                    f'<Relationship Id="s1" Type="{relationships}slide" Target="slides/slide1.xml"/>'
                    f'<Relationship Id="s2" Type="{relationships}slide" Target="slides/slide2.xml"/>'
                    "</Relationships>",
                )
                for number, color in [(1, "red"), (2, "blue")]:
                    archive.writestr(f"ppt/slides/slide{number}.xml", slide)
                    archive.writestr(
                        f"ppt/slides/_rels/slide{number}.xml.rels",
                        "<Relationships>"
                        f'<Relationship Id="picture" Type="{relationships}image" Target="../media/{number}.png"/>'
                        "</Relationships>",
                    )
                    with BytesIO() as output:
                        Image.new("RGB", (20, 20), color).save(output, format="PNG")
                        archive.writestr(f"ppt/media/{number}.png", output.getvalue())
            images = presentation_images(source)
            self.assertEqual(
                [image.section for image in images],
                ["Slide 1, image 1", "Slide 2, image 1"],
            )
            with Image.open(BytesIO(images[0].jpeg)) as first:
                pixel = first.getpixel((0, 0))
                assert isinstance(pixel, tuple)
                self.assertGreater(pixel[2], 240)

    # Decode only unique raster content from the Word media folder.
    def test_reads_unique_word_media(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "document.docx"
            with ZipFile(source, "w") as archive, BytesIO() as output:
                Image.new("RGB", (20, 20), "red").save(output, format="PNG")
                archive.writestr("word/media/image1.png", output.getvalue())
                archive.writestr("word/media/image2.png", output.getvalue())
                archive.writestr("word/other.png", b"outside the media folder")
                archive.writestr("word/media/drawing.svg", b"vector image")
                archive.writestr("media/unreferenced.png", b"unreferenced")
                archive.writestr(
                    "word/_rels/document.xml.rels",
                    '<Relationships><Relationship Id="image" Type="urn:office/image" '
                    'Target="/media/linked.png"/></Relationships>',
                )
                with BytesIO() as linked:
                    Image.new("RGB", (20, 20), "blue").save(linked, format="PNG")
                    archive.writestr("media/linked.png", linked.getvalue())
            self.assertEqual(
                [image.section for image in load_images(source)],
                ["Document image 1", "Document image 2"],
            )
