'''Verify that PDF recovery only rewrites legacy structures and publishes atomically.'''

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.backend.pdf_recovery import normalize_pdf


# Build a PDF prefix and a traditional xref with actual byte offsets.
def pdf_sample(
    ending: bytes, subsection: int, object_id: int, generation: int
) -> bytes:
    head = ending.join((b"%PDF-1.3", b"%sample", b""))
    body = ending.join(
        (f"{object_id} {generation} obj".encode(), b"<< >>", b"endobj", b"")
    )
    return (
        head
        + body
        + ending.join(
            (
                b"xref",
                f"{subsection} 2".encode(),
                b"0000000000 65535 f ",
                f"{len(head):010d} {generation:05d} n ".encode(),
                b"trailer",
                b"<< >>",
                b"startxref",
                str(len(head + body)).encode(),
                b"%%EOF",
                b"",
            )
        )
    )


LEGACY = pdf_sample(b"\r", 4, 4, 7)
NORMAL = pdf_sample(b"\n", 0, 1, 0)


class PdfRecoveryTests(TestCase):
    # Keep valid index offsets untouched regardless of line-ending convention.
    def test_bypass_unaffected_files(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            source = root / "original.pdf"
            cache = root / "cache"
            examples = (
                NORMAL,
                pdf_sample(b"\r\n", 0, 1, 0),
                pdf_sample(b"\r", 0, 1, 0),
                pdf_sample(b"\r", 4, 5, 7),
                LEGACY.replace(b"4 7 obj", b"4 8 obj"),
            )
            with patch("src.backend.pdf_recovery.subprocess.run") as command:
                for data in examples:
                    with self.subTest(data=data):
                        source.write_bytes(data)
                        self.assertEqual(normalize_pdf(source, cache), source)
                        self.assertEqual(source.read_bytes(), data)
                command.assert_not_called()
            self.assertFalse(cache.exists())

    # Write a complete derivative once even if the same bytes have another name.
    def test_normalize_legacy_and_reuse_cache(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            source = root / "scan.pdf"
            other = root / "duplicate.pdf"
            source.write_bytes(LEGACY)
            other.write_bytes(LEGACY)
            cache = root / "cache"

            # Emulate a successful local rewrite inside the private staging directory.
            def rewrite(
                command: list[str], **options: object
            ) -> subprocess.CompletedProcess[bytes]:
                self.assertEqual(command[0], "/usr/bin/gs")
                self.assertIn("-dSAFER", command)
                self.assertIn("-dDownsampleMonoImages=false", command)
                self.assertIs(options["check"], True)
                destination = next(
                    argument.removeprefix("-sOutputFile=")
                    for argument in command
                    if argument.startswith("-sOutputFile=")
                )
                Path(destination).write_bytes(NORMAL)
                return subprocess.CompletedProcess(command, 0)

            with patch(
                "src.backend.pdf_recovery.subprocess.run", side_effect=rewrite
            ) as command:
                output = normalize_pdf(source, cache)
                self.assertEqual(normalize_pdf(other, cache), output)
                command.assert_called_once()

            self.assertEqual(output.name, "document.pdf")
            self.assertEqual(output.read_bytes(), NORMAL)
            self.assertEqual(list(output.parent.iterdir()), [output])
            self.assertEqual(source.read_bytes(), LEGACY)

    # Reject incomplete output even when the converter exits successfully.
    def test_missing_export_is_not_cached(self) -> None:
        with TemporaryDirectory() as work:
            root = Path(work)
            source = root / "scan.pdf"
            source.write_bytes(LEGACY)
            cache = root / "cache"

            with (
                patch("src.backend.pdf_recovery.subprocess.run"),
                self.assertRaisesRegex(RuntimeError, "complete normalized PDF"),
            ):
                normalize_pdf(source, cache)

            self.assertEqual(list(cache.rglob("*.pdf")), [])
            self.assertEqual(source.read_bytes(), LEGACY)
