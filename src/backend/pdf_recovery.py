'''Prepare scans with misaligned PDF indexes without changing originals.'''

import re
import subprocess
from os import SEEK_END
from pathlib import Path
from tempfile import TemporaryDirectory

from src.backend.source_recovery import recovery_folder

PDF_TAIL_BYTES = 64 * 1024
PDF_LINE_END = rb"(?:\r\n|\r|\n)"
PDF_HEADER = re.compile(rb"\A%PDF-[0-9]\.[0-9][ \t]*" + PDF_LINE_END)
NORMAL_PDF_HEADER = re.compile(rb"\A%PDF-[0-9]\.[0-9]\r?\n")
FINAL_STARTXREF = re.compile(
    PDF_LINE_END.join(
        (
            rb"(?:\A|[\r\n])[ \t]*startxref[ \t]*",
            rb"[ \t]*([0-9]{1,20})[ \t]*",
            rb"[ \t]*%%EOF[\x00\f\r\n\t ]*\Z",
        )
    )
)
SKEWED_XREF = re.compile(
    PDF_LINE_END.join(
        (
            rb"\Axref[ \t]*",
            rb"[ \t]*([0-9]{1,20})[ \t]+([0-9]{1,20})[ \t]*",
            rb"[ \t]*0000000000[ \t]+65535[ \t]+f[ \t]*",
            rb"[ \t]*([0-9]{10})[ \t]+([0-9]{5})[ \t]+n[ \t]*",
            b"",
        )
    )
)
OBJECT_HEADER = re.compile(rb"\A([0-9]{1,20})[ \t]+([0-9]{1,5})[ \t]+obj(?=[ \t\r\n])")


# Inspect the last classic xref for a misplaced object-zero sentinel.
# The first in-use entry must point to the subsection's first object, one ID
# lower than the xref row declares. Valid CR-only PDFs pass through.
def has_shifted_pdf_index(source: Path) -> bool:
    with source.open("rb") as document:
        if not PDF_HEADER.match(document.read(64)):
            return False
        size = document.seek(0, SEEK_END)
        document.seek(max(0, size - PDF_TAIL_BYTES))
        start = FINAL_STARTXREF.search(document.read(PDF_TAIL_BYTES))
        if start is None or not 0 < int(start[1]) < size:
            return False

        document.seek(int(start[1]))
        table = SKEWED_XREF.match(document.read(512))
        if table is None:
            return False
        first, count, offset, generation = map(int, table.groups())
        if first == 0 or count < 2 or not 0 < offset < size:
            return False

        document.seek(offset)
        actual = OBJECT_HEADER.match(document.read(64))
        return (
            actual is not None
            and int(actual[1]) == first
            and int(actual[2]) == generation
        )


# Check that a cached or newly written derivative has a normal header and EOF.
def complete_normalized_pdf(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as document:
        if not NORMAL_PDF_HEADER.match(document.read(16)):
            return False
        size = document.seek(0, SEEK_END)
        document.seek(max(0, size - 1024))
        return document.read(1024).rstrip().endswith(b"%%EOF")


# Rewrite confirmed index defects using Ghostscript's normal repair mode.
# Keep images at source resolution and use lossless compression where needed.
def normalize_pdf(source: Path, cache: Path) -> Path:
    if not has_shifted_pdf_index(source):
        return source

    folder = recovery_folder(source, cache, "pdf-normalized-v1")
    output = folder / "document.pdf"
    if complete_normalized_pdf(output):
        return output

    with TemporaryDirectory(prefix=".rewrite-", dir=folder) as directory:
        temporary = Path(directory) / "document.pdf"
        subprocess.run(
            [
                "/usr/bin/gs",
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                "-dAutoRotatePages=/None",
                "-dDownsampleColorImages=false",
                "-dDownsampleGrayImages=false",
                "-dDownsampleMonoImages=false",
                "-dPassThroughJPEGImages=true",
                "-dPassThroughJPXImages=true",
                "-dAutoFilterColorImages=false",
                "-dColorImageFilter=/FlateEncode",
                "-dAutoFilterGrayImages=false",
                "-dGrayImageFilter=/FlateEncode",
                f"-sOutputFile={temporary.resolve()}",
                "-f",
                str(source.resolve()),
            ],
            check=True,
            timeout=180,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not complete_normalized_pdf(temporary):
            raise RuntimeError("Ghostscript did not produce a complete normalized PDF.")
        temporary.replace(output)

    return output
