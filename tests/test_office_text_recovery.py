'''Check literal recovery and explicit empty classification for Office sources.'''

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from src.backend.office_text_recovery import (
    named_slide,
    recover_office_text,
    restore_word_outline,
)


# Create a minimal test package without depending on an Office application.
def make_package(path: Path, parts: dict[str, str]) -> Path:
    with ZipFile(path, "w") as package:
        for name, content in parts.items():
            package.writestr(name, content)
    return path


class OfficeTextRecoveryTests(unittest.TestCase):
    # A question about a named slide cannot borrow figures from its neighbors.
    def test_named_slide_reads_only_the_unique_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "briefing.pptx"
            first = (
                "<sld><sp><p><r><t>Planned Work</t></r></p></sp>"
                "<sp><p><r><t>Inspect the valve</t></r></p></sp></sld>"
            )
            other = (
                "<sld><sp><p><r><t>Spending</t></r></p></sp>"
                "<sp><p><r><t>$8,000</t></r></p></sp></sld>"
            )
            parts = {
                "ppt/slides/slide1.xml": first,
                "ppt/slides/slide2.xml": other,
            }
            question = 'What is on the "Planned Work" slide?'
            self.assertEqual(
                named_slide(make_package(path, parts).read_bytes(), question),
                "Planned Work\n\nInspect the valve",
            )
            parts["ppt/slides/slide3.xml"] = first
            self.assertEqual(
                named_slide(make_package(path, parts).read_bytes(), question), ""
            )

    # Recover body, header, and footer in reading order without duplicating a text box.
    def test_word_text_is_literal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = make_package(
                Path(directory) / "letter.docx",
                {
                    "word/document.xml": (
                        "<document><p><r><t>Opening </t></r><r><t>line</t></r>"
                        "<drawing><p><r><t>Text box</t></r></p></drawing></p></document>"
                    ),
                    "word/footer1.xml": "<ftr><p><r><t>Footer</t></r></p></ftr>",
                    "word/header1.xml": "<hdr><p><r><t>Header</t></r></p></hdr>",
                },
            )
            self.assertEqual(
                recover_office_text(path).text, "Header\nOpening line\nText box\nFooter"
            )

    # Restore equivalent visual indents and keep ambiguous or partial lines unchanged.
    def test_word_outline_retains_verified_indentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = make_package(
                Path(directory) / "outline.docx",
                {
                    "word/document.xml": (
                        '<w:document xmlns:w="urn:word"><w:body>'
                        "<w:p><w:r><w:t>Heading</w:t></w:r></w:p>"
                        '<w:p><w:pPr><w:ind w:left="720"/></w:pPr>'
                        "<w:r><w:t>First child</w:t></w:r></w:p>"
                        '<w:p><w:pPr><w:ind w:firstLine="720"/></w:pPr>'
                        "<w:r><w:t>Second child</w:t></w:r></w:p>"
                        '<w:p><w:pPr><w:ind w:start="720" w:firstLine="720"/></w:pPr>'
                        "<w:r><w:t>Grandchild</w:t></w:r></w:p>"
                        '<w:p><w:pPr><w:ind w:left="720"/></w:pPr>'
                        "<w:r><w:t>Duplicate</w:t></w:r></w:p>"
                        "<w:p><w:r><w:t>Duplicate</w:t></w:r></w:p>"
                        "</w:body></w:document>"
                    ),
                },
            )
            text = "Heading\r\nFirst   child\r\nSecond child\r\nGrandchild\r\nDuplicate\r\nFirst\r\n"
            self.assertEqual(
                restore_word_outline(path.read_bytes(), text),
                "Heading\r\n    First   child\r\n    Second child\r\n"
                "        Grandchild\r\nDuplicate\r\nFirst\r\n",
            )

    # Retain the evidence if the original package or markup cannot be read.
    def test_word_outline_keeps_excerpt_when_markup_is_incomplete(self) -> None:
        text = "Heading\nItem"
        with tempfile.TemporaryDirectory() as directory:
            broken = make_package(
                Path(directory) / "broken.docx", {"word/document.xml": "<document>"}
            )
            self.assertEqual(restore_word_outline(broken.read_bytes(), text), text)
        self.assertEqual(restore_word_outline(b"invalid archive", text), text)

    # A truly empty workbook is different from a formula without a cached value.
    def test_blank_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = {"xl/workbook.xml": "<workbook/>"}
            key = "xl/worksheets/sheet1.xml"
            parts[key] = "<worksheet><c/></worksheet>"
            blank = make_package(root / "blank.xlsx", parts)
            self.assertEqual(
                recover_office_text(blank).no_content_reason, "Blank Excel workbook"
            )
            parts[key] = "<worksheet><c><f>1+2</f></c></worksheet>"
            formula = make_package(root / "formula.xlsx", parts)
            self.assertIsNone(recover_office_text(formula).no_content_reason)

    # Mark empty native structures without requiring zero-byte files.
    def test_blank_word_and_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_package(
                root / "blank.docx",
                {"word/document.xml": "<document><p><r><t> </t></r></p></document>"},
            )
            presentation = make_package(
                root / "blank.pptx",
                {
                    "ppt/presentation.xml": "<presentation/>",
                    "ppt/slides/slide1.xml": "<sld><sp><ph/></sp></sld>",
                },
            )
            self.assertEqual(
                recover_office_text(document).no_content_reason, "Blank Word document"
            )
            self.assertEqual(
                recover_office_text(presentation).no_content_reason,
                "Blank PowerPoint presentation",
            )

    # Image-only slides must reach the visual recovery path.
    def test_image_slide_needs_visual_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = make_package(
                Path(directory) / "image.pptx",
                {
                    "ppt/presentation.xml": "<presentation/>",
                    "ppt/slides/slide1.xml": "<sld><pic/></sld>",
                    "ppt/media/image.jpg": "test image bytes",
                },
            )
            recovered = recover_office_text(path)
            self.assertTrue(recovered.needs_image_recovery)
            self.assertIsNone(recovered.no_content_reason)

    # Classify the verified login shell without discarding same-titled content.
    def test_saved_login_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.html"
            header = "<head><title>CanadaHelps - Sign In</title></head>"
            path.write_text(
                header + "<body><div id='app'></div><script>start()</script></body>",
                encoding="utf-8",
            )
            self.assertEqual(
                recover_office_text(path).no_content_reason,
                "Saved sign-in page",
            )
            path.write_text(
                header + "<body><p>Saved transaction</p></body>", encoding="utf-8"
            )
            self.assertIsNone(recover_office_text(path).no_content_reason)
