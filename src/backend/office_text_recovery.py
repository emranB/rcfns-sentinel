'''Recover Office text, preserve Word outlines, and identify empty files.'''

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import Element, ParseError, fromstring
from zipfile import BadZipFile, ZipFile

MAX_XML_BYTES = 32 * 1024 * 1024
MAX_WORD_OUTLINE_XML_BYTES = 2 * 1024 * 1024
MAX_HTML_BYTES = 256 * 1024
WORD_BODY = "word/document.xml"
WORD_MARGINS = re.compile(r"word/(?:header|footer)[^/]*\.xml")
SHEETS = re.compile(r"xl/worksheets/[^/]+\.xml")
SLIDES = re.compile(r"ppt/slides/slide[0-9]+\.xml")
MEDIA_DIRS = ("word/media/", "xl/media/", "ppt/media/")
EMBEDDED_DIRS = ("word/embeddings/", "xl/embeddings/", "ppt/embeddings/")


@dataclass(frozen=True)
class OfficeTextRecovery:
    text: str = ""
    no_content_reason: str | None = None
    needs_image_recovery: bool = False


class LoginPage(HTMLParser):
    # Track the visible title and detect content beyond an empty page shell.
    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.in_title = False
        self.in_body = False
        self.has_body = False
        self.has_content = False
        self.hidden: list[str] = []

    # Ignore bootstrap markup when deciding whether the body contains content.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.in_title = True
        if tag == "body":
            self.in_body = self.has_body = True
        if tag in {"script", "style", "template", "noscript"}:
            self.hidden.append(tag)
        if (
            self.in_body
            and not self.hidden
            and tag
            in {"iframe", "object", "embed", "canvas", "img", "video", "audio", "a"}
        ):
            self.has_content = True

    # Leave a captured section at its closing tag.
    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "body":
            self.in_body = False
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()

    # Capture title text and recognize visible content in the body.
    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        if self.in_body and not self.hidden and data.strip():
            self.has_content = True


# Recognize an empty sign-in shell while keeping HTML with visible content.
def classify_html(path: Path) -> OfficeTextRecovery:
    with path.open("rb") as source:
        payload = source.read(MAX_HTML_BYTES + 1)
    if len(payload) > MAX_HTML_BYTES:
        return OfficeTextRecovery()
    page = LoginPage()
    page.feed(payload.decode("utf-8-sig"))
    page.close()
    title = " ".join("".join(page.title).split()).casefold()
    login_title = re.search(r"(?:^|\s[-|]\s)(?:sign in|log in|login)$", title)
    if page.has_body and not page.has_content and login_title:
        return OfficeTextRecovery(no_content_reason="Saved sign-in page")
    return OfficeTextRecovery()


# Read only the selected XML parts and bound their total decompressed size.
def read_parts(
    archive: ZipFile, names: list[str], *, limit: int = MAX_XML_BYTES
) -> list[Element]:
    if sum(archive.getinfo(name).file_size for name in names) > limit:
        raise ValueError("Office XML exceeds the recovery size limit.")
    return [fromstring(archive.read(name)) for name in names]


# Match Office element names across transitional and strict namespaces.
def tag_name(element: Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


# Preserve run order, spaces, and line breaks without duplicating nested text boxes.
def paragraph_text(paragraph: Element) -> str:
    parts: list[str] = []
    pending = list(reversed(paragraph))
    while pending:
        run = pending.pop()
        tag = tag_name(run)
        if tag == "p":
            continue
        if tag == "t":
            parts.append(run.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
        pending.extend(reversed(run))
    return "".join(parts).strip()


# Read the first line's explicit visual offset in twips from Word XML.
def word_indent(paragraph: Element, namespace: str) -> int | None:
    indent = paragraph.find(f"{namespace}pPr/{namespace}ind")
    if indent is None:
        return 0
    offset = indent.get(f"{namespace}start", indent.get(f"{namespace}left", "0"))
    first_line = indent.get(f"{namespace}firstLine", "0")
    try:
        return max(0, int(offset) + int(first_line))
    except ValueError:
        return None


# Restore layout only for complete excerpt lines with an unambiguous original indent.
def restore_word_outline(payload: bytes, text: str) -> str:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            (root,) = read_parts(archive, [WORD_BODY], limit=MAX_WORD_OUTLINE_XML_BYTES)
    except (BadZipFile, KeyError, OSError, ParseError, RuntimeError, ValueError):
        return text
    if tag_name(root) != "document":
        return text

    lines = [(line, " ".join(line.split())) for line in text.splitlines(keepends=True)]
    requested = {key for _, key in lines if key}
    namespace = root.tag.removesuffix("document")
    indents: dict[str, int | None] = {}
    for paragraph in root.iter(f"{namespace}p"):
        key = " ".join(paragraph_text(paragraph).split())
        if key in requested:
            offset = word_indent(paragraph, namespace)
            indents[key] = offset if indents.get(key, offset) == offset else None

    result = []
    for line, key in lines:
        offset = indents.get(key)
        if offset and not line.startswith((" ", "\t")):
            # Keep indentation proportional and bounded for the prompt.
            line = " " * min(24, (offset + 179) // 180) + line
        result.append(line)
    return "".join(result)


# Recover literal Word body and margin text before assessing blank or visual content.
def recover_word(archive: ZipFile, names: set[str], media: bool) -> OfficeTextRecovery:
    if WORD_BODY not in names:
        return OfficeTextRecovery()
    margins = sorted(name for name in names if WORD_MARGINS.fullmatch(name))
    parts = [name for name in margins if name.startswith("word/header")]
    parts.append(WORD_BODY)
    parts.extend(name for name in margins if name.startswith("word/footer"))
    roots = read_parts(archive, parts)
    if tag_name(roots[parts.index(WORD_BODY)]) != "document":
        return OfficeTextRecovery()
    paragraphs = [
        text
        for root in roots
        for element in root.iter()
        if tag_name(element) == "p" and (text := paragraph_text(element))
    ]
    if paragraphs:
        return OfficeTextRecovery(text="\n".join(paragraphs))
    tags = {tag_name(element) for root in roots for element in root.iter()}
    if media or tags & {"drawing", "pict", "object"}:
        return OfficeTextRecovery(needs_image_recovery=True)
    if (
        tags & {"altChunk", "instrText", "fldSimple"}
        or any(
            tag_name(element) == "t" and bool((element.text or "").strip())
            for root in roots
            for element in root.iter()
        )
        or any(
            name in names
            for name in ("word/footnotes.xml", "word/endnotes.xml", "word/comments.xml")
        )
    ):
        return OfficeTextRecovery()
    return OfficeTextRecovery(no_content_reason="Blank Word document")


# Read a bounded Word body for a question about the text of a specific original.
def word_body(payload: bytes) -> str:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            (root,) = read_parts(archive, [WORD_BODY], limit=MAX_WORD_OUTLINE_XML_BYTES)
    except (BadZipFile, KeyError, OSError, ParseError, RuntimeError, ValueError):
        return ""
    if tag_name(root) != "document":
        return ""
    return "\n".join(
        text
        for element in root.iter()
        if tag_name(element) == "p" and (text := paragraph_text(element))
    )


# Require real worksheets and rule out formulas, values, comments, and visual content.
def classify_workbook(
    archive: ZipFile, names: set[str], media: bool
) -> OfficeTextRecovery:
    sheets = sorted(name for name in names if SHEETS.fullmatch(name))
    if "xl/workbook.xml" not in names or not sheets:
        return OfficeTextRecovery()
    if media or any(name.startswith(("xl/charts/", "xl/drawings/")) for name in names):
        return OfficeTextRecovery(needs_image_recovery=True)
    if any(
        name.startswith(("xl/comments", "xl/threadedComments/", "xl/externalLinks/"))
        for name in names
    ):
        return OfficeTextRecovery()
    roots = read_parts(archive, ["xl/workbook.xml", *sheets])
    if tag_name(roots[0]) != "workbook" or any(
        tag_name(sheet) != "worksheet" for sheet in roots[1:]
    ):
        return OfficeTextRecovery()
    if any(
        tag_name(element) == "f"
        or (tag_name(element) in {"v", "t"} and bool((element.text or "").strip()))
        for sheet in roots[1:]
        for element in sheet.iter()
    ):
        return OfficeTextRecovery()
    return OfficeTextRecovery(no_content_reason="Blank Excel workbook")


# Treat image and drawing slides as visual sources even when they contain no native text.
def classify_presentation(
    archive: ZipFile, names: set[str], media: bool
) -> OfficeTextRecovery:
    slides = sorted(name for name in names if SLIDES.fullmatch(name))
    if "ppt/presentation.xml" not in names or not slides:
        return OfficeTextRecovery()
    if media or any(
        name.startswith(("ppt/charts/", "ppt/diagrams/")) for name in names
    ):
        return OfficeTextRecovery(needs_image_recovery=True)
    if any(name.startswith(("ppt/notesSlides/", "ppt/comments/")) for name in names):
        return OfficeTextRecovery()
    roots = read_parts(archive, ["ppt/presentation.xml", *slides])
    if tag_name(roots[0]) != "presentation" or any(
        tag_name(slide) != "sld" for slide in roots[1:]
    ):
        return OfficeTextRecovery()
    shapes = {tag_name(element) for slide in roots[1:] for element in slide.iter()}
    if shapes & {"pic", "graphicFrame", "cxnSp", "grpSp", "oleObj"}:
        return OfficeTextRecovery(needs_image_recovery=True)
    if any(
        (tag_name(element) == "t" and bool((element.text or "").strip()))
        or (
            tag_name(element) == "sp"
            and all(tag_name(child) != "ph" for child in element.iter())
        )
        for slide in roots[1:]
        for element in slide.iter()
    ):
        return OfficeTextRecovery()
    return OfficeTextRecovery(no_content_reason="Blank PowerPoint presentation")


# Recover only verifiable text or an explicit no-content result.
def recover_office_text(path: Path) -> OfficeTextRecovery:
    kind = path.suffix.lower()
    try:
        if kind in {".html", ".htm"}:
            return classify_html(path)
        if kind not in {".docx", ".xlsx", ".pptx"}:
            return OfficeTextRecovery()
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            if any(
                name.startswith(EMBEDDED_DIRS) and not name.endswith("/")
                for name in names
            ):
                return OfficeTextRecovery()
            media = any(
                name.startswith(MEDIA_DIRS) and not name.endswith("/") for name in names
            )
            match kind:
                case ".docx":
                    return recover_word(archive, names, media)
                case ".xlsx":
                    return classify_workbook(archive, names, media)
                case ".pptx":
                    return classify_presentation(archive, names, media)
    except (BadZipFile, KeyError, OSError, ParseError, RuntimeError, ValueError):
        return OfficeTextRecovery()
    return OfficeTextRecovery()
