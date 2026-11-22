"""Prepare readable backup content while retaining a link to each original."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass, field
from hashlib import file_digest, sha256
from pathlib import Path, PurePosixPath
from typing import IO, Literal
from xml.etree import ElementTree

from src.backend.office_conversion import (
    OFFICE_EXTENSIONS,
    convert_office,
    office_output_suffix,
)
from src.backend.ragflow import MAX_DOCUMENT_UPLOAD_SIZE
from src.backend.sources import SUPPORTED_EXTENSIONS, SourceCollection

LOGGER = logging.getLogger("uvicorn.error")
ARCHIVE_BYTES = 512 * 1024 * 1024
ARCHIVE_FILES = 1000
OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
REFERENCE_EXTENSIONS = frozenset(
    {
        ".inetloc",
        ".url",
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
        ".wma",
        ".3gp",
        ".avi",
        ".flv",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogv",
        ".webm",
        ".wmv",
    }
)
InventoryKind = Literal["locked", "reference", "artifact", "empty"]


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    kind: InventoryKind
    detail: str = ""


@dataclass
class PreparedFiles:
    files: dict[str, Path] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    originals: dict[str, Path] = field(default_factory=dict)
    inventory: list[InventoryEntry] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    native_inventory: set[str] = field(default_factory=set)

    # Register indexed content with its stable original name and download location.
    def add(self, label: str, prepared: Path, original: Path | None = None) -> None:
        name = Path(label).stem[:110]
        signature = sha256(label.encode("utf-8")).hexdigest()[:16]
        alias = f"{name} [{signature}]{prepared.suffix}"
        self.files[alias] = prepared
        self.labels[alias] = label
        if original is not None:
            self.originals[alias] = original


# Format a location consistently for both stored originals and archive members.
def original_label(path: Path, root: Path) -> str:
    return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()


# Verify Office encryption streams instead of treating a filename as a lock.
def locked_office(path: Path) -> bool:
    if path.suffix.lower() not in {".docx", ".xlsx", ".pptx", ".dotx", ".xltx"}:
        return False
    with path.open("rb") as source:
        if source.read(len(OLE_SIGNATURE)) != OLE_SIGNATURE:
            return False
        keys = {
            key.encode("utf-16le") + bytes(2)
            for key in ("EncryptionInfo", "EncryptedPackage")
        }
        tail = b""
        while block := source.read(65536):
            window = tail + block
            keys = {key for key in keys if key not in window}
            if not keys:
                return True
            tail = window[-64:]
        return False


# Recognize a Visio page with no shapes before using the converter.
def empty_drawing(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        pages = [
            item
            for item in archive.infolist()
            if item.filename.startswith("visio/pages/page")
            and item.filename.removesuffix(".xml").rsplit("page", 1)[-1].isdigit()
            and item.file_size <= 8 * 1024 * 1024
        ]
        return bool(pages) and all(
            ElementTree.fromstring(archive.read(page)).find(".//{*}Shape") is None
            for page in pages
        )


# Keep shortcuts, recordings, and temporary files visible as inventory records.
def inventory_kind(path: Path) -> InventoryKind | None:
    if path.name.startswith(("~$", "._")) or path.name.lower() in {
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
    }:
        return "artifact"
    if path.suffix.lower() in REFERENCE_EXTENSIONS:
        return "reference"
    if not path.stat().st_size or (
        path.suffix.lower() == ".vsdx" and empty_drawing(path)
    ):
        return "empty"
    if locked_office(path):
        return "locked"
    return None


# Store derivatives outside the source tree and reuse them only for identical bytes.
def recovery_folder(path: Path, cache: Path, purpose: str) -> Path:
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    folder = cache / f"{purpose}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# Rasterize an EPS locally using Ghostscript's restricted document mode.
def render_eps(path: Path, folder: Path) -> Path:
    output = folder / "preview.png"
    if not output.is_file():
        subprocess.run(
            [
                "gs",
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-dEPSCrop",
                "-sDEVICE=png16m",
                "-r96",
                f"-sOutputFile={output}",
                "-f",
                str(path),
            ],
            check=True,
            timeout=90,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not output.is_file():
        raise RuntimeError("EPS did not produce a local preview.")
    return output


# Use a cached derivative of an unchanged Office file across app restarts.
def office_derivative(path: Path, cache: Path) -> Path:
    folder = recovery_folder(path, cache, "office")
    # Retain the source suffix as part of the conversion key.
    output = folder / f"document-{path.suffix[1:].lower()}{office_output_suffix(path)}"
    return output if output.exists() else convert_office(path, output)


# Validate an archive member before recording its name or extracting it.
def archive_member(name: str, size: int) -> PurePosixPath | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
        or size < 0
    ):
        raise ValueError("Unsafe archive member.")
    if (
        path.name.startswith("._")
        or path.name in {".DS_Store", "Thumbs.db"}
        or "__MACOSX" in path.parts
    ):
        return None
    suffix = path.suffix.lower()
    if suffix in REFERENCE_EXTENSIONS:
        return path
    if size > MAX_DOCUMENT_UPLOAD_SIZE:
        raise ValueError("Oversized archive member.")
    if suffix not in SUPPORTED_EXTENSIONS | OFFICE_EXTENSIONS:
        raise ValueError(f"Unsupported archive member type {path.suffix}.")
    return path


# Copy to a generated local file; member paths never control the extraction target.
def copy_member(stream: IO[bytes], folder: Path, ordinal: int, suffix: str) -> Path:
    path = folder / f"{ordinal}{suffix}"
    with stream, path.open("wb") as output:
        shutil.copyfileobj(stream, output, length=1024 * 1024)
    return path


# Extract recoverable documents and retain names for the archive's reference files.
def expand_archive(
    path: Path, folder: Path
) -> tuple[list[tuple[str, Path]], list[str]]:
    folder.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, Path]] = []
    references: list[str] = []
    size = 0
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > ARCHIVE_FILES:
                raise ValueError("Archive exceeds the recovery limits.")
            for ordinal, zip_item in enumerate(members):
                member = archive_member(zip_item.filename, zip_item.file_size)
                if member is None:
                    continue
                if member.suffix.lower() in REFERENCE_EXTENSIONS:
                    references.append(member.as_posix())
                    continue
                size += zip_item.file_size
                if size > ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds the recovery limits.")
                files.append(
                    (
                        member.as_posix(),
                        copy_member(
                            archive.open(zip_item), folder, ordinal, member.suffix
                        ),
                    )
                )
    else:
        with tarfile.open(path, mode="r:") as archive:
            for ordinal, item in enumerate(archive):
                if ordinal >= ARCHIVE_FILES:
                    raise ValueError("Archive exceeds the recovery limits.")
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("Unsupported archive member.")
                member = archive_member(item.name, item.size)
                if member is None:
                    continue
                if member.suffix.lower() in REFERENCE_EXTENSIONS:
                    references.append(member.as_posix())
                    continue
                size += item.size
                if size > ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds the recovery limits.")
                stream = archive.extractfile(item)
                if stream is None:
                    raise ValueError("Could not read archive member.")
                files.append(
                    (
                        member.as_posix(),
                        copy_member(stream, folder, ordinal, member.suffix),
                    )
                )
    if not files and not references:
        raise ValueError("Archive contains no recoverable documents.")
    return files, references


# Identify encrypted originals without changing the native dataset fingerprint.
def catalog_native(sources: SourceCollection, root: Path) -> PreparedFiles:
    output = PreparedFiles()
    for name, path in sources.files.items():
        try:
            kind = (
                inventory_kind(path)
                if path.suffix.lower() in {".docx", ".xlsx", ".pptx"}
                else None
            )
        except (OSError, ValueError):
            LOGGER.warning(
                "Could not check protection for %s", path.name, exc_info=True
            )
            continue
        if kind == "locked":
            output.native_inventory.add(name)
            output.inventory.append(InventoryEntry(original_label(path, root), kind))

    return output


# Recover standalone Office files, archive members, and PDFs with a backup suffix.
def prepare_supplemental(
    sources: SourceCollection, root: Path, scratch: Path, cache: Path
) -> PreparedFiles:
    output = PreparedFiles()
    for number, path in enumerate(sources.excluded):
        label = original_label(path, root)
        try:
            kind = inventory_kind(path)
            suffix = path.suffix.lower()
            if kind:
                output.inventory.append(InventoryEntry(label, kind))
            elif suffix in OFFICE_EXTENSIONS:
                output.add(label, office_derivative(path, cache), path)
            elif suffix in {".zip", ".pkgf"}:
                entries, references = expand_archive(
                    path, scratch / f"archive-{number}"
                )
                prepared = [
                    (
                        name,
                        item,
                        office_derivative(item, cache)
                        if item.suffix.lower() in OFFICE_EXTENSIONS
                        else item,
                    )
                    for name, item in entries
                ]
                for name, original, document in prepared:
                    output.add(
                        f"{label}!/{name}",
                        document,
                        original if document != original else None,
                    )
                output.inventory.extend(
                    InventoryEntry(f"{label}!/{name}", "reference")
                    for name in references
                )
            elif ".pdf." in path.name.lower():
                with path.open("rb") as source:
                    if source.read(5) != b"%PDF-":
                        raise ValueError("Backup file does not contain a PDF.")
                folder = scratch / f"backup-{number}"
                folder.mkdir()
                recovered = folder / "document.pdf"
                shutil.copyfile(path, recovered)
                output.add(label, recovered, path)
            else:
                output.unresolved.append(label)
        except (
            ValueError,
            OSError,
            RuntimeError,
            zipfile.BadZipFile,
            tarfile.TarError,
        ):
            LOGGER.warning("Recovery could not prepare %s", label, exc_info=True)
            output.unresolved.append(label)
    return output
