'''Resolve configured sources and identify their contents without loading them into memory.'''

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
from hashlib import file_digest, sha256
from os import walk
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import BaseModel, Field

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".csv",
        ".doc",
        ".docx",
        ".eml",
        ".gif",
        ".htm",
        ".html",
        ".jpeg",
        ".jpg",
        ".md",
        ".msg",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".tif",
        ".tiff",
        ".txt",
        ".webp",
        ".xls",
        ".xlsx",
    }
)


class SourceSettings(BaseModel):
    data_path: str = Field(min_length=1, strict=True)


@dataclass
class SourceCollection:
    files: dict[str, Path]
    excluded: tuple[Path, ...] = ()
    identity: str = ""
    volatile: bool = False

    # Count the originals that still need classification or a recovery parser.
    @property
    def skipped_count(self) -> int:
        return len(self.excluded)


# Resolve a single file, a directory, or a direct download.
async def resolve_sources(
    client: httpx.AsyncClient, config_path: Path, download_dir: Path
) -> SourceCollection:
    source = SourceSettings.model_validate_json(
        config_path.read_bytes()
    ).data_path.strip()
    if not source:
        raise ValueError("Set data_path to a file, directory, or direct download URL.")

    if urlsplit(source).scheme in {"http", "https"}:
        try:
            path = await download_source(client, source, download_dir)
        except httpx.HTTPError:
            # HTTPX error messages can contain private download tokens.
            raise RuntimeError("Could not download the configured source.") from None
        identity = sha256(source.encode("utf-8")).hexdigest()
        return SourceCollection({path.name: path}, identity=identity, volatile=True)

    return await asyncio.to_thread(local_sources, config_path.parent / source)


# Recursively discover supported files and assign stable names for RAGFlow.
def local_sources(path: Path) -> SourceCollection:
    files = (
        sorted(Path(folder) / name for folder, _, names in walk(path) for name in names)
        if path.is_dir()
        else [path]
    )
    paths = [
        file
        for file in files
        if not file.name.startswith("~$")
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
        and file.stat().st_size
    ]
    if not files:
        raise ValueError("The configured directory contains no files.")
    if not path.exists():
        raise ValueError("The configured path does not exist.")

    reserved = {file.name.casefold() for file in paths}
    used: set[str] = set()
    next_number: dict[str, int] = {}
    sources: dict[str, Path] = {}
    for file in paths:
        name = file.name
        key = name.casefold()
        if key in used:
            number = next_number.get(key, 2)
            name = f"{file.stem} ({number}){file.suffix}"
            while name.casefold() in reserved or name.casefold() in used:
                number += 1
                name = f"{file.stem} ({number}){file.suffix}"
            next_number[key] = number + 1
        used.add(name.casefold())
        sources[name] = file
    selected = set(paths)
    return SourceCollection(
        sources,
        tuple(file for file in files if file not in selected),
        str(path.resolve()),
    )


# Download a supported file to temporary storage and close the network stream.
async def download_source(client: httpx.AsyncClient, url: str, directory: Path) -> Path:
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        headers = Message()
        headers["Content-Disposition"] = response.headers.get("content-disposition", "")
        names = (
            headers.get_filename() or "",
            unquote(response.url.path),
            unquote(urlsplit(url).path),
        )
        candidates = (Path(name.replace("\\", "/")).name for name in names)
        name = next(
            (
                item
                for item in candidates
                if Path(item).suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            "",
        )
        content_type = (
            response.headers.get("content-type", "").partition(";")[0].strip()
        )
        if content_type in {"text/html", "application/xhtml+xml"} and Path(
            name
        ).suffix.lower() not in {".htm", ".html"}:
            raise ValueError(
                "The source URL returned a web page. Use a direct download URL."
            )
        if not name:
            raise ValueError("Use a direct download URL for a supported file.")
        path = directory / name
        with path.open("wb") as output:
            async for chunk in response.aiter_bytes():
                output.write(chunk)
    return path


# Keep the dataset name compatible with existing indexes for unchanged sources.
def dataset_name(
    sources: dict[str, Path], *, digests: Mapping[str, bytes] | None = None
) -> str:
    digest = sha256()
    for name, path in sources.items():
        digest.update(name.encode("utf-8"))
        if digests is None:
            with path.open("rb") as document:
                digest.update(file_digest(document, "sha256").digest())
        else:
            digest.update(digests[name])
    return f"RCFNS {digest.hexdigest()[:20]}"
