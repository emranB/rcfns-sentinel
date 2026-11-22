'''Persist the index versions for each configured source and recovery scope.'''

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.backend.sources import dataset_name


class CachedFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str
    name: str = Field(min_length=1)
    size: int = Field(ge=0)
    mtime_ns: int
    ctime_ns: int
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    remote_id: str = ""
    remote_name: str = ""

    # Match a local file version without reading its content again.
    def signature(self) -> tuple[int, int, int]:
        return self.size, self.mtime_ns, self.ctime_ns


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    files: dict[str, CachedFile] = Field(default_factory=dict)
    retired: dict[str, CachedFile] = Field(default_factory=dict)


@dataclass
class IndexCache:
    path: Path
    manifest: IndexManifest
    names: dict[str, str]

    # Commit a complete manifest before changing the corresponding remote documents.
    def save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        pending: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as output:
                pending = Path(output.name)
                output.write(self.manifest.model_dump_json())
                output.flush()
                os.fsync(output.fileno())
            pending.replace(self.path)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)


# Capture size and both timestamps for a native source without opening it.
def file_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


# Record a consistent byte digest, retaining remote identity for the same content.
def inspect_file(
    label: str,
    name: str,
    path: Path,
    previous: CachedFile | None,
    *,
    volatile: bool,
) -> CachedFile:
    signature = file_signature(path)
    if previous is not None and not volatile and previous.signature() == signature:
        return previous

    with path.open("rb") as document:
        digest = file_digest(document, "sha256").hexdigest()
    if signature != file_signature(path):
        raise RuntimeError(f"Source changed while preparing its index: {label}")

    size, mtime_ns, ctime_ns = signature
    if previous is not None and previous.digest == digest:
        return previous.model_copy(
            update={"size": size, "mtime_ns": mtime_ns, "ctime_ns": ctime_ns}
        )
    return CachedFile(
        label=label,
        name=name,
        size=size,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        digest=digest,
    )


# Name a replacement so upload retries cannot mistake old bytes for its revision.
def version_name(file: CachedFile, suffix: str, reserved: set[str]) -> str:
    stem = (
        PurePosixPath(file.label).stem.encode("utf-8")[:120].decode("utf-8", "ignore")
    )
    fingerprint = sha256(f"{file.label}\0{file.digest}".encode()).hexdigest()[:24]
    name = f"{stem} [{fingerprint}]{suffix}"
    number = 1
    while name.casefold() in reserved:
        number += 1
        name = f"{stem} [{fingerprint}-{number}]{suffix}"
    return name


# Reuse a legacy dataset only if a different configured source has not adopted it.
def initial_dataset_name(
    cache_path: Path,
    namespace: str,
    scope: str,
    files: dict[str, Path],
    digests: Mapping[str, bytes],
) -> str:
    suffix = f" {scope}" if scope else ""
    legacy = f"{dataset_name(files, digests=digests)}{suffix}"
    for candidate in cache_path.parent.glob("*.json"):
        if IndexManifest.model_validate_json(candidate.read_bytes()).dataset == legacy:
            return f"RCFNS {namespace[:20]}{suffix}"
    return legacy


# Keep the existing dataset while planning only new and changed source versions.
def prepare_index_cache(
    state_dir: Path,
    identity: str,
    scope: str,
    files: dict[str, Path],
    labels: Mapping[str, str],
    *,
    volatile: bool = False,
) -> IndexCache:
    key = sha256(f"{identity}\0{scope}".encode()).hexdigest()
    path = state_dir / "index-cache" / f"{key}.json"
    try:
        previous = IndexManifest.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        previous = None

    discovered = {name: labels.get(name, name) for name in files}
    if len(set(discovered.values())) != len(discovered):
        raise ValueError("The index scope contains duplicate source labels.")

    prior_files = previous.files if previous is not None else {}
    current = {
        label: inspect_file(
            label, name, files[name], prior_files.get(label), volatile=volatile
        )
        for name, label in discovered.items()
    }
    if previous is None:
        digests = {
            name: bytes.fromhex(current[label].digest)
            for name, label in discovered.items()
        }
        dataset = initial_dataset_name(path, key, scope, files, digests)
        return IndexCache(
            path,
            IndexManifest(dataset=dataset, files=current),
            {name: name for name in files},
        )

    retired = previous.retired.copy()
    for label, file in prior_files.items():
        if label not in current or file.digest != current[label].digest:
            retired[file.name] = file

    reserved = {
        name.casefold()
        for file in (*prior_files.values(), *retired.values())
        for name in (file.name, file.remote_name)
        if name
    }
    retired_versions = {(file.label, file.digest): file for file in retired.values()}
    for input_name, label in discovered.items():
        file = current[label]
        if label in prior_files and file.digest == prior_files[label].digest:
            continue
        if old := retired_versions.get((label, file.digest)):
            retired.pop(old.name)
            file.name = old.name
            file.remote_id = old.remote_id
            file.remote_name = old.remote_name
        else:
            file.name = version_name(file, Path(input_name).suffix, reserved)
        reserved.add(file.name.casefold())

    manifest = IndexManifest(dataset=previous.dataset, files=current, retired=retired)
    return IndexCache(
        path,
        manifest,
        {name: current[label].name for name, label in discovered.items()},
    )
