'''Manage the RAGFlow session, embedding provider, and document index.'''

from __future__ import annotations

import asyncio
import logging
import os
from base64 import b64encode
from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from hashlib import file_digest, sha256
from pathlib import Path
from secrets import token_urlsafe
from time import monotonic
from urllib.parse import quote

import httpx
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from pydantic import BaseModel, TypeAdapter

from src.backend.http_retry import retry_http
from src.backend.index_cache import IndexCache
from src.backend.sources import dataset_name

RAGFLOW_URL = os.getenv("RAGFLOW_URL", "http://localhost:9380/api/v1").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_THREADS = int(os.getenv("OLLAMA_THREADS", "0"))
OLLAMA_SECONDARY_VISION_URL = os.getenv("OLLAMA_SECONDARY_VISION_URL", "").rstrip("/")
OLLAMA_SECONDARY_VISION_THREADS = int(os.getenv("OLLAMA_SECONDARY_VISION_THREADS", "2"))
STATE_DIR = Path(os.getenv("APP_STATE", "/state"))

CHAT_MODEL = os.getenv("CHAT_MODEL") or "qwen3.5:4b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
PROVIDER_NAME = "Ollama"
INSTANCE_NAME = "local"
PROVIDER_INSTANCES = f"/providers/{PROVIDER_NAME}/instances"
EMBEDDING_REFERENCE = f"{EMBEDDING_MODEL}@{INSTANCE_NAME}@{PROVIDER_NAME}"

LOGGER = logging.getLogger("uvicorn.error")
FAILED_STATES = {
    "2",
    "4",
    "CANCEL",
    "CANCELED",
    "CANCELLED",
    "FAIL",
    "FAILED",
    "STOPPED",
}
RETRYABLE_STATES = FAILED_STATES | {
    "",
    "0",
    "UNSTART",
    "UNSTARTED",
    "NOT_STARTED",
    "NOT_INGESTED",
    "PENDING",
}
COMPLETED_STATES = {"3", "DONE", "COMPLETED"}
PAGE_SIZE = 100
POLL_INTERVAL_SECONDS = 30
READ_RETRY_DELAYS = (2, 6, 15, 30)
STALL_TIMEOUT_SECONDS = 3600
STARTUP_ATTEMPTS = 150
BATCH_FILES = 32
BATCH_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_UPLOAD_SIZE = int(os.getenv("MAX_DOCUMENT_UPLOAD_SIZE", "268435456"))
UPLOAD_WORKERS = 2
LAYOUT_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff", ".webp"}
)

UploadFiles = Sequence[tuple[str, tuple[str, bytes]]]
RequestParams = Mapping[str, str | int]


class ApiResult(BaseModel):
    code: int
    message: str | None = None
    data: object = None


class NamedDataset(BaseModel):
    id: str
    name: str


class ProviderInstance(BaseModel):
    instance_name: str


class OllamaModel(BaseModel):
    name: str


class OllamaTags(BaseModel):
    models: list[OllamaModel]


class Document(NamedDataset):
    ingestion_status: str | None = None
    run: str | int | None = None
    chunk_count: int | None = None
    progress: float | None = None
    size: int | None = None

    # Normalize the status across list and upload response formats.
    @property
    def phase(self) -> str:
        return str(self.ingestion_status or self.run or "").upper()


class DocumentPage(BaseModel):
    docs: list[Document]
    total: int


DATASETS = TypeAdapter(list[NamedDataset])
PROVIDER_LIST = TypeAdapter(list[ProviderInstance])
DOCUMENTS = TypeAdapter(list[Document])


class RagflowError(RuntimeError):
    # Preserve the RAGFlow application code for handling lookup misses.
    def __init__(self, code: int, message: str | None):
        super().__init__(message or "RAGFlow request failed.")
        self.code = code


@dataclass
class IndexProgress:
    status: str = "Starting"
    total_count: int = 0
    indexed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    uploaded_count: int = 0
    complete: bool = False


@dataclass
class ImportState:
    progress: IndexProgress
    files: dict[str, Path]
    documents: dict[str, Document] = field(default_factory=dict)
    rejected: set[str] = field(default_factory=set)
    queued: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    inventory: AbstractSet[str] = frozenset()
    supplemental: str = ""


@dataclass
class SearchIndex:
    dataset_id: str
    documents: dict[str, str] = field(default_factory=dict)


@dataclass
class RagflowSession:
    client: httpx.AsyncClient
    authorization: str = field(default="", repr=False)
    auth_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    dataset_id: str = ""
    documents: dict[str, str] = field(default_factory=dict)
    source_paths: dict[str, str] = field(default_factory=dict)
    extras: dict[str, SearchIndex] = field(default_factory=dict)
    without_chunks: dict[str, str] = field(default_factory=dict)
    original_files: dict[str, Path] = field(default_factory=dict)
    last_ingestion_activity: float = field(default_factory=monotonic)

    # Locate a cited original in either active dataset.
    def locate_document(self, document_id: str) -> tuple[str, str] | None:
        if document_id in self.documents:
            return self.dataset_id, self.documents[document_id]
        for index in self.extras.values():
            if document_id in index.documents:
                return index.dataset_id, index.documents[document_id]
        return None


# Read RAGFlow's application-level result as well as the HTTP status.
def response_data(response: httpx.Response) -> object:
    response.raise_for_status()
    body = ApiResult.model_validate_json(response.content)
    if body.code != 0:
        raise RagflowError(body.code, body.message)
    return body.data


# Load or create the service password without exposing it to other filesystem users.
def service_password() -> bytes:
    path = STATE_DIR / "password"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as password_file:
            password_file.write(token_urlsafe(24))
    path.chmod(0o600)
    return path.read_bytes()


# Wait for RAGFlow startup, then sign in or register the local service account.
async def authenticate(state: RagflowSession) -> None:
    public_key = Path(__file__).with_name("ragflow-public.pem").read_bytes()
    key = load_pem_public_key(public_key)
    if not isinstance(key, RSAPublicKey):
        raise TypeError("The RAGFlow public key must be RSA.")
    encrypted_password = b64encode(key.encrypt(service_password(), padding.PKCS1v15()))
    credentials = {
        "email": "chatbot@example.com",
        "password": encrypted_password.decode(),
    }

    for attempt in range(STARTUP_ATTEMPTS):
        try:
            response = await state.client.post(
                f"{RAGFLOW_URL}/auth/login", json=credentials
            )
            if not response.headers.get("Authorization"):
                response = await state.client.post(
                    f"{RAGFLOW_URL}/users", json={**credentials, "nickname": "RCFNS"}
                )
            response_data(response)
            state.authorization = response.headers["Authorization"]
            return
        except (httpx.TransportError, httpx.HTTPStatusError, KeyError, RuntimeError):
            if attempt == STARTUP_ATTEMPTS - 1:
                raise
            await asyncio.sleep(2)


# Send a request and serialize refreshes when the upstream session expires.
async def authenticated_response(
    state: RagflowSession,
    method: str,
    path: str,
    *,
    stream: bool = False,
    params: RequestParams | None = None,
    json: object = None,
    files: UploadFiles | None = None,
) -> httpx.Response:
    token = state.authorization
    url = RAGFLOW_URL + path

    # Rebuild the request so a retry carries the refreshed token.
    async def send(authorization: str) -> httpx.Response:
        upstream = state.client.build_request(
            method,
            url,
            headers={"Authorization": authorization},
            params=params,
            json=json,
            files=files,
            timeout=300 if files else state.client.timeout,
        )
        return await state.client.send(upstream, stream=stream)

    response = await send(token)
    if response.status_code == 401:
        await response.aclose()
        async with state.auth_lock:
            if state.authorization == token:
                await authenticate(state)
        response = await send(state.authorization)
    return response


# Read the JSON payload from an authenticated RAGFlow API request.
async def request(
    state: RagflowSession,
    method: str,
    path: str,
    *,
    params: RequestParams | None = None,
    json: object = None,
    files: UploadFiles | None = None,
) -> object:
    # Validate a single request before deciding whether it is safe to repeat.
    async def send() -> object:
        return response_data(
            await authenticated_response(
                state, method, path, params=params, json=json, files=files
            )
        )

    if method == "GET":
        return await retry_http(send, READ_RETRY_DELAYS)
    return await send()


# Verify the chat model and provision the embedding provider for a cold start.
async def prepare_models(state: RagflowSession) -> None:
    response = await state.client.get(f"{OLLAMA_URL}/api/tags")
    response.raise_for_status()
    tags = OllamaTags.model_validate_json(response.content)
    model_names = {model.name for model in tags.models}
    if CHAT_MODEL not in model_names:
        raise RuntimeError(
            f"Install the Ollama chat model with ollama pull {CHAT_MODEL}."
        )
    if EMBEDDING_MODEL not in model_names:
        response = await state.client.post(
            f"{OLLAMA_URL}/api/pull",
            json={"model": EMBEDDING_MODEL, "stream": False},
            timeout=900,
        )
        response.raise_for_status()

    await request(state, "PUT", "/providers", json={"provider_name": PROVIDER_NAME})
    instances = PROVIDER_LIST.validate_python(
        await request(state, "GET", PROVIDER_INSTANCES) or []
    )
    if not any(instance.instance_name == INSTANCE_NAME for instance in instances):
        await request(
            state,
            "POST",
            PROVIDER_INSTANCES,
            json={
                "instance_name": INSTANCE_NAME,
                "base_url": OLLAMA_URL,
                "api_key": "",
                "model_info": [
                    {
                        "model_name": EMBEDDING_MODEL,
                        "model_type": ["embedding"],
                        "max_tokens": 8192,
                    }
                ],
            },
        )


# Find the existing index by content fingerprint or create an empty one.
async def ensure_dataset(state: RagflowSession, name: str) -> str:
    datasets = DATASETS.validate_python(
        await request(state, "GET", "/datasets", params={"name": name}) or []
    )
    for dataset in datasets:
        if dataset.name == name:
            return dataset.id

    dataset = NamedDataset.model_validate(
        await request(
            state,
            "POST",
            "/datasets",
            json={
                "name": name,
                "embedding_model": EMBEDDING_REFERENCE,
                "parse_type": 1,
                "parser_id": "general",
            },
        )
    )
    return dataset.id


# Read the current dataset state using the API's maximum supported page size.
async def list_documents(state: RagflowSession, path: str) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    page = 1
    while True:
        result = DocumentPage.model_validate(
            await request(
                state, "GET", path, params={"page": page, "page_size": PAGE_SIZE}
            )
        )
        documents.update((document.name, document) for document in result.docs)
        if not result.docs or page * PAGE_SIZE >= result.total:
            return documents
        page += 1


# Read a source digest only for a potential filename collision.
def source_digest(path: Path) -> tuple[int, bytes]:
    with path.open("rb") as source:
        return path.stat().st_size, file_digest(source, "sha256").digest()


# Verify the stored original by streaming it without relying on filename collation.
async def stored_digest(state: RagflowSession, path: str, document_id: str) -> bytes:
    response = await authenticated_response(
        state, "GET", f"{path}/{quote(document_id, safe='')}", stream=True
    )
    digest = sha256()
    try:
        response.raise_for_status()
        async for block in response.aiter_bytes():
            digest.update(block)
    finally:
        await response.aclose()
    return digest.digest()


# Match upstream collision suffixes only if the stored original is byte-identical.
async def match_documents(
    state: RagflowSession,
    path: str,
    files: dict[str, Path],
    documents: dict[str, Document],
    known_aliases: Mapping[str, str] | None = None,
) -> dict[str, Document]:
    matched = {name: documents[name] for name in files.keys() & documents.keys()}
    known = known_aliases or {}
    for name, alias in known.items():
        if name in files and name not in matched and alias in documents:
            matched[name] = documents[alias]

    claimed = {document.name for document in matched.values()}
    alternatives: dict[str, list[tuple[int, Document]]] = {}
    for alias in documents.keys() - files.keys() - claimed:
        basename = Path(alias)
        stem, separator, number = basename.stem.removesuffix(")").rpartition("(")
        if (
            separator
            and basename.stem.endswith(")")
            and number.isascii()
            and number.isdecimal()
            and len(number) < 9
        ):
            alternatives.setdefault(stem + basename.suffix, []).append(
                (int(number), documents[alias])
            )
    for name in (files.keys() - matched.keys()) & alternatives.keys():
        size, expected = await asyncio.to_thread(source_digest, files[name])
        for _, candidate in sorted(
            alternatives[name], key=lambda item: (item[0], item[1].id)
        ):
            if (
                candidate.size == size
                and await stored_digest(state, path, candidate.id) == expected
            ):
                matched[name] = candidate
                LOGGER.info("Reconciled upstream filename alias for %s", name)
                break
    return matched


# Refresh original-name mappings after polling the upstream dataset.
async def refresh_documents(
    state: RagflowSession,
    path: str,
    sync: ImportState,
    documents: dict[str, Document] | None = None,
) -> None:
    if documents is None:
        documents = await list_documents(state, path)
    sync.documents = await match_documents(
        state, path, sync.files, documents, sync.aliases
    )
    sync.aliases.update(
        (name, document.name)
        for name, document in sync.documents.items()
        if name != document.name
    )


# Retire only provider documents verified to belong to a removed or changed original.
async def retire_cached_documents(
    state: RagflowSession,
    path: str,
    documents: dict[str, Document],
    cache: IndexCache,
) -> dict[str, Document]:
    if not cache.manifest.retired:
        return documents
    by_id = {document.id: document for document in documents.values()}
    current_ids = {item.remote_id for item in cache.manifest.files.values()}
    retired: list[str] = []
    for item in cache.manifest.retired.values():
        candidate = (
            by_id.get(item.remote_id) if item.remote_id else documents.get(item.name)
        )
        if candidate is None or candidate.id in current_ids:
            continue
        if not item.remote_id and (
            await stored_digest(state, path, candidate.id) != bytes.fromhex(item.digest)
        ):
            continue
        retired.append(candidate.id)

    for start in range(0, len(retired), PAGE_SIZE):
        await request(
            state,
            "DELETE",
            path,
            json={"ids": retired[start : start + PAGE_SIZE]},
        )
    if retired:
        documents = await list_documents(state, path)
        remaining = {document.id for document in documents.values()}
        if remaining.intersection(retired):
            raise RuntimeError("RAGFlow did not retire all superseded index entries.")
    cache.manifest.retired.clear()
    return documents


# Save exact provider identities after reconciling an upload or an existing alias.
def record_cached_documents(cache: IndexCache, documents: dict[str, Document]) -> bool:
    changed = False
    for item in cache.manifest.files.values():
        document = documents.get(item.name)
        if document and (item.remote_id, item.remote_name) != (
            document.id,
            document.name,
        ):
            item.remote_id = document.id
            item.remote_name = document.name
            changed = True
    return changed


# Use exact name lookup to reconcile a multipart write before retrying it.
async def find_document(state: RagflowSession, path: str, name: str) -> Document | None:
    try:
        response = await request(state, "GET", path, params={"name": name})
    except RagflowError as error:
        if error.code != 102 or "don't own the document" not in str(error):
            raise
        return None
    result = DocumentPage.model_validate(response)
    return next((document for document in result.docs if document.name == name), None)


# Interleave three Office or text files with one layout-heavy document.
def prioritized_files(
    files: dict[str, Path],
) -> Iterator[tuple[int, str, Path]]:
    basic: list[tuple[int, str, Path]] = []
    layout: list[tuple[int, str, Path]] = []
    for entry in sorted(
        (path.stat().st_size, name, path) for name, path in files.items()
    ):
        target = layout if entry[2].suffix.lower() in LAYOUT_EXTENSIONS else basic
        target.append(entry)
    for index, entry in enumerate(layout):
        yield from basic[index * 3 : index * 3 + 3]
        yield entry
    yield from basic[len(layout) * 3 :]


# Bound each multipart request by total bytes and file count.
def upload_batches(files: dict[str, Path]) -> Iterator[dict[str, Path]]:
    batch: dict[str, Path] = {}
    size = 0
    for file_size, name, path in prioritized_files(files):
        if batch and (len(batch) == BATCH_FILES or size + file_size > BATCH_BYTES):
            yield batch
            batch = {}
            size = 0
        batch[name] = path
        size += file_size
    if batch:
        yield batch


# Guard the memory budget even if a source changes after batch discovery.
def read_upload_files(files: dict[str, Path]) -> list[tuple[str, tuple[str, bytes]]]:
    uploads: list[tuple[str, tuple[str, bytes]]] = []
    total_size = 0
    for name, file in files.items():
        size = file.stat().st_size
        if size > MAX_DOCUMENT_UPLOAD_SIZE:
            raise ValueError(
                f"{name} exceeds the upload limit of {MAX_DOCUMENT_UPLOAD_SIZE} bytes."
            )
        total_size += size
        if len(files) > 1 and total_size > BATCH_BYTES:
            raise ValueError("Source file sizes changed while preparing a batch.")
        with file.open("rb") as source:
            data = source.read(size + 1)
        if len(data) != size:
            raise ValueError(f"{name} changed while being read.")
        uploads.append(("file", (name, data)))
    return uploads


# Read the bounded batch off the event loop before streaming it to RAGFlow.
async def upload_documents(
    state: RagflowSession, path: str, files: dict[str, Path]
) -> dict[str, Document]:
    payload = await asyncio.to_thread(read_upload_files, files)
    response = await request(state, "POST", path, files=payload)
    documents = {
        document.name: document for document in DOCUMENTS.validate_python(response)
    }
    return await match_documents(state, path, files, documents)


# Salvage a partially accepted batch, then retry only the missing uploads.
async def upload_batch(
    state: RagflowSession, path: str, files: dict[str, Path]
) -> dict[str, Document]:
    try:
        uploaded = await upload_documents(state, path, files)
    except (OSError, httpx.HTTPError, RagflowError, ValueError) as error:
        LOGGER.warning(
            "Upload batch of %d files needs reconciliation: %s", len(files), error
        )
        uploaded = {}

    for name in files.keys() - uploaded.keys():
        try:
            existing = await find_document(state, path, name)
            if existing:
                uploaded[name] = existing
            else:
                uploaded.update(
                    await upload_documents(state, path, {name: files[name]})
                )
        except (OSError, httpx.HTTPError, RagflowError, ValueError) as error:
            LOGGER.warning("Could not upload %s: %s", name, error)
    return uploaded


# Submit documents in batches while leaving already scheduled work untouched.
async def queue_documents(
    state: RagflowSession, path: str, sync: ImportState, candidates: Sequence[Document]
) -> None:
    pending = [
        document
        for document in candidates
        if document.phase in RETRYABLE_STATES and document.id not in sync.queued
    ]
    for start in range(0, len(pending), PAGE_SIZE):
        batch = pending[start : start + PAGE_SIZE]
        try:
            await request(
                state,
                "POST",
                f"{path}/chunks",
                json={"document_ids": [doc.id for doc in batch]},
            )
        except (httpx.HTTPError, RagflowError, ValueError) as error:
            LOGGER.warning(
                "Parse batch of %d files needs reconciliation: %s", len(batch), error
            )
            for document in batch:
                try:
                    current = await find_document(
                        state, f"{path}/documents", document.name
                    )
                    if current is None or current.phase in RETRYABLE_STATES:
                        await request(
                            state,
                            "POST",
                            f"{path}/chunks",
                            json={"document_ids": [document.id]},
                        )
                except (httpx.HTTPError, RagflowError, ValueError) as error:
                    sync.rejected.add(document.name)
                    LOGGER.warning("Could not schedule %s: %s", document.name, error)
        sync.queued.update(document.id for document in batch)


# Publish only documents with usable chunks and count terminal failures separately.
def report_index(state: RagflowSession, sync: ImportState) -> None:
    indexed: dict[str, str] = {}
    failures = sync.rejected - sync.inventory
    empty: dict[str, str] = {}
    for name, document in sync.documents.items():
        if name not in sync.files or name in sync.inventory:
            continue
        if document.phase in COMPLETED_STATES and (document.chunk_count or 0) > 0:
            indexed[document.id] = name
            failures.discard(name)
        elif document.phase in FAILED_STATES or document.phase in COMPLETED_STATES:
            failures.add(name)
            if document.phase in COMPLETED_STATES:
                empty[document.id] = name

    if sync.supplemental:
        state.extras[sync.supplemental].documents = indexed
    else:
        state.documents = indexed
        state.without_chunks = empty
    sync.progress.total_count = len(sync.files) - len(sync.inventory)
    sync.progress.indexed_count = len(indexed)
    sync.progress.failed_count = len(failures)
    sync.progress.uploaded_count = len(
        (sync.documents.keys() & sync.files.keys()) - sync.inventory
    )


# Resume the current corpus and make completed documents searchable while parsing continues.
async def index_sources(
    state: RagflowSession,
    sources: dict[str, Path],
    progress: IndexProgress,
    *,
    inventory: AbstractSet[str] = frozenset(),
    supplemental: str = "",
    cache: IndexCache | None = None,
) -> None:
    progress.total_count = len(sources) - len(inventory)
    if not progress.total_count and (cache is None or not cache.manifest.retired):
        progress.complete = True
        return
    if cache is None:
        index_name = await asyncio.to_thread(dataset_name, sources)
        if supplemental:
            index_name = f"{index_name} {supplemental}"
    else:
        await asyncio.to_thread(cache.save)
        index_name = cache.manifest.dataset
    dataset_id = await ensure_dataset(state, index_name)
    if supplemental:
        state.extras[supplemental] = SearchIndex(dataset_id)
    else:
        state.dataset_id = dataset_id
    dataset_path = f"/datasets/{dataset_id}"
    documents_path = f"{dataset_path}/documents"
    sync = ImportState(
        progress, sources, inventory=inventory, supplemental=supplemental
    )
    if cache is None:
        await refresh_documents(state, documents_path, sync)
    else:
        documents = await list_documents(state, documents_path)
        had_retirements = bool(cache.manifest.retired)
        documents = await retire_cached_documents(
            state, documents_path, documents, cache
        )
        sync.aliases = {
            item.name: item.remote_name
            for item in cache.manifest.files.values()
            if item.remote_name in documents
            and documents[item.remote_name].id == item.remote_id
        }
        await refresh_documents(state, documents_path, sync, documents)
        recorded = record_cached_documents(cache, sync.documents)
        if had_retirements or recorded:
            await asyncio.to_thread(cache.save)
    report_index(state, sync)
    await queue_documents(
        state,
        dataset_path,
        sync,
        [
            document
            for name, document in sync.documents.items()
            if name in sources and name not in inventory
        ],
    )

    next_poll = monotonic() + POLL_INTERVAL_SECONDS
    missing = {
        name: path
        for name, path in sources.items()
        if name not in sync.documents and name not in inventory
    }
    batches = await asyncio.to_thread(lambda: list(upload_batches(missing)))
    for start in range(0, len(batches), UPLOAD_WORKERS):
        group = batches[start : start + UPLOAD_WORKERS]
        uploads = await asyncio.gather(
            *(upload_batch(state, documents_path, batch) for batch in group)
        )
        new_documents: list[Document] = []
        for batch, uploaded in zip(group, uploads, strict=True):
            sync.documents.update(uploaded)
            sync.aliases.update(
                (name, document.name)
                for name, document in uploaded.items()
                if name != document.name
            )
            sync.rejected.update(batch.keys() - uploaded.keys())
            new_documents.extend(uploaded.values())
        if new_documents:
            state.last_ingestion_activity = monotonic()
        progress.uploaded_count = len(
            (sync.documents.keys() & sources.keys()) - inventory
        )
        await queue_documents(state, dataset_path, sync, new_documents)
        if monotonic() >= next_poll:
            await refresh_documents(state, documents_path, sync)
            report_index(state, sync)
            next_poll = monotonic() + POLL_INTERVAL_SECONDS

    if batches and cache is not None and record_cached_documents(cache, sync.documents):
        await asyncio.to_thread(cache.save)
    previous: dict[str, tuple[str, float | None, int | None]] = {}
    while True:
        await refresh_documents(state, documents_path, sync)
        report_index(state, sync)
        current = {
            name: (document.phase, document.progress, document.chunk_count)
            for name, document in sync.documents.items()
            if name in sources
        }
        now = monotonic()
        # All datasets wait on the same worker queue.
        if current != previous:
            previous = current
            state.last_ingestion_activity = now
        if progress.indexed_count + progress.failed_count >= progress.total_count:
            progress.complete = True
            return
        if now - state.last_ingestion_activity >= STALL_TIMEOUT_SECONDS:
            raise RuntimeError("RAGFlow ingestion made no progress for one hour.")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)
