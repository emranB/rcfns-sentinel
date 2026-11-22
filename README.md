# RCFNS Sentinel

## Run locally

1. Start Ollama on the host and run `ollama pull qwen3.5:4b`. The app pulls its embedding model on first startup.
2. From this directory, run `docker compose up --build -d` or `bash run.sh`.
3. Open [RCFNS Sentinel](http://localhost:9388/) to see the library progress. Chat becomes available as soon as a document is searchable. Ask a question, then select a source filename to open or download the complete indexed file.

Numbered citations in answers open a PDF preview at the cited region when it is available. The preview also links to the full original.

The composer defaults to **Standard**. Both modes start with the same RAGFlow search and use the local Ollama model to answer. **Agentic** can resolve references to earlier turns and run up to three follow-up searches to fill gaps in the evidence. It uses the standard results if those searches find no new excerpts or cannot complete.

Set `CHAT_MODEL` in `.env` to use a different model already installed in Ollama.

## Run on Ubuntu with an NVIDIA GPU

1. Place the source directory on the Ubuntu host and set `DATA_DIRECTORY` in its private `.env`.
2. Run the Ubuntu overlay to start Ollama and install the chat model.

   ```sh
   docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml up -d ollama
   docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml exec ollama ollama pull qwen3.5:4b
   docker compose -f docker-compose.yml -f docker-compose.ubuntu.yml up --build -d
   ```

3. Run `ssh -N -L 127.0.0.1:9388:127.0.0.1:9388 <ubuntu-host>` from the monitoring computer and open the local URL.

The Ubuntu overlay keeps the models, index, and services on the server. Set `RAGFLOW_CPU_LIMIT` and `RAGFLOW_MEMORY_LIMIT` in `.env` to fit the server.

## Develop the frontend

With the Docker stack running on port 9388, run `npm ci` and `npm run dev` in `src/frontend`. Vite proxies chat and status requests to the running app. Docker builds the React bundle automatically.

## Configure the source

Set `data_path` in `config.json` to a file, a directory to scan recursively, or a direct HTTP(S) download. The default is `data`. Directory paths are relative to the config unless they are absolute.

The importer handles PDF, Office documents, CSV, TXT, Markdown, HTML, email exports, and common images. Recovery also opens ZIP and RTFD members, converts legacy Office documents, and uses local vision for image recovery. Source files and model inference stay local.

The library's file inventory lists locked originals, shortcuts, audio and video recordings, empty files, and backup artifacts. Audio and video appear by original filename and source path. Files that still need recovery are reported separately.

To use a different host folder in Docker, set `DATA_DIRECTORY` in `.env` to the folder's absolute path. It is mounted read-only at `/app/data`. Leave `data_path` at `data` for the full folder, or set it to a subfolder or file such as `data/handbooks/policy.pdf`.

Run `docker compose up -d --force-recreate app` after changing the source or its files. The app keeps a file manifest in its named state volume and resumes the existing index for that source. It indexes new and changed files and retires entries for removed files. Keep Compose running until the status reports completion. Large collections and scanned PDFs can take substantially longer than the first searchable documents.

## Operate the service

- Check searchable, uploaded, skipped, and failed counts with `docker compose ps` or [the status endpoint](http://localhost:9388/status). View startup errors with `docker compose logs -f app ragflow`.
- For repeated bulk imports, allow disk room for the original files in object storage, search indexes, and PDF processing. Document uploads are limited to 256 MiB per file by this Compose setup.
- Stop the service with `docker compose down`. Named volumes retain the index and service account.
- If Elasticsearch reports a low map count on Docker Desktop, run `wsl -d docker-desktop -u root -- sysctl -w vm.max_map_count=262144`.
- The UI binds to loopback. For shared use, put an authenticated HTTPS proxy in front of it and verify that only the intended users can query the configured collection.
- For a fresh deployment, set separate `MYSQL_PASSWORD`, `ELASTIC_PASSWORD`, `MINIO_PASSWORD`, and `REDIS_PASSWORD` values in a private `.env`. Use shell-safe random hex values. Changing these settings on an existing deployment requires rotating the live service credentials as well.
- Back up the source files, configuration, and named volumes together. The index, recovery cache, and generated service account password must be restored together.
