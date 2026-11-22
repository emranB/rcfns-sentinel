# Architecture

`main.py` starts the FastAPI service. The backend handles HTTP routes in `server.py`, source resolution in `sources.py`, and RAGFlow integration in `ragflow.py`. Vite builds the React frontend in `src/frontend`, and FastAPI serves the production bundle.

On startup the app fingerprints the configured source and resumes the matching RAGFlow dataset. It uploads missing files in bounded batches, interleaves lightweight documents with PDF and image work, and exposes documents once they have searchable chunks. Unsupported files and terminal ingestion errors are counted separately.

Recovery indexes share the native dataset's embedding model and are searched alongside it. This preserves the existing native index while adding recovered document text. Converted items cite their exact local originals. Images and presentations that completed with zero chunks can gain local OCR or description text under their original document ID.

Both chat modes retrieve from the active datasets and generate answers locally with Ollama. Agentic can resolve follow-ups and look for additional documents while retaining the standard evidence. Citations point to the exact original. The frontend uses strict TypeScript, and PDF previews load on demand.

Compose keeps the databases, queue, object store, and RAGFlow API on its internal network. The app publishes one loopback port and uses named volumes to retain index and account state.
