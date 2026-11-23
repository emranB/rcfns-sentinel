# Architecture

`main.py` starts the FastAPI service. The backend handles HTTP routes in `server.py`, source resolution in `sources.py`, and RAGFlow integration in `ragflow.py`. Vite builds the React frontend in `src/frontend`, and FastAPI serves the production bundle.

On startup the app fingerprints the configured source and resumes the matching RAGFlow dataset. It uploads missing files in bounded batches, interleaves lightweight documents with PDF and image work, and exposes documents once they have searchable chunks. Unsupported files and terminal ingestion errors are counted separately.

Recovery indexes share the native dataset's embedding model and are searched alongside it. This preserves the existing native index while adding recovered document text. Converted items cite their exact local originals. Images and presentations that completed with zero chunks can gain local OCR or description text under their original document ID.

Both chat modes use the same evidence checks. Explicit titles and dates are resolved before broad retrieval, and Agentic follow-ups stay within the resolved set. The app reads complete worksheet data for supported counts and computes comparisons from aligned table rows. Ollama returns claims with supporting passages, and the app validates the references before it renders an answer. Citations point to the exact original. PDF previews load on demand.

For calculations, the model plans measures and operations before seeing figures. The app resolves each measure against the original report period, PDF column, or worksheet cell, then computes from the matched values. The optional `REASONING_MODEL` setting selects the local Ollama model for that planning step.

Compose keeps the databases, queue, object store, and RAGFlow API on its internal network. The app publishes one loopback port and uses named volumes to retain index and account state.
