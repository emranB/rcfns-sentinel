'''Check source switching and recursive discovery without a running index.'''

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from src.backend.sources import dataset_name, resolve_sources


class SourceTests(unittest.IsolatedAsyncioTestCase):
    # Verify a different directory can contain mixed formats and duplicate names.
    async def test_switch_source_without_reusing_old_files(self) -> None:
        async with httpx.AsyncClient() as client:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.json"
                first = root / "first.txt"
                first.write_text("first collection", encoding="utf-8")
                config.write_text(
                    json.dumps({"data_path": first.name}), encoding="utf-8"
                )
                original = await resolve_sources(client, config, root)
                self.assertEqual(list(original.files.values()), [first])

                alternate = root / "alternate"
                nested = alternate / "policies"
                nested.mkdir(parents=True)
                (nested / "Guide.TXT").write_text("rules", encoding="utf-8")
                (alternate / "guide.txt").write_text("index", encoding="utf-8")
                (alternate / "~$Office.docx").write_bytes(b"lock")
                (alternate / "asset.zip").write_bytes(b"archive")

                config.write_text(
                    json.dumps({"data_path": alternate.name}), encoding="utf-8"
                )
                changed = await resolve_sources(client, config, root)
                self.assertEqual(changed.skipped_count, 2)
                self.assertEqual(len(changed.files), 2)
                self.assertEqual(len({name.casefold() for name in changed.files}), 2)
                self.assertNotEqual(
                    dataset_name(original.files), dataset_name(changed.files)
                )

    # Verify direct downloads can get the original filename from a response header.
    async def test_download_without_extension_in_url(self) -> None:
        # Respond with an attachment name and body.
        def serve(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"source content",
                headers={"Content-Disposition": 'attachment; filename="notice.txt"'},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.json"
                config.write_text(
                    json.dumps({"data_path": "https://example.test/download?id=24"}),
                    encoding="utf-8",
                )
                result = await resolve_sources(client, config, root)
                self.assertEqual(
                    result.files["notice.txt"].read_bytes(), b"source content"
                )
