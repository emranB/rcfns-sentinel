'''Verify stable document versions and safe index migration across startups.'''

import os
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pydantic import ValidationError

from src.backend.index_cache import prepare_index_cache
from src.backend.sources import dataset_name, local_sources


class IndexCacheTests(TestCase):
    # Seed from the exact legacy scope identity and keep earlier duplicate names stable.
    def test_legacy_index_and_earlier_duplicate(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            data = root / "data"
            older = data / "z-last" / "Report.txt"
            older.parent.mkdir(parents=True)
            older.write_text("older", encoding="utf-8")
            original = local_sources(data).files
            labels = {
                name: path.relative_to(root).as_posix()
                for name, path in original.items()
            }
            legacy = dataset_name(original)
            with patch(
                "src.backend.index_cache.file_digest", wraps=file_digest
            ) as digest:
                saved = prepare_index_cache(root, "collection", "", original, labels)
                self.assertEqual(digest.call_count, 1)
            self.assertEqual(saved.manifest.dataset, legacy)
            self.assertEqual(saved.names, {"Report.txt": "Report.txt"})
            saved.manifest.files[labels["Report.txt"]].remote_id = "prior-id"
            saved.manifest.files[labels["Report.txt"]].remote_name = "Report(2).txt"
            saved.save()

            newer = data / "a-first" / "Report.txt"
            newer.parent.mkdir()
            newer.write_text("newer", encoding="utf-8")
            files = local_sources(data).files
            labels = {
                name: path.relative_to(root).as_posix() for name, path in files.items()
            }
            with patch(
                "src.backend.index_cache.file_digest", wraps=file_digest
            ) as digest:
                planned = prepare_index_cache(root, "collection", "", files, labels)
                self.assertEqual(digest.call_count, 1)
            self.assertEqual(planned.manifest.dataset, legacy)
            self.assertEqual(planned.names["Report (2).txt"], "Report.txt")
            self.assertNotEqual(planned.names["Report.txt"], "Report.txt")
            self.assertEqual(
                planned.manifest.files[labels["Report (2).txt"]].remote_id, "prior-id"
            )
            self.assertEqual(
                planned.manifest.files[labels["Report (2).txt"]].remote_name,
                "Report(2).txt",
            )
            planned.save()
            with patch(
                "src.backend.index_cache.file_digest",
                side_effect=AssertionError("read unchanged bytes"),
            ):
                restarted = prepare_index_cache(root, "collection", "", files, labels)
            self.assertEqual(restarted.names, planned.names)
            self.assertFalse(restarted.manifest.retired)

    # Journal replaced and removed versions until the corresponding generation is revived.
    def test_replacement_removal_and_revival(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            source = root / "report.txt"
            source.write_bytes(b"old")
            files = {source.name: source}
            labels = {source.name: "folder/report.txt"}
            legacy = prepare_index_cache(root, "collection", "recovery", files, labels)
            legacy.manifest.files["folder/report.txt"].remote_id = "legacy-id"
            legacy.save()

            stamp = source.stat().st_mtime_ns
            source.write_bytes(b"new")
            os.utime(source, ns=(stamp, stamp + 1_000_000_000))
            replacement = prepare_index_cache(
                root, "collection", "recovery", files, labels
            )
            replacement_name = replacement.names["report.txt"]
            self.assertNotEqual(replacement_name, "report.txt")
            self.assertEqual(replacement.manifest.dataset, legacy.manifest.dataset)
            self.assertEqual(
                replacement.manifest.retired["report.txt"].remote_id, "legacy-id"
            )
            self.assertEqual(
                replacement.manifest.files["folder/report.txt"].remote_id, ""
            )
            replacement.save()

            removed = prepare_index_cache(root, "collection", "recovery", {}, {})
            self.assertFalse(removed.names)
            self.assertEqual(
                set(removed.manifest.retired), {"report.txt", replacement_name}
            )
            removed.save()

            source.write_bytes(b"old")
            revived = prepare_index_cache(root, "collection", "recovery", files, labels)
            self.assertEqual(revived.names, {"report.txt": "report.txt"})
            self.assertEqual(
                revived.manifest.files["folder/report.txt"].remote_id, "legacy-id"
            )
            self.assertEqual(set(revived.manifest.retired), {replacement_name})

    # Keep identically seeded source namespaces separate while matching volatile files by bytes.
    def test_independent_namespace_and_volatile_staging(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            first = root / "first-stage.txt"
            second = root / "new-stage.txt"
            for path in (first, second):
                path.write_bytes(b"archive content")
            name = "archive-member.txt"
            labels = {name: "archive.zip!/member.txt"}
            saved = prepare_index_cache(
                root, "source one", "media", {name: first}, labels
            )
            saved.save()
            restart = prepare_index_cache(
                root, "source one", "media", {name: second}, labels, volatile=True
            )
            self.assertEqual(restart.names, {name: name})
            self.assertFalse(restart.manifest.retired)
            isolated = prepare_index_cache(
                root, "source two", "media", {name: first}, labels
            )
            self.assertNotEqual(isolated.manifest.dataset, saved.manifest.dataset)
            self.assertTrue(isolated.manifest.dataset.endswith(" media"))

    # Surface corruption instead of silently creating a new mutable dataset.
    def test_invalid_manifest_fails_before_inspecting_sources(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            source = root / "report.txt"
            source.write_bytes(b"file")
            files = {source.name: source}
            planned = prepare_index_cache(root, "collection", "", files, {})
            planned.save()
            planned.path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValidationError):
                prepare_index_cache(root, "collection", "", files, {})
