'''Exercise office output publication and failure handling.'''

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.backend.office_conversion import OfficeConversionError, convert_office


class OfficeConversionTests(TestCase):
    # Verify that the converter isolates profiles and never rewrites the source.
    def test_publish_derivative_and_preserve_original(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "meeting.rtf"
            original.write_bytes(b"original")
            output = root / "work" / "renamed.docx"

            # Emulate the headless process writing its native export name.
            def launch(command: list[str], **_options: object) -> MagicMock:
                work = Path(command[command.index("--outdir") + 1])
                profile = work / "profile"
                self.assertIn(f"-env:UserInstallation={profile.as_uri()}", command)
                settings = profile / "user" / "registrymodifications.xcu"
                self.assertIn("DisableMacrosExecution", settings.read_text("utf-8"))
                self.assertIn("docx:Office Open XML Text", command)
                input_copy = Path(command[-1])
                self.assertEqual(input_copy.read_bytes(), b"original")
                self.assertNotEqual(input_copy, original)
                (work / "source.docx").write_bytes(b"derivative")
                process = MagicMock()
                process.__enter__.return_value = process
                process.wait.return_value = 0
                return process

            with (
                patch(
                    "src.backend.office_conversion.office_executable",
                    return_value="soffice",
                ),
                patch(
                    "src.backend.office_conversion.subprocess.Popen", side_effect=launch
                ),
            ):
                self.assertEqual(convert_office(original, output), output.resolve())

            self.assertEqual(output.read_bytes(), b"derivative")
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual(list(output.parent.iterdir()), [output])

    # Verify that a successful exit alone cannot replace a valid derivative.
    def test_missing_export_preserves_previous_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "locked.rtf"
            original.write_bytes(b"original")
            output = root / "previous.docx"
            output.write_bytes(b"previous")
            process = MagicMock()
            process.__enter__.return_value = process
            process.wait.return_value = 0

            with (
                patch(
                    "src.backend.office_conversion.office_executable",
                    return_value="soffice",
                ),
                patch(
                    "src.backend.office_conversion.subprocess.Popen",
                    return_value=process,
                ),
                self.assertRaises(OfficeConversionError),
            ):
                convert_office(original, output)

            self.assertEqual(output.read_bytes(), b"previous")
