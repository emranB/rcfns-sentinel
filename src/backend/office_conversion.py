'''Convert legacy office documents locally for text and layout extraction.'''

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from shutil import copyfile, which
from tempfile import TemporaryDirectory
from time import sleep

OFFICE_CONVERSIONS = {
    ".dotx": (".docx", "Office Open XML Text"),
    ".numbers": (".xlsx", "Calc MS Excel 2007 XML"),
    ".pages": (".docx", "Office Open XML Text"),
    ".rtf": (".docx", "Office Open XML Text"),
    ".vsdx": (".pdf", "draw_pdf_Export"),
    ".wpd": (".docx", "Office Open XML Text"),
    ".xltx": (".xlsx", "Calc MS Excel 2007 XML"),
}
OFFICE_EXTENSIONS = frozenset(OFFICE_CONVERSIONS)
CONVERSION_TIMEOUT_SECONDS = 120

PROFILE_SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
  <item oor:path="/org.openoffice.Office.Common/Security/Scripting">
    <prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop>
    <prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop>
    <prop oor:name="DisableActiveContent" oor:op="fuse"><value>true</value></prop>
    <prop oor:name="BlockUntrustedRefererLinks" oor:op="fuse"><value>true</value></prop>
  </item>
  <item oor:path="/org.openoffice.Office.Calc/Content/Update">
    <prop oor:name="Link" oor:op="fuse"><value>1</value></prop>
  </item>
</oor:items>
"""


class OfficeConversionError(RuntimeError):
    """An office document could not produce a usable local derivative."""


# Get the RAGFlow-supported output type for an input format.
def office_output_suffix(source: Path) -> str:
    conversion = OFFICE_CONVERSIONS.get(source.suffix.lower())
    if conversion is None:
        raise OfficeConversionError("This office file format cannot be converted.")
    return conversion[0]


# Locate the headless command without attaching to an open desktop session.
def office_executable() -> str:
    executable = os.getenv("LIBREOFFICE_EXECUTABLE")
    if executable:
        location = which(executable)
        if location:
            return location
        raise OfficeConversionError("LIBREOFFICE_EXECUTABLE is unavailable.")

    location = which("soffice.com") if sys.platform == "win32" else None
    location = location or which("soffice") or which("libreoffice")
    if location:
        return location
    if sys.platform == "win32":
        program_files = Path(os.getenv("ProgramFiles", "C:/Program Files"))
        installed = program_files / "LibreOffice" / "program" / "soffice.com"
        if installed.is_file():
            return str(installed)
    raise OfficeConversionError("LibreOffice is unavailable.")


# Terminate only the disposable converter and any child using its private profile.
def stop_conversion(process: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill.exe", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
                check=False,
            )
        if process.poll() is None:
            process.kill()
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


# Windows can briefly keep the converter's profile open after its process exits.
@contextmanager
def conversion_workspace(parent: Path) -> Iterator[Path]:
    directory = TemporaryDirectory(prefix=".office-", dir=parent)
    try:
        yield Path(directory.name)
    finally:
        for attempt in range(5):
            try:
                directory.cleanup()
                break
            except OSError:
                if sys.platform != "win32" or attempt == 4:
                    raise
                sleep(0.2)


# Convert in an isolated profile and publish the completed derivative atomically.
# LibreOffice can exit successfully without producing output on locked documents.
def convert_office(source: Path, output: Path) -> Path:
    suffix = office_output_suffix(source)
    if output.suffix.lower() != suffix:
        raise ValueError(f"The conversion output must use the {suffix} extension.")

    executable = office_executable()
    format_name = f"{suffix[1:]}:{OFFICE_CONVERSIONS[source.suffix.lower()][1]}"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with conversion_workspace(output.parent) as temporary:
        profile = temporary / "profile"
        settings = profile / "user" / "registrymodifications.xcu"
        settings.parent.mkdir(parents=True)
        settings.write_text(PROFILE_SETTINGS, encoding="utf-8")
        # Open a private input copy so the converter cannot lock the original.
        working = temporary / "input"
        working.mkdir()
        input_file = working / f"source{source.suffix.lower()}"
        copyfile(source, input_file)
        converted = temporary / f"source{suffix}"
        command = [
            executable,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--norestore",
            "--convert-to",
            format_name,
            "--outdir",
            str(temporary),
            str(input_file),
        ]
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW
        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=sys.platform != "win32",
                creationflags=creation_flags,
            ) as process:
                try:
                    exit_code = process.wait(timeout=CONVERSION_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    stop_conversion(process)
                    raise OfficeConversionError(
                        "LibreOffice conversion timed out."
                    ) from None
        except OSError:
            raise OfficeConversionError("Could not launch LibreOffice.") from None

        if exit_code or not converted.is_file() or not converted.stat().st_size:
            raise OfficeConversionError("LibreOffice did not produce a document.")
        converted.replace(output)
    return output
