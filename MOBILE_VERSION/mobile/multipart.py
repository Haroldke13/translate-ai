"""Read an uploaded file out of a multipart/form-data request body.

The standard library used to do this with cgi.FieldStorage, but that module was
removed in Python 3.13, and nothing replaced it. Since the whole point of this
build is that it starts on a bare "pkg install python", pulling in a web
framework to parse one upload is not an option, so it is done here.

Uploads are streamed to a temporary file rather than held in memory, because a
recording of a sermon can be hundreds of megabytes and a phone cannot spare
that.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Anything larger is refused before it fills the phone's storage.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
CHUNK = 64 * 1024
NAME_RE = re.compile(r'name="([^"]*)"')
FILENAME_RE = re.compile(r'filename="([^"]*)"')


class UploadTooLarge(Exception):
    pass


class MalformedUpload(Exception):
    pass


@dataclass
class Upload:
    filename: str
    path: Path
    size: int


def boundary_from(content_type: str) -> bytes:
    """The delimiter the client picked, from the Content-Type header."""
    for piece in content_type.split(";"):
        piece = piece.strip()
        if piece.casefold().startswith("boundary="):
            value = piece.split("=", 1)[1].strip().strip('"')
            if value:
                return value.encode("latin-1")
    raise MalformedUpload("the upload did not name a boundary")


def _buffer_body(stream, length: int, max_bytes: int, work_dir: Path):
    """Copy the request body to disk, refusing anything oversized."""
    if length > max_bytes:
        raise UploadTooLarge(f"{length} bytes is over the {max_bytes} byte limit")
    handle = tempfile.NamedTemporaryFile(dir=work_dir, suffix=".body", delete=False)
    remaining = length
    try:
        while remaining > 0:
            block = stream.read(min(CHUNK, remaining))
            if not block:
                break
            handle.write(block)
            remaining -= len(block)
    finally:
        handle.flush()
        handle.seek(0)
    return handle


def parse(stream, content_type: str, length: int, work_dir: Path,
          max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[dict[str, str], dict[str, Upload]]:
    """Return the plain fields and the uploaded files of one request.

    Files land in work_dir under generated names; the caller owns them and is
    responsible for deleting them.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    delimiter = b"--" + boundary_from(content_type)

    body = _buffer_body(stream, length, max_bytes, work_dir)
    body_path = Path(body.name)
    fields: dict[str, str] = {}
    files: dict[str, Upload] = {}

    try:
        name: str | None = None
        filename: str | None = None
        reading_headers = False
        header_lines: list[bytes] = []
        target = None
        target_path: Path | None = None
        chunks: list[bytes] = []

        def finish() -> None:
            """Close off the part just read, trimming its trailing newline.

            The CRLF before a boundary belongs to the delimiter, not to the
            content, so leaving it in would corrupt every uploaded file by two
            bytes and put a stray blank line on every text field.
            """
            nonlocal target, target_path, name, filename, chunks
            if name is None:
                return
            if target is not None and target_path is not None:
                position = target.tell()
                target.seek(max(0, position - 2))
                tail = target.read(2)
                trim = 2 if tail == b"\r\n" else (1 if tail.endswith(b"\n") else 0)
                size = position - trim
                # truncate() does not move the file position, so the size has to
                # be the value computed above rather than a later tell().
                target.truncate(size)
                target.close()
                files[name] = Upload(filename or "recording", target_path, size)
            else:
                value = b"".join(chunks)
                if value.endswith(b"\r\n"):
                    value = value[:-2]
                elif value.endswith(b"\n"):
                    value = value[:-1]
                fields[name] = value.decode("utf-8", "replace")
            target, target_path, name, filename, chunks = None, None, None, None, []

        for line in body:
            if line.startswith(delimiter):
                finish()
                if line.rstrip(b"\r\n").endswith(b"--"):
                    break
                reading_headers = True
                header_lines = []
                continue

            if reading_headers:
                if line in (b"\r\n", b"\n"):
                    reading_headers = False
                    disposition = ""
                    for raw in header_lines:
                        text = raw.decode("latin-1")
                        if text.split(":", 1)[0].strip().casefold() == "content-disposition":
                            disposition = text
                    found = NAME_RE.search(disposition)
                    name = found.group(1) if found else None
                    found = FILENAME_RE.search(disposition)
                    filename = found.group(1) if found else None
                    if name is not None and filename is not None:
                        target = tempfile.NamedTemporaryFile(
                            dir=work_dir, suffix=".upload", delete=False
                        )
                        target_path = Path(target.name)
                else:
                    header_lines.append(line)
                continue

            if name is None:
                continue  # preamble, or a part with no name to file it under
            if target is not None:
                target.write(line)
            else:
                chunks.append(line)
        finish()
    finally:
        body.close()
        body_path.unlink(missing_ok=True)

    return fields, files
