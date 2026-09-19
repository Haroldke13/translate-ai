from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class RemoteAPIError(RuntimeError):
    pass


def _base_url(api_url: str) -> str:
    api_url = api_url.strip()
    if not api_url:
        raise ValueError("api_url cannot be empty")
    return api_url.rstrip("/") + "/"


def absolutize_url(api_url: str, value: object) -> object:
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    return urljoin(_base_url(api_url), value.lstrip("/"))


def absolutize_session_urls(api_url: str, payload: dict) -> dict:
    result = dict(payload)
    for key in ("audio", "subtitles", "english_subtitles"):
        if key in result:
            result[key] = absolutize_url(api_url, result[key])
    if isinstance(result.get("file_urls"), dict):
        result["file_urls"] = {
            str(name): absolutize_url(api_url, value) for name, value in result["file_urls"].items()
        }
    return result


def _read_json_response(response) -> dict:
    body = response.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RemoteAPIError(f"remote API returned non-JSON response: {body[:500]}") from exc
    if not isinstance(payload, dict):
        raise RemoteAPIError("remote API returned a non-object JSON response")
    return payload


def _request_json(request: Request, timeout: float) -> dict:
    try:
        with urlopen(request, timeout=timeout) as response:
            return _read_json_response(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        raise RemoteAPIError(f"remote API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RemoteAPIError(f"could not reach remote API: {exc.reason}") from exc


def check_remote_api(api_url: str, timeout: float = 2.0) -> dict:
    request = Request(
        urljoin(_base_url(api_url), "health"),
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "kikuyu-ai-client/0.1",
        },
    )
    return _request_json(request, timeout)


def _multipart_audio_body(audio_path: Path) -> tuple[bytes, str]:
    boundary = f"kikuyu-ai-{uuid.uuid4().hex}"
    filename = audio_path.name or "audio.wav"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + audio_path.read_bytes() + footer, boundary


def translate_audio_remote(
    api_url: str,
    audio_path: Path,
    session_id: str | None = None,
    timeout: float | None = None,
) -> dict:
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    query = f"?{urlencode({'session_id': session_id})}" if session_id else ""
    body, boundary = _multipart_audio_body(audio_path)
    request = Request(
        urljoin(_base_url(api_url), f"translate{query}"),
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
            "User-Agent": "kikuyu-ai-client/0.1",
        },
    )
    payload = _request_json(request, timeout or float(os.getenv("KIKUYU_API_TIMEOUT", "900")))
    return absolutize_session_urls(api_url, payload)


def translate_text_remote(api_url: str, text: str, timeout: float | None = None) -> dict:
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    request = Request(
        urljoin(_base_url(api_url), "translate-text"),
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "kikuyu-ai-client/0.1",
        },
    )
    payload = _request_json(request, timeout or float(os.getenv("KIKUYU_API_TIMEOUT", "900")))
    return absolutize_session_urls(api_url, payload)
