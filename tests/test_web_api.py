import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient
from kikuyu_ai.web.api import app

client = TestClient(app)


def test_root_serves_the_installable_mobile_app():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'href="manifest.webmanifest"' in response.text
    # The phone must not pin an old shell; the service worker handles offline use.
    assert response.headers["cache-control"] == "no-store"


def test_status_page_still_links_the_api_docs():
    response = client.get("/status")

    assert response.status_code == 200
    assert "API running" in response.text
    assert 'href="/docs"' in response.text


@pytest.mark.parametrize(
    "path, content_type",
    [
        ("/app.js", "javascript"),
        ("/styles.css", "css"),
        ("/sw.js", "javascript"),
        ("/manifest.webmanifest", "json"),
        ("/icons/icon-192.png", "image/png"),
    ],
)
def test_app_shell_assets_are_served_from_the_site_root(path, content_type):
    """The service worker's scope is "/", so its assets must live there too."""
    response = client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_health_reports_what_the_app_can_actually_do():
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert payload["offline"] is True
    for key in ("asr_model", "translation_model", "bible_verses", "tts", "max_upload_mb"):
        assert key in payload


def test_unknown_paths_are_not_swallowed_by_the_static_mount():
    assert client.get("/definitely-not-a-real-asset.css").status_code == 404


def test_api_routes_still_win_over_the_static_mount():
    assert client.post("/translate-text", json={"text": "  "}).status_code == 400
