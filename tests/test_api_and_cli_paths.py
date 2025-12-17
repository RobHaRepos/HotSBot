from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.replay_parser.src import core
from app.replay_parser.src import parse_service
from app.replay_parser.src import parser_api


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    """Calling configure_logging multiple times does not add duplicate handlers."""
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        root.handlers.clear()
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        core.configure_logging()
        assert root.handlers

        before = len(root.handlers)
        core.configure_logging()
        assert len(root.handlers) == before
    finally:
        root.handlers[:] = saved


def test_parse_api_error_str():
    """Test the string representation of ParseApiError."""
    err = core.ParseApiError(status_code=500, detail="boom")
    assert str(err) == "Parse API error (500): boom"


def test_parser_api_maps_known_errors_to_400(monkeypatch: pytest.MonkeyPatch):
    """Test that known errors are mapped to HTTP 400."""
    def boom(_path: str) -> bytes:
        raise core.MissingReplayArtifactsError("missing")

    monkeypatch.setattr(parser_api, "parse_and_build_table", boom)

    with TestClient(parser_api.app) as client:
        resp = client.post("/parse-replay/", json={"replay_path": "dummy"})

    assert resp.status_code == 400


def test_parser_api_maps_unknown_errors_to_500(monkeypatch: pytest.MonkeyPatch):
    """Test that unknown errors are mapped to HTTP 500."""
    def boom(_path: str) -> bytes:
        raise RuntimeError("nope")

    monkeypatch.setattr(parser_api, "parse_and_build_table", boom)

    with TestClient(parser_api.app) as client:
        resp = client.post("/parse-replay/", json={"replay_path": "dummy"})

    assert resp.status_code == 500


def test_parser_api_upload_cleans_up_even_if_unlink_fails(monkeypatch: pytest.MonkeyPatch):
    """Test that the uploaded file is cleaned up even if unlink raises an exception."""
    monkeypatch.setattr(parser_api.parse_service, "parse_and_build_table", lambda p: (_ for _ in ()).throw(ValueError("bad replay")))

    def unlink_raises(self: Path):
        raise OSError("locked")

    monkeypatch.setattr(parser_api.Path, "unlink", unlink_raises, raising=True)

    with TestClient(parser_api.app) as client:
        resp = client.post(
            "/parse-replay/upload",
            files={"file": ("test.StormReplay", b"abc", "application/octet-stream")},
        )

    assert resp.status_code == 400


def test_parse_replay_with_cli_skips_failed_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that parse_replay_with_cli skips flags that fail."""
    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)

    def fake_run(_cmd, capture_output=True, text=True):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad")

    monkeypatch.setattr(parse_service.subprocess, "run", fake_run)

    out = parse_service.parse_replay_with_cli("dummy", flags=["--header"])
    assert out == {}


def test_parse_replay_with_cli_returns_empty_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that parse_replay_with_cli returns empty dict if an exception occurs."""
    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)

    def fake_run(_cmd, capture_output=True, text=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(parse_service.subprocess, "run", fake_run)

    out = parse_service.parse_replay_with_cli("dummy", flags=["--header"])
    assert out == {}


def test_parse_and_build_table_team_level_parse_failure_is_handled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that parse_and_build_table handles team level parse failures gracefully."""
    artifacts = {
        "details": str(tmp_path / "d.txt"),
        "trackerevents": str(tmp_path / "t.txt"),
        "header": str(tmp_path / "h.txt"),
    }
    for p in artifacts.values():
        Path(p).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(parse_service, "parse_replay_with_cli", lambda *_a, **_k: artifacts)

    class DummyPlayer:
        def __init__(self, name: str, hero: str):
            self.name = name
            self.hero = hero

    class DummyHeader:
        elapsed_seconds = 123
        players = [DummyPlayer("A", "HeroA"), DummyPlayer("B", "HeroB")]

    monkeypatch.setattr(parse_service, "extract_header_from_file", lambda *_a, **_k: DummyHeader())
    monkeypatch.setattr(parse_service, "extract_details_from_file", lambda *_a, **_k: DummyHeader())
    monkeypatch.setattr(parse_service, "parse_tracker_events_file", lambda *_a, **_k: [])

    def build_rows(*, store, **_kw):
        """Build rows and force team level parse failure."""
        store.set_row("Team Level Achieved", ["x", "y"])  # force int() failure

    monkeypatch.setattr(parse_service, "build_dynamic_rows", build_rows)
    monkeypatch.setattr(parse_service, "get_or_download_hero_image_path", lambda *_a, **_k: None)
    monkeypatch.setattr(parse_service, "render_stats_store_to_png", lambda **_kw: b"\x89PNG\r\n\x1a\n")

    out = parse_service.parse_and_build_table("dummy")
    assert out[:8] == b"\x89PNG\r\n\x1a\n"

    # cleanup should have removed artifacts
    assert all(not Path(p).exists() for p in artifacts.values())
