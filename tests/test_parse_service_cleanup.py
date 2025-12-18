"""Tests for artifact cleanup behavior in parse_service."""

from pathlib import Path

import pytest

from app.replay_parser.src import parse_service


def _patch_common(monkeypatch, tmp_path: Path):
    """Patch common parse_service dependencies for deterministic tests."""
    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)

    class DummyPlayer:
        def __init__(self, name: str, hero: str):
            self.name = name
            self.hero = hero

    class DummyHeader:
        players = [DummyPlayer("Alpha", "Nova"), DummyPlayer("Bravo", "Raynor")]

    monkeypatch.setattr(parse_service, "extract_header_from_file", lambda _path: "header")
    monkeypatch.setattr(parse_service, "extract_details_from_file", lambda _path, game_header: DummyHeader)
    monkeypatch.setattr(parse_service, "parse_tracker_events_file", lambda _path: [])
    monkeypatch.setattr(
        parse_service,
        "build_dynamic_rows",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        parse_service,
        "render_stats_store_to_png",
        lambda **kwargs: b"\x89PNG\r\n\x1a\n",
    )


def _create_artifacts(tmp_path: Path) -> dict[str, str]:
    """Create dummy artifact files and return the artifact map."""
    details = tmp_path / "example.details"
    tracker = tmp_path / "example.trackerevents"
    header = tmp_path / "example.header"
    for p in (details, tracker, header):
        p.write_text("dummy", encoding="utf-8")
    return {"details": str(details), "trackerevents": str(tracker), "header": str(header)}


def test_parse_and_build_table_removes_artifacts_after_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Delete artifact files after a successful parse_and_build_table run."""
    _patch_common(monkeypatch, tmp_path)

    artifacts = _create_artifacts(tmp_path)

    def fake_parse(_replay: str, flags=None):
        return artifacts

    monkeypatch.setattr(parse_service, "parse_replay_with_cli", fake_parse)

    png = parse_service.parse_and_build_table("dummy")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    for value in artifacts.values():
        assert not Path(value).exists()


def test_parse_and_build_table_cleans_up_even_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Delete artifact files even when the build pipeline raises."""
    _patch_common(monkeypatch, tmp_path)

    artifacts = _create_artifacts(tmp_path)

    def fake_parse(_replay: str, flags=None):
        return artifacts

    monkeypatch.setattr(parse_service, "parse_replay_with_cli", fake_parse)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(parse_service, "build_dynamic_rows", explode)

    with pytest.raises(RuntimeError):
        parse_service.parse_and_build_table("dummy")

    for value in artifacts.values():
        assert not Path(value).exists()


def test_derive_winning_team_uses_team_id_and_result():
    """Derive winning team from teamId + result (1=win, 2=loss)."""

    class P:
        def __init__(self, team_id: int, result: int):
            setattr(self, "teamId", team_id)
            self.result = result

    players = [
        P(0, 2),
        P(0, 2),
        P(1, 1),
        P(1, 1),
    ]
    assert parse_service._derive_winning_team(players) == "red"


def test_derive_winning_team_falls_back_to_halves_when_team_id_missing():
    """Fall back to first-half/second-half ordering if teamId is unavailable."""

    class P:
        def __init__(self, result: int):
            self.result = result

    players = [
        P(1),
        P(1),
        P(2),
        P(2),
    ]
    assert parse_service._derive_winning_team(players) == "blue"
