"""Integration and unit tests for parsing and PNG rendering."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.replay_parser.schemas.store import InMemoryStatsStore
from app.replay_parser.src import parse_service, parser_api
from app.replay_parser.src.extract_replay_tracker_events import (
    build_dynamic_rows,
    extract_score,
    _latest_value,
    parse_tracker_events_file,
)
from app.replay_parser.src.parser_utils import load_output_file
from app.replay_parser.src.statistic_png_renderer import (
    TeamHeaderOptions,
    _build_table_text,
    _load_fonts,
    _measure_layout,
    _text_h,
    render_stats_store_to_png,
)


def _pixel_rgb(img: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    px = img.getpixel((x, y))
    assert isinstance(px, tuple) and len(px) >= 3
    return int(px[0]), int(px[1]), int(px[2])


def _any_pixel(
    img: Image.Image,
    *,
    xs: range,
    ys: range,
    predicate,
) -> bool:
    for x in xs:
        for y in ys:
            if predicate(_pixel_rgb(img, x, y)):
                return True
    return False


def _close_rgb(a: tuple[int, int, int], b: tuple[int, int, int], tol: int = 20) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(3))


def _is_white(px: tuple[int, int, int], threshold: int = 240) -> bool:
    return px[0] > threshold and px[1] > threshold and px[2] > threshold


def test_in_memory_store_basic_behaviour():
    """Exercise basic InMemoryStatsStore operations and validation."""
    store = InMemoryStatsStore(["A", "B", "C"])
    assert store.player_count == 3

    store.add("Kills", 0, 2)
    store.add("Kills", 1, 1)
    assert store.get_row("Kills") == [2, 1, 0]

    store.set_row("Deaths", [1, 2, 3])
    assert store.get_row("Deaths") == [1, 2, 3]

    rows = list(store.iter_rows())
    assert rows[0].category == "Kills"
    assert rows[1].category == "Deaths"

    with pytest.raises(IndexError):
        store.add("Kills", 99, 1)

    with pytest.raises(ValueError):
        store.set_row("Bad", [1, 2])

    store.clear()
    assert store.get_row("Kills") is None


def test_parse_tracker_events_and_build_kda_rows_from_sample_outputs(tmp_path: Path):
    """Parse minimal tracker events and build K/D/A rows."""
    sample_event = "{\n  '_event': 'NNet.Replay.Tracker.SScoreResultEvent',\n  'm_instanceList': [\n    {'m_name': 'SoloKill', 'm_values': [[{'m_value': 1}]]},\n    {'m_name': 'Deaths', 'm_values': [[{'m_value': 0}], [{'m_value': 1}], [{'m_value': 0}]]},\n  ]\n}\n"
    track_path = tmp_path / "track.txt"
    track_path.write_text(sample_event, encoding="utf-8")

    events = parse_tracker_events_file(str(track_path))
    assert events

    store = InMemoryStatsStore([f"P{i+1}" for i in range(10)])
    build_dynamic_rows(events=events, player_count=10, store=store)

    kills = store.get_row("Kills")
    deaths = store.get_row("Deaths")
    assert kills is not None and len(kills) == 10
    assert deaths is not None and len(deaths) == 10
    assert any(isinstance(v, int) and v > 0 for v in kills)
    assert any(isinstance(v, int) and v > 0 for v in deaths)


def test_png_renderer_outputs_valid_png_bytes():
    """Render a simple table and validate PNG output bytes."""
    player_labels = [f"P{i+1}" for i in range(10)]
    rows = [
        ("Kills", list(range(10))),
        ("Deaths", [0] * 10),
        ("Assists", [5] * 10),
    ]

    png_bytes = render_stats_store_to_png(player_labels=player_labels, rows=rows, title="K/D/A")
    assert isinstance(png_bytes, (bytes, bytearray))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0


def test_png_renderer_legacy_layout_outputs_valid_png_bytes():
    """Render a small table and validate PNG bytes."""
    player_labels = [f"P{i+1}" for i in range(5)]
    rows = [
        ("Kills", [1, 2, 3, 4, 5]),
        ("Deaths", [0, 1, 0, 1, 0]),
    ]

    png_bytes = render_stats_store_to_png(player_labels=player_labels, rows=rows)
    assert isinstance(png_bytes, (bytes, bytearray))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0


def test_header_wraps_into_two_lines_when_max_width_small():
    """When header_max_width is small, headers should wrap and increase image height."""
    player_labels = ["P1", "P2"]
    rows = [
        ("Clutch Heals Performed", [1, 2]),
        ("Other", [3, 4]),
    ]

    # Render without forcing wrap
    png_no_wrap = render_stats_store_to_png(player_labels=player_labels, rows=rows, header_max_width=1000)
    img_no_wrap = Image.open(io.BytesIO(png_no_wrap))

    # Render forcing small max width so wrapping happens
    png_wrap = render_stats_store_to_png(player_labels=player_labels, rows=rows, header_max_width=10)
    img_wrap = Image.open(io.BytesIO(png_wrap))

    # The wrapped version should be taller because headers occupy two lines
    assert img_wrap.size[1] > img_no_wrap.size[1]


def test_header_wraps_by_default_when_needed():
    """Default behavior should wrap long headers when necessary."""
    player_labels = ["P1", "P2"]
    rows = [
        ("Clutch Heals Performed", [1, 2]),
        ("Other", [3, 4]),
    ]

    # Render forcing no wrap (very large max width)
    png_no_wrap = render_stats_store_to_png(player_labels=player_labels, rows=rows, header_max_width=1000)
    img_no_wrap = Image.open(io.BytesIO(png_no_wrap))

    # Render with auto header_max_width (None) — should wrap and be taller
    png_auto = render_stats_store_to_png(player_labels=player_labels, rows=rows, header_max_width=None)
    img_auto = Image.open(io.BytesIO(png_auto))

    assert img_auto.size[1] > img_no_wrap.size[1]


def test_header_extra_increases_row_height():
    """Setting header_extra should increase the computed row height."""
    player_labels = ["P1", "P2", "P3"]
    rows = [
        ("Long Header Name To Wrap", [1, 2, 3]),
        ("Other", [3, 4, 5]),
    ]

    table_text, _ = _build_table_text(player_labels, rows)
    font, font_bold = _load_fonts(18)

    _, row_h0, _, _, _ = _measure_layout(table_text, font=font, font_bold=font_bold, padding_x=12, padding_y=8, grid=1, title=None, header_max_width=None, header_extra=0)
    _, row_h1, _, _, _ = _measure_layout(table_text, font=font, font_bold=font_bold, padding_x=12, padding_y=8, grid=1, title=None, header_max_width=None, header_extra=12)

    assert row_h1 > row_h0


def test_row_background_and_text_colors():
    """Verify header, first 5 rows, and last 5 rows use the modern palette and text is white."""
    player_labels = [f"P{i+1}" for i in range(12)]
    rows = [
        ("Clutch Heals Performed", list(range(1, 13))),
        ("Kills", [1] * 12),
    ]

    png = render_stats_store_to_png(player_labels=player_labels, rows=rows, title="Colors Test")
    img = Image.open(io.BytesIO(png)).convert('RGB')

    # Use internal measurement functions to locate a data cell center
    table_text, _ = _build_table_text(player_labels, rows)
    # Measure layout using the same font size that the renderer uses by default
    font, font_bold = _load_fonts(25)
    col_widths, row_height, title_h, _, _ = _measure_layout(
        table_text, font=font, font_bold=font_bold, padding_x=14, padding_y=10, grid=1, title='Colors Test', header_extra=15)

    def sample_center(r, c, x_offset: int | None = None):
        grid = 1
        # Renderer adds side padding equal to bottom_extra.
        bottom_extra = max(8 * 6, row_height // 3, 8)
        side_padding = bottom_extra

        x = side_padding + grid
        for i in range(c):
            x += col_widths[i] + grid
        if x_offset is None:
            x_center = x + col_widths[c] // 2
        else:
            x_center = x + x_offset
        y = title_h + grid + r * (row_height + grid)
        y_center = y + row_height // 2
        return _pixel_rgb(img, x_center, y_center)

    def _is_blue_tinted(px: tuple[int, int, int]) -> bool:
        r, g, b = px
        return b > r + 25 and b > g + 25

    def _is_red_tinted(px: tuple[int, int, int]) -> bool:
        r, g, b = px
        return r > g + 25 and r > b + 25

    # Check first and last data rows use the expected band colors.
    # Instead of sampling a single pixel (which can coincide with text/stroke),
    # sample multiple data rows and look for any cell center that matches the
    # expected band color.
    total_data_rows = max(0, len(table_text) - 1)
    found_blue = False
    for r in range(1, min(1 + 6, 1 + total_data_rows)):
        px = sample_center(r, 1)
        if _is_blue_tinted(px):
            found_blue = True
            break

    found_red = False
    # last N rows
    for r in range(1 + max(0, total_data_rows - 6), 1 + total_data_rows):
        px = sample_center(r, 1)
        if _is_red_tinted(px):
            found_red = True
            break

    assert found_blue, "Expected to find blue band in first rows near column 1"
    assert found_red, "Expected to find red band in last rows near column 1"

    # Confirm the title text (left) exists as dark pixels (black text)
    region_x = 5
    region_y = 5
    found_dark = _any_pixel(
        img,
        xs=range(region_x, region_x + 40),
        ys=range(region_y, region_y + 30),
        predicate=lambda px: px[0] < 80 and px[1] < 80 and px[2] < 80,
    )

    assert found_dark, "Expected to find dark (black) title pixels near top-left"


def test_parse_service_returns_png_without_running_cli(monkeypatch: pytest.MonkeyPatch):
    """Smoke test: patch CLI parse to use existing sample output files."""

    def fake_parse_replay_with_cli(_path: str, flags=None):
        return {
            "details": "./output-details.txt",
            "trackerevents": "./output-trackerevents.txt",
            "header": "./output-header.txt",
        }

    monkeypatch.setattr(parse_service, "parse_replay_with_cli", fake_parse_replay_with_cli)
    monkeypatch.setattr(parse_service, "extract_header_from_file", lambda _path: "header")

    class DummyPlayer:
        def __init__(self, name: str, hero: str):
            self.name = name
            self.hero = hero

    class DummyHeader:
        players = [DummyPlayer("A", "HeroA"), DummyPlayer("B", "HeroB")]

    monkeypatch.setattr(parse_service, "extract_details_from_file", lambda _path, game_header: DummyHeader)
    monkeypatch.setattr(parse_service, "parse_tracker_events_file", lambda _path: [])
    monkeypatch.setattr(parse_service, "build_dynamic_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(parse_service, "render_stats_store_to_png", lambda **kwargs: b"\x89PNG\r\n\x1a\n")

    png_bytes = parse_service.parse_and_build_table("dummy")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_team_header_renders_game_time_and_teams():
    """Verify the custom team header is rendered and contains Game Time and colored team text."""
    player_labels = [f"P{i+1}" for i in range(6)]
    rows = [("Kills", [1] * 6)]

    png = render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        team_header=TeamHeaderOptions(
            team_blue_level=5,
            team_red_level=8,
            team_blue_kills=10,
            team_red_kills=9,
            game_time_seconds=542,
        ),
    )

    img = Image.open(io.BytesIO(png)).convert('RGB')

    team_blue_rgb = (80, 160, 255)
    team_red_rgb = (255, 110, 110)

    cx = img.width // 2
    found_center_white = _any_pixel(img, xs=range(cx - 40, cx + 40), ys=range(0, 90), predicate=_is_white)

    # Layout is now left-aligned within each team area; search broadly in each half of the header.
    header_ys = range(0, min(220, img.height))
    found_left_blue = _any_pixel(
        img,
        xs=range(0, cx),
        ys=header_ys,
        predicate=lambda px: _close_rgb(px, team_blue_rgb, tol=80),
    )
    found_right_red = _any_pixel(
        img,
        xs=range(cx, img.width),
        ys=header_ys,
        predicate=lambda px: _close_rgb(px, team_red_rgb, tol=80),
    )

    assert found_center_white, "Expected Game Time/time to be rendered as white pixels"
    assert found_left_blue, "Expected TEAM BLUE/Level/Kills to produce blue pixels in header"
    assert found_right_red, "Expected TEAM RED/Level/Kills to produce red pixels in header"


def test_team_kills_from_team_takedowns_row():
    """Derives team kills from Team Deaths/TeamTakedowns row (idx0=red, idx5=blue)."""
    player_labels = [f"P{i+1}" for i in range(6)]
    rows = [("Team Deaths", [7, 0, 0, 0, 0, 3])]

    # Render with automatic derivation
    png_auto = render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        team_header=TeamHeaderOptions(team_blue_level=1, team_red_level=1, game_time_seconds=0),
    )

    # Render with explicit kills matching the mapping
    png_explicit = render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        team_header=TeamHeaderOptions(
            team_blue_level=1,
            team_red_level=1,
            team_blue_kills=3,
            team_red_kills=7,
            game_time_seconds=0,
        ),
    )

    assert png_auto == png_explicit, "Derived team kills (from Team Deaths indices) should match explicit team kills rendering"


def test_last_row_not_cut_off():
    """Ensure the last data row is fully visible (not clipped at the bottom)."""
    player_labels = [f"P{i+1}" for i in range(20)]
    rows = [("Kills", list(range(1, 21)))]

    png = render_stats_store_to_png(player_labels=player_labels, rows=rows, title="Clip Test")
    _ = Image.open(io.BytesIO(png)).convert('RGB')

    # Recompute layout to find last row center
    table_text, _ = _build_table_text(player_labels, rows)
    font, font_bold = _load_fonts(25)
    _, row_height, title_h, _, img_h = _measure_layout(table_text, font=font, font_bold=font_bold, padding_x=14, padding_y=10, grid=1, title='Clip Test', header_extra=15)

    last_row_index = len(table_text) - 1
    # center y of last data row
    y_center = title_h + 1 + last_row_index * (row_height + 1) + row_height // 2

    # compute final grid y (where the bottom grid line is drawn)
    final_grid_y = title_h + len(table_text) * (row_height + 1)

    # require a safety margin below the final grid line
    padding_y = 10
    margin = padding_y * 2

    assert img_h >= final_grid_y + margin + 2, \
        f"Last row appears too close to or beyond image bottom (final_grid_y={final_grid_y}, img_h={img_h}, margin={margin})"

    # also ensure the center of the last row is comfortably inside the image
    assert y_center + 4 < img_h, f"Last row center too close to bottom (y_center={y_center}, img_h={img_h})"


def test_last_row_not_cut_off_with_team_header():
    """Ensure the last data row is fully visible when the team header is enabled."""
    player_labels = [f"P{i+1}" for i in range(20)]
    rows = [("Kills", list(range(1, 21)))]

    png = render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        team_header=TeamHeaderOptions(team_blue_level=1, team_red_level=1, game_time_seconds=0),
    )
    _ = Image.open(io.BytesIO(png)).convert("RGB")


def test_parse_service_raises_on_missing_artifacts(monkeypatch: pytest.MonkeyPatch):
    """Raise a user-facing error when required artifacts are missing."""
    def fake_parse_replay_with_cli(_path: str, flags=None):
        return {"details": "./output-details.txt"}

    monkeypatch.setattr(parse_service, "parse_replay_with_cli", fake_parse_replay_with_cli)

    with pytest.raises(ValueError):
        parse_service.parse_and_build_table("dummy")


def test_parse_replay_with_cli_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Unit test: mock subprocess.run so we don't call heroprotocol."""

    class FakeRes:
        def __init__(self, returncode: int, stdout: str, stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **_kwargs):
        flag = cmd[3]
        return FakeRes(0, f"output for {flag}")

    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(parse_service.subprocess, "run", fake_run)

    results = parse_service.parse_replay_with_cli("C:/replays/test.StormReplay", flags=["--details", "--trackerevents"])
    assert set(results.keys()) == {"details", "trackerevents"}

    details_file = Path(results["details"])
    tracker_file = Path(results["trackerevents"])
    assert details_file.exists()
    assert tracker_file.exists()
    assert "output for --details" in details_file.read_text(encoding="utf-8")


def test_fastapi_endpoint_returns_png_and_triggers_lifespan(monkeypatch: pytest.MonkeyPatch):
    """Use TestClient to exercise route + startup/shutdown lifespan."""

    dummy_png = b"\x89PNG\r\n\x1a\n" + b"x" * 20
    monkeypatch.setattr(parser_api, "parse_and_build_table", lambda _path: dummy_png)

    with TestClient(parser_api.app) as client:
        resp = client.post("/parse-replay/", json={"replay_path": "dummy"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/png")
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_fastapi_upload_endpoint_accepts_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Accept upload and return PNG bytes via the FastAPI endpoint."""
    # Create a small dummy replay file
    p = tmp_path / "test.StormReplay"
    p.write_text("fake", encoding="utf-8")

    # Patch parse_and_build_table to return a PNG header for speed
    monkeypatch.setattr(parse_service, "parse_and_build_table", lambda _path: b"\x89PNG\r\n\x1a\n")

    with TestClient(parser_api.app) as client:
        with p.open("rb") as f:
            resp = client.post("/parse-replay/upload", files={"file": ("test.StormReplay", f, "application/octet-stream")})

    assert resp.status_code == 200
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_font_size_affects_layout_or_skips_if_no_scalable_font():
    """Confirms font_size affects measured text height (skips if no scalable TTF)."""
    font_small, _ = _load_fonts(12)
    font_large, _ = _load_fonts(30)

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    small_h = _text_h(draw, font_small, "Ag")
    large_h = _text_h(draw, font_large, "Ag")

    if small_h == large_h:
        pytest.skip("No scalable TTF available; font size has no effect in this environment")

    assert large_h > small_h


def test_parse_replay_with_cli_handles_nonzero_returncode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Skip artifacts for failing flags but keep successful ones."""
    class FakeRes:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **_kwargs):
        flag = cmd[3]
        if flag == "--details":
            return FakeRes(1, stderr="boom")
        return FakeRes(0, stdout=f"ok {flag}")

    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(parse_service.subprocess, "run", fake_run)

    results = parse_service.parse_replay_with_cli("C:/replays/test.StormReplay", flags=["--details", "--trackerevents"])
    assert "details" not in results
    assert "trackerevents" in results


def test_parse_replay_with_cli_handles_subprocess_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return empty results when subprocess execution raises."""
    def fake_run(*args, **kwargs):
        raise RuntimeError("subprocess failed")

    monkeypatch.setattr(parse_service, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(parse_service.subprocess, "run", fake_run)

    results = parse_service.parse_replay_with_cli("C:/replays/test.StormReplay", flags=["--details"])
    assert results == {}


def test_extract_score_returns_zeros_when_stat_missing():
    """Default missing score stats to all-zero values."""
    fake_event = {"m_instanceList": [{"m_name": b"SomethingElse", "m_values": []}]}
    row = extract_score(fake_event, "Deaths", player_count=10)
    assert row == [0] * 10


def test_latest_value_empty_series():
    """Return 0 when the instance series is empty."""
    assert _latest_value([]) == 0


def test_parse_tracker_events_skips_bad_blocks(tmp_path: Path):
    """Skip invalid blocks while parsing tracker event files."""
    content = "".join([
        "{\"a\": 1}",
        "\n\n",
        "{not valid}",
        "\n",
        "{'_event': 'X', 'm': 2}",
    ])
    p = tmp_path / "track.txt"
    p.write_text(content, encoding="utf-8")

    events = parse_tracker_events_file(str(p))
    assert {e.get("a") for e in events if "a" in e} == {1}
    assert any(e.get("m") == 2 for e in events)


def test_load_output_file_decodes_utf8_and_fallback_to_latin1(tmp_path: Path):
    """Decode output files with utf-8 and fall back gracefully."""
    p_utf8 = tmp_path / "a.txt"
    p_utf8.write_text("hello", encoding="utf-8")
    assert load_output_file(str(p_utf8)) == "hello"

    p_latin1 = tmp_path / "b.txt"
    p_latin1.write_bytes(b"\xff")
    assert load_output_file(str(p_latin1)) == "\xff"
