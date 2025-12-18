"""Unit tests for statistic_png_renderer helpers and rendering extras."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from app.replay_parser.src import statistic_png_renderer as r


def test_format_cell_variants():
    """Format common cell value types consistently."""
    assert r._format_cell(None) == ""
    assert r._format_cell(1234) == "1,234"
    assert r._format_cell(True) == "True"
    assert r._format_cell(1.2345) == "1.23"
    assert r._format_cell("x") == "x"


def test_safe_int_variants():
    """Parse integers safely with None/zero fallbacks."""
    assert r._safe_int("42") == 42
    assert r._safe_int(object()) is None
    assert r._safe_int_or_zero("5") == 5
    assert r._safe_int_or_zero(object()) == 0


def test_build_background_image_falls_back_to_white_on_open_error(monkeypatch: pytest.MonkeyPatch):
    """Fall back to a plain background when the image cannot be opened."""
    def open_raises(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setattr(r.Image, "open", open_raises)
    img = r._build_background_image(50, 20)
    assert img.size == (50, 20)


def test_build_background_image_falls_back_when_size_invalid(monkeypatch: pytest.MonkeyPatch):
    """Fall back when the background image has an invalid size."""
    class _Fake:
        size = (0, 0)

        def convert(self, _mode: str):
            return self

    monkeypatch.setattr(r.Image, "open", lambda *_a, **_k: _Fake())
    img = r._build_background_image(50, 20)
    assert img.size == (50, 20)


def test_talent_header_detection_and_apply_sizing():
    """Detect talent columns and apply the talent sizing policy."""
    table_text = [
        ["Stat", "Talent 1", "Other"],
        ["P1", "A", "B"],
    ]
    assert r._is_talent_header("Talent 1")
    assert r._is_talent_header("   ") is False
    assert r._detect_talent_cols(table_text) == {1}

    col_widths = [10, 10, 10]
    col_widths2, row_h, img_w, img_h, opts = r._apply_talent_sizing(
        [row[:] for row in table_text],
        col_widths=col_widths,
        row_height=12,
        font_size=12,
        padding_x=4,
        padding_y=2,
        grid=1,
        title_h=0,
    )

    assert col_widths2[1] >= 10
    assert row_h >= 12
    assert img_w > 0 and img_h > 0
    assert opts is not None


def test_detect_talent_cols_and_dimensions_empty_inputs():
    """Return empty/default dimensions when there is no data."""
    assert r._detect_talent_cols([]) == set()
    assert r._recompute_img_dimensions(table_text=[], col_widths=[], row_height=1, grid=1, padding_y=1, title_h=0) == (0, 0)


def test_load_fonts_falls_back_to_default_when_truetype_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Fall back to default bitmap fonts when TrueType loading fails."""
    orig_truetype = r.ImageFont.truetype

    def fake_truetype(font, *a, **k):
        if isinstance(font, (str, Path)):
            raise OSError("no font")
        return orig_truetype(font, *a, **k)

    monkeypatch.setattr(r.ImageFont, "truetype", fake_truetype)
    font, font_bold = r._load_fonts(12)
    assert font is not None and font_bold is not None


def test_load_fonts_uses_regular_when_bold_missing(monkeypatch: pytest.MonkeyPatch):
    """Use regular font when bold font cannot be loaded."""
    real_default = ImageFont.load_default()

    def fake_truetype(name: str, size: int):
        if "Bold" in name or "bd" in name:
            raise OSError("no bold")
        return real_default

    monkeypatch.setattr(r.ImageFont, "truetype", fake_truetype)
    font, font_bold = r._load_fonts(12)
    assert font is font_bold


def test_wrap_header_cells_handles_empty_and_skips_non_wrappable():
    """Skip wrapping for empty/non-wrappable header cells."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    _, font_bold = r._load_fonts(12)

    # empty table_text -> early return
    r._wrap_header_cells([], draw=draw, font_bold=font_bold, header_max_width=10)

    # header cell without spaces should be skipped
    table = [["NoSpaceHere", "Two Words"], ["x", "y"]]
    r._wrap_header_cells(table, draw=draw, font_bold=font_bold, header_max_width=1_000)
    assert "\n" not in table[0][0]


def test_wrap_header_cells_handles_header_with_only_spaces():
    """Handle whitespace-only header cells without errors."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    _, font_bold = r._load_fonts(12)

    table = [["   ", "Two Words"], ["x", "y"]]
    # Force it down the split path; the spaces-only header results in words=[] and should just continue.
    r._wrap_header_cells(table, draw=draw, font_bold=font_bold, header_max_width=-1)

def test_auto_wrap_threshold_and_maybe_wrap_headers_false_paths():
    """Avoid wrapping when thresholds/inputs disable header wrapping."""
    assert r._auto_header_wrap_threshold([]) == 150
    assert r._maybe_wrap_headers([], font_bold=r._load_fonts(12)[1], header_max_width=10) is False

    table = [["Short"], ["x"]]
    assert r._maybe_wrap_headers(table, font_bold=r._load_fonts(12)[1], header_max_width=10_000) is False


def test_derive_team_kills_and_levels_from_rows():
    """Derive team kills and levels from rows when values are absent."""
    rows = [
        ("Team Deaths", [1, 1, 2, 2]),
        ("Team Level Achieved", [5, 5, 6, 7]),
    ]
    blue_k, red_k = r._derive_team_kills(existing_blue_kills=None, existing_red_kills=None, rows=rows, player_count=4)
    assert blue_k == 2 and red_k == 4  # sums of halves

    blue_l, red_l = r._derive_team_levels(existing_blue_level=None, existing_red_level=None, rows=rows, player_count=4)
    assert blue_l == 5 and red_l == 7


def test_derive_team_levels_returns_existing_when_provided():
    """Keep existing team levels when explicit values are provided."""
    blue_l, red_l = r._derive_team_levels(
        existing_blue_level=10,
        existing_red_level=12,
        rows=[("Team Level Achieved", [1, 2, 3, 4])],
        player_count=4,
    )
    assert blue_l == 10 and red_l == 12


def test_format_game_time_defaults_to_zero_when_invalid():
    """Format invalid/negative game times as 00:00."""
    assert r._format_game_time(None) == "00:00"
    assert r._format_game_time(-1) == "00:00"


def test_ellipsize_edge_cases(monkeypatch: pytest.MonkeyPatch):
    """Handle ellipsize edge cases including tiny widths."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    font, _ = r._load_fonts(12)

    assert r._ellipsize(draw, font=font, text="abc", max_width=0) == ""

    # Force the ellipsis itself to be "too wide".
    orig_text_w = r._text_w

    def fake_text_w(d, f, s: str) -> int:
        if s in ("…", "�"):
            return 10_000
        return orig_text_w(d, f, s)

    monkeypatch.setattr(r, "_text_w", fake_text_w)
    assert r._ellipsize(draw, font=font, text="abcdef", max_width=5) == ""


def test_ellipsize_trims_when_text_is_too_wide():
    """Trim text with an ellipsis when it exceeds available width."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    font, _ = r._load_fonts(12)

    out = r._ellipsize(draw, font=font, text="abcdef", max_width=1)
    assert out in ("", "…") or out.endswith("…")


def test_draw_player_cell_handles_portrait_resize_exception():
    """Avoid raising if portrait resize fails."""
    dummy = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    font, font_bold = r._load_fonts(12)

    class BadPortrait:
        size = (64, 64)

        def resize(self, *_a, **_k):
            raise RuntimeError("nope")

    # Should not raise
    r._draw_player_cell(
        draw=draw,
        img=dummy,
        cell_x0=0,
        cell_y0=0,
        cell_w=150,
        row_h=40,
        padding_x=6,
        padding_y=4,
        player_name="P1",
        hero_name="Hero",
        portrait=BadPortrait(),  # type: ignore[arg-type]
        font=font,
        font_bold=font_bold,
    )


def test_draw_player_cell_ignores_paste_failures():
    """Ignore paste failures when composing the player portrait."""
    dummy = Image.new("RGBA", (200, 60), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    font, font_bold = r._load_fonts(12)

    portrait = Image.new("RGBA", (64, 64), (255, 0, 0, 255))

    class BadImg:
        def paste(self, *_a, **_k):
            raise RuntimeError("nope")

    r._draw_player_cell(
        draw=draw,
        img=BadImg(),  # type: ignore[arg-type]
        cell_x0=0,
        cell_y0=0,
        cell_w=150,
        row_h=40,
        padding_x=6,
        padding_y=4,
        player_name="P1",
        hero_name="Hero",
        portrait=portrait,
        font=font,
        font_bold=font_bold,
    )


def test_parse_number_and_highlight_helpers_cover_failure_modes():
    """Cover parsing/highlight helper failure modes and defaults."""
    assert r._parse_number("") is None
    assert r._parse_number("1.25") == pytest.approx(1.25)
    assert r._parse_number("nope") is None

    assert r._is_highlightable_stat("Kills")
    assert r._is_highlightable_stat("Something Else") is False


def test_compute_best_value_cells_handles_mismatched_masks_and_empty_data():
    """Handle empty/mismatched masks when computing best-value highlight cells."""
    table_text = [["Player", "Kills"], ["B1", "1"], ["R1", "2"]]
    table_is_numeric = [[False, False], [False, True], [False, True]]
    ctx = r._TableDrawContext(
        col_widths=[10, 10, 10, 10],
        row_height=12,
        start_x=0,
        start_y=0,
        padding_x=1,
        padding_y=1,
        grid=1,
    )

    cells = r._compute_best_value_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)
    assert cells  # should still find at least the numeric cell it can see

    ctx_empty = r._TableDrawContext(
        col_widths=[],
        row_height=12,
        start_x=0,
        start_y=0,
        padding_x=1,
        padding_y=1,
        grid=1,
    )
    assert r._compute_best_value_cells(table_text=[], table_is_numeric=[], ctx=ctx_empty) == set()


def test_draw_team_cell_backgrounds_returns_early_on_empty_table():
    """Return early without errors when drawing backgrounds for empty tables."""
    overlay = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ctx = r._TableDrawContext(
        col_widths=[],
        row_height=10,
        start_x=0,
        start_y=0,
        padding_x=1,
        padding_y=1,
        grid=1,
    )
    r._draw_team_cell_backgrounds(draw, table_text=[], ctx=ctx, highlight_cells=set())


def test_apply_talent_sizing_no_talent_cols_returns_none_options():
    """Return no talent options when no talent columns are detected."""
    table_text = [["Stat", "Value"], ["P1", "1"]]
    _, row_h, img_w, img_h, opts = r._apply_talent_sizing(
        [row[:] for row in table_text],
        col_widths=[10, 10],
        row_height=12,
        font_size=12,
        padding_x=4,
        padding_y=2,
        grid=1,
        title_h=0,
    )
    assert opts is None
    assert img_w > 0 and img_h > 0
    assert row_h == 12


def test_render_with_hero_portraits_and_talent_cols(tmp_path: Path):
    """Render with portrait images and talent columns enabled."""
    # Create tiny hero portrait files.
    p1 = tmp_path / "Hero1.png"
    p2 = tmp_path / "Hero2.png"
    Image.new("RGBA", (64, 64), (0, 255, 0, 255)).save(p1, format="PNG")
    Image.new("RGBA", (64, 64), (0, 0, 255, 255)).save(p2, format="PNG")

    player_labels = ["P1", "P2"]
    rows = [
        ("Talent 1", ["A", "B"]),
        ("Kills", [1, 2]),
    ]

    png = r.render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        title=None,
        player_hero_names=["Hero1", "Hero2"],
        player_hero_portrait_paths=[p1, p2],
    )

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size[0] > 0 and img.size[1] > 0


def test_compute_best_value_cells_highlights_best_across_all_players():
    """Highlight best values across all players (no per-team split)."""
    table_text, table_is_numeric = r._build_table_text(
        ["B1", "B2", "R1", "R2"],
        rows=[
            ("Kills", [1, 9, 3, 4]),
            ("Deaths", [2, 1, 9, 0]),
        ],
    )

    ctx = r._TableDrawContext(
        col_widths=[10] * len(table_text[0]),
        row_height=12,
        start_x=0,
        start_y=0,
        padding_x=1,
        padding_y=1,
        grid=1,
    )

    cells = r._compute_best_value_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)

    # Global best Kills is 9 (B2 -> r=2, c=1)
    assert (2, 1) in cells
    # Global best Deaths is 0 (R2 -> r=4, c=2)
    assert (4, 2) in cells


def test_compute_best_value_cells_suppresses_highlight_when_three_or_more_ties():
    """Suppress highlights when >=3 players share the best value."""
    table_text, table_is_numeric = r._build_table_text(
        ["P1", "P2", "P3", "P4"],
        rows=[
            ("Kills", [5, 5, 5, 1]),
        ],
    )

    ctx = r._TableDrawContext(
        col_widths=[10] * len(table_text[0]),
        row_height=12,
        start_x=0,
        start_y=0,
        padding_x=1,
        padding_y=1,
        grid=1,
    )

    cells = r._compute_best_value_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)
    assert not cells


def test_draw_team_cell_backgrounds_writes_pixels_for_team_and_highlight():
    """Draw team backgrounds and gold highlight overlays into the overlay image."""
    table_text, table_is_numeric = r._build_table_text(
        ["B1", "R1"],
        rows=[
            ("Kills", [1, 2]),
        ],
    )
    ctx = r._TableDrawContext(
        col_widths=[60, 60],
        row_height=30,
        start_x=0,
        start_y=0,
        padding_x=2,
        padding_y=2,
        grid=1,
    )

    cells = r._compute_best_value_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)

    overlay = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r._draw_team_cell_backgrounds(draw, table_text=table_text, ctx=ctx, highlight_cells=cells)

    # Sample a pixel inside the blue team's first data cell (col 1, row 1).
    px = overlay.getpixel((70, 40))
    assert isinstance(px, tuple) and px[3] > 0

    # Sample a pixel where the gold inner glow should touch (edge area) for the best cell.
    # Best cell is (r=2, c=1) in this table.
    px2 = overlay.getpixel((63, 64))
    assert isinstance(px2, tuple) and px2[3] > 0
    assert abs(int(px2[0]) - 255) < 60 and abs(int(px2[1]) - 215) < 60


def test_render_with_missing_hero_portrait_paths_is_resilient(tmp_path: Path):
    """Render successfully when portrait paths are missing or None."""
    player_labels = ["P1", "P2"]
    rows = [("Kills", [1, 2])]

    png = r.render_stats_store_to_png(
        player_labels=player_labels,
        rows=rows,
        player_hero_names=["Hero1", "Hero2"],
        player_hero_portrait_paths=[tmp_path / "missing.png", None],
    )

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
