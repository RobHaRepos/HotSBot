from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

CellValue = int | float | str | None
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

DEFAULT_TEAM_BLUE_NAME = "TEAM BLUE"
DEFAULT_TEAM_RED_NAME = "TEAM RED"

TEAM_BLUE_RGB = (80, 160, 255)
TEAM_RED_RGB = (255, 110, 110)

TEAM_BLUE_CELL_RGBA = (*TEAM_BLUE_RGB, 70)
TEAM_RED_CELL_RGBA = (*TEAM_RED_RGB, 70)
HIGHLIGHT_GOLD_RGB = (255, 215, 0)

BACKGROUND_IMAGE_RELATIVE_PATH = Path("data") / "images" / "HotS_background.jpg"

HIGHLIGHT_KEYWORDS = (
        "kill",
        "assist",
        "death",
        "damage",
        "healing",
        "protection",
        "experience",
        "time ",
        "streak",
        "multikill",
        "escape",
        "vengeance",
        "heals",
        "dead"
    )

@dataclass(frozen=True)
class RenderLayoutOptions:
    font_size: int = 25
    padding_x: int = 14
    padding_y: int = 10
    grid: int = 1


@dataclass(frozen=True)
class TeamHeaderOptions:
    team_blue_name: str = DEFAULT_TEAM_BLUE_NAME
    team_red_name: str = DEFAULT_TEAM_RED_NAME
    team_blue_level: int | None = None
    team_red_level: int | None = None
    team_blue_kills: int | None = None
    team_red_kills: int | None = None
    game_time_seconds: int | None = None


@dataclass(frozen=True)
class TalentRenderOptions:
    cols: set[int]
    font: Font
    font_bold: Font


@dataclass(frozen=True)
class _TableDrawContext:
    col_widths: Sequence[int]
    row_height: int
    start_x: int
    start_y: int
    padding_x: int
    padding_y: int
    grid: int


@dataclass(frozen=True)
class _PlayerCellRenderOptions:
    hero_names: Sequence[str] | None
    hero_portraits: Sequence[Image.Image | None] | None

@dataclass(frozen=True)
class TeamHeaderLayout:
    team_blue_name: str
    team_red_name: str
    team_blue_level: int | None
    team_red_level: int | None
    team_blue_kills: int | None
    team_red_kills: int | None
    game_time_seconds: int | None
    title_height: int
    header_top: int
    left_x: int
    right_x: int
    center_x: int
    blue_text_left_x: int
    red_text_left_x: int
    blue_kills_left_x: int
    red_kills_left_x: int
    name_top_y: int
    level_top_y: int
    game_label_center_y: int
    time_center_y: int
    team_font_bold: Font
    level_font_bold: Font
    game_label_font_bold: Font
    time_font_bold: Font
    team_font_size: int


def _format_cell(value: CellValue) -> str:
    """Format a table cell for display."""
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _is_numeric(value: object) -> bool:
    """Return True if value is a numeric scalar for alignment."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_int(value: object) -> int | None:
    """Convert value to int, returning None on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_int_or_zero(value: object) -> int:
    """Convert value to int, returning 0 on failure."""
    converted = _safe_int(value)
    return 0 if converted is None else converted


def _get_default_background_path() -> Path:
    """Return the default background image path (repo-relative)."""
    # This file lives at app/replay_parser/src/statistic_png_renderer.py
    # Repo root is 3 parents up.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / BACKGROUND_IMAGE_RELATIVE_PATH


def _build_background_image(img_w: int, img_h: int) -> Image.Image:
    """Create an RGB base image using the HotS background (fallback to white)."""
    bg_path = _get_default_background_path()
    try:
        bg = Image.open(bg_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (img_w, img_h), "white")

    src_w, src_h = bg.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (img_w, img_h), "white")

    # Scale-to-cover then center-crop
    scale = max(img_w / src_w, img_h / src_h)
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))
    bg = bg.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)

    left = max(0, (resized_w - img_w) // 2)
    top = max(0, (resized_h - img_h) // 2)
    return bg.crop((left, top, left + img_w, top + img_h))


def _is_talent_header(cell_text: str) -> bool:
    """Return True if a header cell denotes a Talent tier/slot."""
    s = cell_text.strip().lower()
    if not s:
        return False
    # Handles: Talent 1, Talent\n1, Tier1Talent, etc.
    return "talent" in s


def _detect_talent_cols(table_text: Sequence[Sequence[str]]) -> set[int]:
    """Return column indices that correspond to talent columns."""
    if not table_text:
        return set()
    return {i for i, s in enumerate(table_text[0]) if _is_talent_header(s)}


def _recompute_img_dimensions(
    *,
    table_text: Sequence[Sequence[str]],
    col_widths: Sequence[int],
    row_height: int,
    grid: int,
    padding_y: int,
    title_h: int,
) -> tuple[int, int]:
    """Compute image width/height for the table based on current geometry."""
    if not table_text:
        return 0, 0
    col_count = len(table_text[0])
    row_count = len(table_text)
    img_w = sum(col_widths) + grid * (col_count + 1)

    bottom_extra = max(padding_y * 6, row_height // 3, 8)
    img_h = title_h + row_height * row_count + grid * (row_count + 1) + padding_y + bottom_extra
    required_h = title_h + row_count * (row_height + grid) + padding_y * 2 + 2
    img_h = max(img_h, required_h)
    return img_w, img_h


def _max_text_height_in_cols(
    *,
    draw: ImageDraw.ImageDraw,
    table_text: Sequence[Sequence[str]],
    cols: set[int],
    font: Font,
    font_bold: Font,
) -> int:
    """Return the maximum multiline text height across the given columns."""
    max_h = 0
    for r in range(len(table_text)):
        f = font_bold if r == 0 else font
        for c in cols:
            max_h = max(max_h, _text_h(draw, f, table_text[r][c]))
    return max_h


def _widen_columns_for_large_font(
    *,
    draw: ImageDraw.ImageDraw,
    table_text: Sequence[Sequence[str]],
    cols: set[int],
    col_widths: List[int],
    font: Font,
    font_bold: Font,
    padding_x: int,
) -> None:
    """Increase column widths so all text fits using the given fonts."""
    for c in cols:
        for r in range(len(table_text)):
            f = font_bold if r == 0 else font
            col_widths[c] = max(col_widths[c], _text_w(draw, f, table_text[r][c]) + padding_x * 2)


def _apply_talent_sizing(
    table_text: List[List[str]],
    *,
    col_widths: List[int],
    row_height: int,
    font_size: int,
    padding_x: int,
    padding_y: int,
    grid: int,
    title_h: int,
) -> tuple[List[int], int, int, int, TalentRenderOptions | None]:
    """Adjust widths/heights for Talent columns and return updated geometry + options."""
    talent_cols = _detect_talent_cols(table_text)
    if not talent_cols:
        img_w, img_h = _recompute_img_dimensions(
            table_text=table_text,
            col_widths=col_widths,
            row_height=row_height,
            grid=grid,
            padding_y=padding_y,
            title_h=title_h,
        )
        return col_widths, row_height, img_w, img_h, None

    talent_font, talent_font_bold = _load_fonts(max(font_size * 2, font_size + 1))
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)

    _widen_columns_for_large_font(
        draw=dummy_draw,
        table_text=table_text,
        cols=talent_cols,
        col_widths=col_widths,
        font=talent_font,
        font_bold=talent_font_bold,
        padding_x=padding_x,
    )

    max_talent_h = _max_text_height_in_cols(
        draw=dummy_draw,
        table_text=table_text,
        cols=talent_cols,
        font=talent_font,
        font_bold=talent_font_bold,
    )
    row_height = max(row_height, max_talent_h + padding_y * 2)

    img_w, img_h = _recompute_img_dimensions(
        table_text=table_text,
        col_widths=col_widths,
        row_height=row_height,
        grid=grid,
        padding_y=padding_y,
        title_h=title_h,
    )
    return col_widths, row_height, img_w, img_h, TalentRenderOptions(cols=talent_cols, font=talent_font, font_bold=talent_font_bold)

def _load_fonts(font_size: int) -> tuple[Font, Font]:
    """Load a font and bold font (TTF preferred, else Pillow default)."""
    candidates: list[tuple[str, str]] = [
        ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"),
        ("NotoSansCJKjp-Regular.otf", "NotoSansCJKjp-Bold.otf"),
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("arial.ttf", "arialbd.ttf"),
    ]
    font_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("C:/Windows/Fonts"),
    ]

    def _try_load(name_or_path: str, size: int) -> Font:
        """Load a truetype font by name or by searching known font dirs."""
        try:
            return ImageFont.truetype(name_or_path, size)
        except Exception:
            for d in font_dirs:
                p = d / name_or_path
                if p.exists():
                    return ImageFont.truetype(str(p), size)
            raise

    for regular_name, bold_name in candidates:
        try:
            font = _try_load(regular_name, font_size)
            try:
                font_bold = _try_load(bold_name, font_size)
            except Exception:
                font_bold = font
            return font, font_bold
        except Exception:
            continue

    # Last resort: pillow default bitmap font (fixed size)
    font = ImageFont.load_default()
    return font, font

def _build_table_text(
    player_labels: Sequence[str],
    rows: Sequence[Tuple[str, Sequence[CellValue]]]) -> tuple[List[List[str]], List[List[bool]]]:
    """Build table strings plus a numeric mask for alignment."""
    table_text: List[List[str]] = []
    table_is_numeric: List[List[bool]] = []
    
    categories = [category for category, _ in rows]
    header = ["Player", *categories]
    table_text.append(header)
    table_is_numeric.append([False] * len(header))

    # Number of players assumed to match the length of values in rows
    for i, player in enumerate(player_labels):
        row_vals = [player]
        row_mask = [False]
        for (_cat, values) in rows:
            v = values[i]
            row_vals.append(_format_cell(v))
            row_mask.append(_is_numeric(v))
        table_text.append(row_vals)
        table_is_numeric.append(row_mask)

    return table_text, table_is_numeric

def _text_w(draw: ImageDraw.ImageDraw, font: Font, s: str) -> int:
    """Measure multiline text width."""
    bbox = draw.multiline_textbbox((0, 0), s, font=font)
    return int(bbox[2] - bbox[0])

def _text_h(draw: ImageDraw.ImageDraw, font: Font, s: str = "Ag") -> int:
    """Measure multiline text height."""
    bbox = draw.multiline_textbbox((0, 0), s, font=font)
    return int(bbox[3] - bbox[1])

def _measure_layout(
    table_text: List[List[str]],
    *,
    font: Font,
    font_bold: Font,
    padding_x: int,
    padding_y: int,
    grid: int,
    title: str | None,
    header_max_width: int | None = None,
    header_extra: int = 0,
) -> tuple[List[int], int, int, int, int]:
    """Measure layout, optionally wrapping header cells."""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)

    if header_max_width is not None:
        _wrap_header_cells(table_text, draw=draw, font_bold=font_bold, header_max_width=header_max_width)

    col_count = len(table_text[0])
    row_count = len(table_text)

    col_widths = _compute_col_widths(table_text, draw=draw, font=font, font_bold=font_bold, padding_x=padding_x)
    row_height = _compute_row_height(table_text, draw=draw, font=font, font_bold=font_bold, padding_y=padding_y, header_extra=header_extra)
    title_h = (_text_h(draw, font_bold, title) + padding_y * 2 + header_extra) if title else 0

    img_w = sum(col_widths) + grid * (col_count + 1)
    bottom_extra = max(padding_y * 6, row_height // 3, 8)
    img_h = title_h + row_height * row_count + grid * (row_count + 1) + padding_y + bottom_extra

    required_h = title_h + row_count * (row_height + grid) + padding_y * 2 + 2
    img_h = max(img_h, required_h)
    return col_widths, row_height, title_h, img_w, img_h


def _wrap_header_cells(
    table_text: List[List[str]],
    *,
    draw: ImageDraw.ImageDraw,
    font_bold: Font,
    header_max_width: int,
) -> None:
    """Wrap header cells into two lines when too wide."""
    if not table_text:
        return

    header = table_text[0]
    for i, cell_text in enumerate(header):
        if " " not in cell_text:
            continue
        if _text_w(draw, font_bold, cell_text) <= header_max_width:
            continue

        words = cell_text.split()
        if len(words) < 2:
            continue

        best_split_index = 1
        best_diff: int | None = None
        for j in range(1, len(words)):
            left = " ".join(words[:j])
            right = " ".join(words[j:])
            diff = abs(len(left) - len(right))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_split_index = j

        header[i] = " ".join(words[:best_split_index]) + "\n" + " ".join(words[best_split_index:])


def _compute_col_widths(
    table_text: Sequence[Sequence[str]],
    *,
    draw: ImageDraw.ImageDraw,
    font: Font,
    font_bold: Font,
    padding_x: int,
) -> List[int]:
    """Compute column widths (including padding)."""
    col_count = len(table_text[0])
    row_count = len(table_text)
    widths: list[int] = []
    for c in range(col_count):
        max_w = 0
        for r in range(row_count):
            cell_text = table_text[r][c]
            f = font_bold if r == 0 else font
            max_w = max(max_w, _text_w(draw, f, cell_text))
        widths.append(max_w + padding_x * 2)
    return widths


def _compute_row_height(
    table_text: Sequence[Sequence[str]],
    *,
    draw: ImageDraw.ImageDraw,
    font: Font,
    font_bold: Font,
    padding_y: int,
    header_extra: int,
) -> int:
    """Compute row height based on tallest cell and header height."""
    col_count = len(table_text[0])
    row_count = len(table_text)

    max_row_text_h = 0
    for r in range(row_count):
        f = font_bold if r == 0 else font
        max_row_text_h = max(max_row_text_h, max(_text_h(draw, f, table_text[r][c]) for c in range(col_count)))

    header_text_h = 0
    if row_count > 0:
        header_text_h = max(_text_h(draw, font_bold, cell_text) for cell_text in table_text[0])

    header_h = header_text_h + padding_y * 2 + header_extra
    return max(max_row_text_h + padding_y * 2, header_h)


def _auto_header_wrap_threshold(col_widths: Sequence[int]) -> int:
    """Compute a sensible header wrap threshold."""
    if not col_widths:
        return 150
    return max(150, int(max(col_widths) * 0.75))


def _maybe_wrap_headers(
    table_text: List[List[str]],
    *,
    font_bold: Font,
    header_max_width: int,
) -> bool:
    """Wrap header cells in-place if any exceed max width."""
    if not table_text:
        return False
    header = table_text[0]
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    needs_wrap = any(_text_w(draw, font_bold, s) > header_max_width for s in header)
    if not needs_wrap:
        return False
    _wrap_header_cells(table_text, draw=draw, font_bold=font_bold, header_max_width=header_max_width)
    return True


def _has_team_header(team_header: TeamHeaderOptions | None) -> bool:
    """Return True if a team header is requested."""
    return team_header is not None


def _derive_team_kills(
    *,
    existing_blue_kills: int | None,
    existing_red_kills: int | None,
    rows: Sequence[Tuple[str, Sequence[CellValue]]],
    player_count: int,
) -> tuple[int | None, int | None]:
    """Derive team kills from rows when not explicitly supplied."""
    if existing_blue_kills is not None and existing_red_kills is not None:
        return existing_blue_kills, existing_red_kills

    team_deaths_vals: Sequence[CellValue] | None = None
    for category, values in rows:
        category_lower = category.lower()
        if "team" in category_lower and ("death" in category_lower or "takedown" in category_lower):
            team_deaths_vals = values
            break

    if team_deaths_vals is None:
        return existing_blue_kills, existing_red_kills

    values_int = [_safe_int_or_zero(v) for v in team_deaths_vals]
    if len(values_int) >= 6:
        red_kills = existing_red_kills if existing_red_kills is not None else _safe_int(values_int[0])
        blue_kills = existing_blue_kills if existing_blue_kills is not None else _safe_int(values_int[5])
        return blue_kills, red_kills

    padded = values_int + [0] * max(0, player_count - len(values_int))
    half = player_count // 2
    blue_kills = existing_blue_kills if existing_blue_kills is not None else sum(padded[:half])
    red_kills = existing_red_kills if existing_red_kills is not None else sum(padded[half:])
    return blue_kills, red_kills


def _derive_team_levels(
    *,
    existing_blue_level: int | None,
    existing_red_level: int | None,
    rows: Sequence[Tuple[str, Sequence[CellValue]]],
    player_count: int,
) -> tuple[int | None, int | None]:
    """Derive team levels from the "Team Level Achieved"/"TeamLevel" row when not explicitly supplied."""
    if existing_blue_level is not None and existing_red_level is not None:
        return existing_blue_level, existing_red_level

    level_vals: Sequence[CellValue] | None = None
    for category, values in rows:
        category_lower = category.lower()
        if "team" in category_lower and "level" in category_lower:
            level_vals = values
            break

    if level_vals is None:
        return existing_blue_level, existing_red_level

    values_int = [_safe_int_or_zero(v) for v in level_vals]
    padded = values_int + [0] * max(0, player_count - len(values_int))
    half = player_count // 2
    blue = existing_blue_level if existing_blue_level is not None else max(padded[:half] or [0])
    red = existing_red_level if existing_red_level is not None else max(padded[half:] or [0])

    return (_safe_int(blue), _safe_int(red))


def _build_team_header_layout(
    *,
    draw: ImageDraw.ImageDraw,
    img_w: int,
    font_size: int,
    padding_y: int,
    team_header: TeamHeaderOptions,
) -> TeamHeaderLayout:
    """Compute team header layout geometry and fonts."""
    team_font_size = font_size * 3
    _team_font, team_font_bold = _load_fonts(team_font_size)

    _base_font, _ = _load_fonts(font_size)

    level_font_size = max(int(font_size * 1.35), font_size + 6)
    _level_font, level_font_bold = _load_fonts(level_font_size)

    game_label_font_size = max(int(team_font_size * 0.6), font_size + 1)
    _game_label_font, game_label_font_bold = _load_fonts(game_label_font_size)

    time_font_size = max(game_label_font_size - 6, max(font_size - 2, 10))
    _time_font, time_font_bold = _load_fonts(time_font_size)

    name_h = _text_h(draw, team_font_bold, team_header.team_blue_name or "TEAM")
    level_h = _text_h(draw, level_font_bold, "Level 00")
    game_label_h = _text_h(draw, game_label_font_bold, "Game Time")
    game_time_h = _text_h(draw, time_font_bold, "00:00")

    header_top = padding_y + max(8, font_size // 4)
    gap_name_level = max(16, font_size)

    name_top_y = header_top
    level_top_y = name_top_y + name_h + gap_name_level

    left_block_h = (level_top_y - name_top_y) + level_h
    center_block_h = max(name_h, game_label_h + max(18, font_size) + game_time_h)
    title_height = header_top + max(left_block_h, center_block_h) + padding_y * 2

    center_x = img_w // 2
    sep = max(48, img_w // 8)
    left_x = center_x - sep
    right_x = center_x + sep

    blue_name_w = _text_w(draw, team_font_bold, team_header.team_blue_name or "TEAM")
    red_name_w = _text_w(draw, team_font_bold, team_header.team_red_name or "TEAM")
    blue_text_left_x = int(left_x - (blue_name_w // 2))
    red_text_left_x = int(right_x - (red_name_w // 2))

    kill_gap = max(22, team_font_size // 5)
    blue_kills_w = _text_w(draw, team_font_bold, "00")
    red_kills_w = blue_kills_w
    if team_header.team_blue_kills is not None:
        blue_kills_w = _text_w(draw, team_font_bold, str(team_header.team_blue_kills))
    if team_header.team_red_kills is not None:
        red_kills_w = _text_w(draw, team_font_bold, str(team_header.team_red_kills))

    # Place kills next to team name, staying within each half to avoid overlapping the center Game Time block.
    half_gap = max(10, padding_y)
    blue_kills_left_x = blue_text_left_x + blue_name_w + kill_gap
    blue_kills_left_x = max(0, min(blue_kills_left_x, center_x - half_gap - blue_kills_w))

    red_kills_left_x = red_text_left_x - kill_gap - red_kills_w
    red_kills_left_x = max(center_x + half_gap, min(red_kills_left_x, img_w - red_kills_w))

    game_label_center_y = header_top + (game_label_h // 2)
    game_time_gap = max(34, font_size + 8)
    time_center_y = game_label_center_y + (game_label_h // 2) + game_time_gap

    return TeamHeaderLayout(
        team_blue_name=team_header.team_blue_name,
        team_red_name=team_header.team_red_name,
        team_blue_level=team_header.team_blue_level,
        team_red_level=team_header.team_red_level,
        team_blue_kills=team_header.team_blue_kills,
        team_red_kills=team_header.team_red_kills,
        game_time_seconds=team_header.game_time_seconds,
        title_height=title_height,
        header_top=header_top,
        left_x=left_x,
        right_x=right_x,
        center_x=center_x,
        blue_text_left_x=blue_text_left_x,
        red_text_left_x=red_text_left_x,
        blue_kills_left_x=blue_kills_left_x,
        red_kills_left_x=red_kills_left_x,
        name_top_y=name_top_y,
        level_top_y=level_top_y,
        game_label_center_y=game_label_center_y,
        time_center_y=time_center_y,
        team_font_bold=team_font_bold,
        level_font_bold=level_font_bold,
        game_label_font_bold=game_label_font_bold,
        time_font_bold=time_font_bold,
        team_font_size=team_font_size,
    )


def _format_game_time(game_time_seconds: int | None) -> str:
    """Format game time as MM:SS."""
    if game_time_seconds is None or game_time_seconds < 0:
        return "00:00"
    mins = game_time_seconds // 60
    secs = game_time_seconds % 60
    return f"{mins:02d}:{secs:02d}"


def _draw_team_header(
    *,
    img: Image.Image,
    layout: TeamHeaderLayout,
) -> None:
    """Draw the team header area."""
    draw = ImageDraw.Draw(img)

    draw.multiline_text(
        (layout.blue_text_left_x, layout.name_top_y),
        layout.team_blue_name,
        font=layout.team_font_bold,
        anchor="lt",
        fill=TEAM_BLUE_RGB,
        stroke_width=1,
        stroke_fill="black",
    )
    draw.multiline_text(
        (layout.red_text_left_x, layout.name_top_y),
        layout.team_red_name,
        font=layout.team_font_bold,
        anchor="lt",
        fill=TEAM_RED_RGB,
        stroke_width=1,
        stroke_fill="black",
    )

    if layout.team_blue_kills is not None:
        draw.text(
            (layout.blue_kills_left_x, layout.name_top_y),
            str(layout.team_blue_kills),
            font=layout.team_font_bold,
            anchor="lt",
            fill=TEAM_BLUE_RGB,
            stroke_width=1,
            stroke_fill="black",
        )
    if layout.team_red_kills is not None:
        draw.text(
            (layout.red_kills_left_x, layout.name_top_y),
            str(layout.team_red_kills),
            font=layout.team_font_bold,
            anchor="lt",
            fill=TEAM_RED_RGB,
            stroke_width=1,
            stroke_fill="black",
        )

    if layout.team_blue_level is not None:
        level_text = f"Level {layout.team_blue_level}"
        if layout.team_blue_kills is not None:
            kills_w = _text_w(draw, layout.team_font_bold, str(layout.team_blue_kills))
            kills_right_x = layout.blue_kills_left_x + kills_w
            level_w = _text_w(draw, layout.level_font_bold, level_text)
            level_x = max(0, int(kills_right_x - level_w))
        else:
            level_x = layout.blue_text_left_x
        draw.text(
            (level_x, layout.level_top_y),
            level_text,
            font=layout.level_font_bold,
            anchor="lt",
            fill=TEAM_BLUE_RGB,
            stroke_width=1,
            stroke_fill="black",
        )
    if layout.team_red_level is not None:
        if layout.team_red_kills is not None:
            level_x = layout.red_kills_left_x
        else:
            level_x = layout.red_text_left_x
        draw.text(
            (level_x, layout.level_top_y),
            f"Level {layout.team_red_level}",
            font=layout.level_font_bold,
            anchor="lt",
            fill=TEAM_RED_RGB,
            stroke_width=1,
            stroke_fill="black",
        )

    # Kills are rendered as big numbers on the TEAM line.

    draw.text(
        (layout.center_x, layout.game_label_center_y),
        "Game Time",
        font=layout.game_label_font_bold,
        anchor="mm",
        fill="white",
        stroke_width=1,
        stroke_fill="black",
    )
    draw.text(
        (layout.center_x, layout.time_center_y),
        _format_game_time(layout.game_time_seconds),
        font=layout.time_font_bold,
        anchor="mm",
        fill="white",
        stroke_width=1,
        stroke_fill="black",
    )


def _draw_table(
    *,
    img: Image.Image,
    table_text: List[List[str]],
    table_is_numeric: List[List[bool]],
    ctx: _TableDrawContext,
    font: Font,
    font_bold: Font,
    talent: TalentRenderOptions | None,
    player: _PlayerCellRenderOptions,
) -> None:
    """Draw the table onto the given image."""
    comp = _draw_row_bands(
        img,
        start_x=ctx.start_x,
        start_y=ctx.start_y,
        row_height=ctx.row_height,
        grid=ctx.grid,
        row_count=len(table_text),
        col_widths=ctx.col_widths,
    )

    highlight_cells = _compute_best_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)
    comp_rgba = comp.convert("RGBA")
    overlay = Image.new("RGBA", comp_rgba.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    _draw_team_cell_backgrounds(
        od,
        table_text=table_text,
        ctx=ctx,
        highlight_cells=highlight_cells,
    )
    comp = Image.alpha_composite(comp_rgba, overlay).convert("RGB")

    draw = ImageDraw.Draw(comp)
    _draw_grid(
        draw,
        start_x=ctx.start_x,
        start_y=ctx.start_y,
        col_widths=ctx.col_widths,
        row_count=len(table_text),
        row_height=ctx.row_height,
        grid=ctx.grid,
    )
    _draw_cell_text(
        draw,
        img=comp,
        table_text=table_text,
        table_is_numeric=table_is_numeric,
        ctx=ctx,
        font=font,
        font_bold=font_bold,
        talent=talent,
        player=player,
    )
    img.paste(comp)


def _parse_number(text: str) -> float | None:
    """Parse a formatted numeric cell back into a number."""
    s = text.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(int(s))
    except Exception:
        try:
            return float(s)
        except Exception:
            return None


def _is_highlightable_stat(label: str) -> bool:
    """Return True if a stat label is reasonable to highlight."""
    s = label.strip().lower()

    return any(k in s for k in HIGHLIGHT_KEYWORDS)


def _prefers_lower_value(label: str) -> bool:
    """Return True if lower values are better for this stat."""
    s = label.strip().lower()
    return any(k in s for k in ("death", "time spent dead", "time dead", "damage taken"))


def _team_split_count(player_count: int) -> int:
    """Return the blue team size for a given player count."""
    return min(max(1, player_count // 2), player_count)


def _best_value(values: Sequence[float], *, prefer_low: bool) -> float:
    """Return the best value from a list under the chosen direction."""
    best_value = min(values) if prefer_low else max(values)
    if values.count(best_value) < 3:
        return best_value
    else:
        return -1


def _collect_numeric_values_by_row(
    *,
    col_index: int,
    rows: range,
    table_text: Sequence[Sequence[str]],
    table_is_numeric: Sequence[Sequence[bool]],
    row_count: int,
) -> list[tuple[float, int]]:
    """Collect parsed numeric values for a column, along with their row index."""
    values: list[tuple[float, int]] = []
    for r in rows:
        if r >= row_count or r >= len(table_is_numeric) or col_index >= len(table_is_numeric[r]):
            continue
        if not table_is_numeric[r][col_index]:
            continue
        v = _parse_number(table_text[r][col_index])
        if v is None:
            continue
        values.append((v, r))
    return values


def _highlight_best_positions(values: Sequence[tuple[float, int]], *, prefer_low: bool) -> set[int]:
    """Return indices of positions whose value equals the best in the group."""
    if not values:
        return set()
    best = _best_value([v for v, _ in values], prefer_low=prefer_low)
    return {pos for v, pos in values if v == best}

def _best_rows_in_col(
    *,
    col_index: int,
    rows: range,
    table_text: Sequence[Sequence[str]],
    table_is_numeric: Sequence[Sequence[bool]],
    row_count: int,
    prefer_low: bool,
) -> set[int]:
    """Return row indices of best values for one column and team range."""
    values = _collect_numeric_values_by_row(
        col_index=col_index,
        rows=rows,
        table_text=table_text,
        table_is_numeric=table_is_numeric,
        row_count=row_count,
    )
    return _highlight_best_positions(values, prefer_low=prefer_low)


def _compute_best_value_cells(
    *,
    table_text: Sequence[Sequence[str]],
    table_is_numeric: Sequence[Sequence[bool]],
    ctx: _TableDrawContext,
) -> set[tuple[int, int]]:
    """Compute best-value cells for the transposed layout."""
    col_count = len(ctx.col_widths)
    row_count = len(table_text)
    player_count = max(0, row_count - 1)
    player_rows = range(1, 1 + player_count)

    highlighted: set[tuple[int, int]] = set()
    for c in range(1, col_count):
        label = table_text[0][c] if c < len(table_text[0]) else ""
        if not _is_highlightable_stat(label):
            continue
        prefer_low = _prefers_lower_value(label)

        for r in _best_rows_in_col(
            col_index=c,
            rows=player_rows,
            table_text=table_text,
            table_is_numeric=table_is_numeric,
            row_count=row_count,
            prefer_low=prefer_low,
        ):
            highlighted.add((r, c))

    return highlighted


def _compute_best_cells(
    *,
    table_text: Sequence[Sequence[str]],
    table_is_numeric: Sequence[Sequence[bool]],
    ctx: _TableDrawContext,
) -> set[tuple[int, int]]:
    """Return (r, c) cells to highlight as best-in-team."""
    col_count = len(ctx.col_widths)
    row_count = len(table_text)
    if row_count == 0 or col_count == 0:
        return set()

    return _compute_best_value_cells(table_text=table_text, table_is_numeric=table_is_numeric, ctx=ctx)
    

def _cell_team_fill(*, r: int, c: int, row_count: int) -> tuple[int, int, int, int] | None:
    """Return the RGBA fill for a team cell, or None if not applicable."""
    if r <= 0 or c <= 0:
        return None

    player_count = max(0, row_count - 1)
    blue_end_r = 1 + _team_split_count(player_count)
    return TEAM_BLUE_CELL_RGBA if r < blue_end_r else TEAM_RED_CELL_RGBA


def _draw_inner_glow(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    rgb: tuple[int, int, int],
    max_alpha: int = 140,
    steps: int = 7,
) -> None:
    """Draw a soft inner border using inset rectangles."""
    x0, y0, x1, y1 = bbox
    for i in range(max(1, steps)):
        a = int(max_alpha * (1.0 - (i / max(1, steps - 1))))
        if a <= 0:
            continue
        draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i), outline=(*rgb, a), width=1)


def _draw_team_cell_backgrounds(
    draw: ImageDraw.ImageDraw,
    *,
    table_text: Sequence[Sequence[str]],
    ctx: _TableDrawContext,
    highlight_cells: set[tuple[int, int]],
) -> None:
    """Fill team cells and highlight best values."""
    col_count = len(ctx.col_widths)
    row_count = len(table_text)
    if row_count == 0 or col_count == 0:
        return

    y = ctx.start_y
    for r in range(row_count):
        _draw_team_cell_background_row(
            draw,
            r=r,
            y=y,
            ctx=ctx,
            highlight_cells=highlight_cells,
            col_count=col_count,
            row_count=row_count,
        )
        y += ctx.row_height + ctx.grid


def _draw_team_cell_background_row(
    draw: ImageDraw.ImageDraw,
    *,
    r: int,
    y: int,
    ctx: _TableDrawContext,
    highlight_cells: set[tuple[int, int]],
    col_count: int,
    row_count: int,
) -> None:
    """Draw team cell backgrounds for a single row."""
    if r <= 0 or col_count <= 1:
        return

    x = ctx.start_x + ctx.col_widths[0] + ctx.grid
    for c in range(1, col_count):
        cell_w = ctx.col_widths[c]
        cell_x0 = x + ctx.grid
        cell_y0 = y + ctx.grid
        cell_x1 = cell_x0 + cell_w - 1
        cell_y1 = cell_y0 + ctx.row_height - 1

        fill = _cell_team_fill(r=r, c=c, row_count=row_count)
        if fill is not None:
            draw.rectangle((cell_x0, cell_y0, cell_x1, cell_y1), fill=fill)

        if (r, c) in highlight_cells:
            inset = max(1, ctx.grid)
            _draw_inner_glow(
                draw,
                bbox=(cell_x0 + inset, cell_y0 + inset, cell_x1 - inset, cell_y1 - inset),
                rgb=HIGHLIGHT_GOLD_RGB,
            )

        x += cell_w + ctx.grid


def _font_for_cell(*, r: int, c: int, font: Font, font_bold: Font, talent: TalentRenderOptions | None) -> Font:
    """Return the font to use for a specific table cell."""
    if talent is not None and c in talent.cols:
        return talent.font_bold if r == 0 else talent.font
    return font_bold if r == 0 else font


def _draw_row_bands(
    img: Image.Image,
    *,
    start_x: int,
    start_y: int,
    row_height: int,
    grid: int,
    row_count: int,
    col_widths: Sequence[int],
) -> Image.Image:
    """Draw row band backgrounds and return a composited RGB image."""
    # Lower alpha => more transparent.
    header_rgba = (109, 62, 181, 90)
    first_rows_rgba = (23, 78, 166, 90)
    last_rows_rgba = (178, 34, 34, 90)

    table_w = int(sum(col_widths) + grid * (len(col_widths) + 1))

    total_data_rows = max(0, row_count - 1)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    def band_bbox(top_row: int, bottom_row: int) -> tuple[int, int, int, int]:
        """Return the pixel bbox for a row band inclusive of the inner grid."""
        x0 = start_x + grid
        x1 = start_x + table_w - grid
        y0 = start_y + top_row * (row_height + grid) + grid
        y1 = start_y + bottom_row * (row_height + grid) + row_height + grid - 1
        return (x0, y0, x1, y1)

    if row_count > 0:
        od.rectangle(band_bbox(0, 0), fill=header_rgba)
    if total_data_rows > 0:
        n = min(5, total_data_rows)
        od.rectangle(band_bbox(1, 1 + n - 1), fill=first_rows_rgba)
        start = 1 + (total_data_rows - n)
        od.rectangle(band_bbox(start, start + n - 1), fill=last_rows_rgba)

    base_rgba = img.convert("RGBA")
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    start_x: int,
    start_y: int,
    col_widths: Sequence[int],
    row_count: int,
    row_height: int,
    grid: int,
) -> None:
    """Draw grid lines."""
    grid_rgb = (0, 0, 0)
    col_count = len(col_widths)

    outer = max(grid + 1, grid * 2)

    table_w = int(sum(col_widths) + grid * (col_count + 1))

    x = start_x
    for c in range(col_count):
        cell_w = col_widths[c]
        width = outer if c == 0 else grid
        draw.line([(x, start_y), (x, start_y + row_count * (row_height + grid) + grid)], fill=grid_rgb, width=width)
        x += cell_w + grid
    draw.line([(x, start_y), (x, start_y + row_count * (row_height + grid) + grid)], fill=grid_rgb, width=outer)

    y = start_y
    for r in range(row_count):
        width = outer if r == 0 else grid
        draw.line([(start_x, y), (start_x + table_w, y)], fill=grid_rgb, width=width)
        y += row_height + grid
    draw.line([(start_x, y), (start_x + table_w, y)], fill=grid_rgb, width=outer)


def _draw_cell_text(
    draw: ImageDraw.ImageDraw,
    *,
    img: Image.Image,
    table_text: Sequence[Sequence[str]],
    table_is_numeric: Sequence[Sequence[bool]],
    ctx: _TableDrawContext,
    font: Font,
    font_bold: Font,
    talent: TalentRenderOptions | None,
    player: _PlayerCellRenderOptions,
) -> None:
    """Draw cell text."""
    _ = table_is_numeric
    col_count = len(ctx.col_widths)
    row_count = len(table_text)

    y = ctx.start_y
    for r in range(row_count):
        x = ctx.start_x
        for c in range(col_count):
            cell_w = ctx.col_widths[c]
            cell_x0 = x + ctx.grid
            cell_y0 = y + ctx.grid

            if not _maybe_draw_player_cell(
                draw=draw,
                img=img,
                r=r,
                c=c,
                ctx=ctx,
                table_text=table_text,
                cell_x0=cell_x0,
                cell_y0=cell_y0,
                cell_w=cell_w,
                font=font,
                font_bold=font_bold,
                player=player,
            ):
                _draw_default_cell_text(
                    draw=draw,
                    r=r,
                    c=c,
                    cell_x0=cell_x0,
                    cell_y0=cell_y0,
                    cell_w=cell_w,
                    row_h=ctx.row_height,
                    text=table_text[r][c],
                    font=font,
                    font_bold=font_bold,
                    talent=talent,
                )

            x += cell_w + ctx.grid
        y += ctx.row_height + ctx.grid


def _draw_default_cell_text(
    *,
    draw: ImageDraw.ImageDraw,
    r: int,
    c: int,
    cell_x0: int,
    cell_y0: int,
    cell_w: int,
    row_h: int,
    text: str,
    font: Font,
    font_bold: Font,
    talent: TalentRenderOptions | None,
) -> None:
    """Draw centered cell text (bold header row)."""
    f = _font_for_cell(r=r, c=c, font=font, font_bold=font_bold, talent=talent)

    text_x = cell_x0 + (cell_w // 2)
    text_y = cell_y0 + (row_h // 2)
    stroke_w = 2 if r == 0 else 1
    draw.multiline_text(
        (text_x, text_y),
        text,
        fill="white",
        font=f,
        anchor="mm",
        stroke_width=stroke_w,
        stroke_fill="black",
    )


def _maybe_draw_player_cell(
    *,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    r: int,
    c: int,
    ctx: _TableDrawContext,
    table_text: Sequence[Sequence[str]],
    cell_x0: int,
    cell_y0: int,
    cell_w: int,
    font: Font,
    font_bold: Font,
    player: _PlayerCellRenderOptions,
) -> bool:
    """Draw the Player column cell (name/hero/portrait) when enabled."""
    if not (c == 0 and r > 0 and player.hero_names is not None):
        return False

    player_index = r - 1
    player_name = table_text[r][c]
    hero_name = player.hero_names[player_index] if player_index < len(player.hero_names) else ""

    portrait = None
    if player.hero_portraits is not None and player_index < len(player.hero_portraits):
        portrait = player.hero_portraits[player_index]

    _draw_player_cell(
        draw=draw,
        img=img,
        cell_x0=cell_x0,
        cell_y0=cell_y0,
        cell_w=cell_w,
        row_h=ctx.row_height,
        padding_x=ctx.padding_x,
        padding_y=ctx.padding_y,
        player_name=player_name,
        hero_name=hero_name,
        portrait=portrait,
        font=font,
        font_bold=font_bold,
    )
    return True


def _ellipsize(draw: ImageDraw.ImageDraw, *, font: Font, text: str, max_width: int) -> str:
    """Truncate text to fit in max_width by adding an ellipsis."""
    if max_width <= 0:
        return ""
    if _text_w(draw, font, text) <= max_width:
        return text
    ell = "…"
    if _text_w(draw, font, ell) > max_width:
        return ""
    s = text
    while s and _text_w(draw, font, s + ell) > max_width:
        s = s[:-1]
    return s + ell if s else ""


def _resize_square_image(img: Image.Image, *, size: int) -> Image.Image | None:
    """Resize image to a square, returning None if it fails."""
    if size <= 0:
        return None
    try:
        return img.resize((size, size), resample=Image.Resampling.LANCZOS)
    except Exception:
        try:
            return img.resize((size, size))
        except Exception:
            return None


def _prepare_player_portrait(
    portrait: Image.Image, *, inner_h: int, inner_w: int, padding_y: int
) -> tuple[Image.Image | None, int]:
    """Return a resized portrait and its square size."""
    max_h = max(1, inner_h - padding_y * 2)
    max_w = max(1, inner_w // 2)
    img_size = max(1, min(max_h, max_w))

    portrait_resized = _resize_square_image(portrait, size=img_size)
    if portrait_resized is None:
        return None, 0
    return portrait_resized, img_size


def _paste_portrait_and_get_text_left(
    *,
    img: Image.Image,
    portrait_resized: Image.Image,
    cell_x0: int,
    cell_y0: int,
    inner_h: int,
    img_size: int,
    padding_x: int,
) -> int:
    """Paste portrait if possible and return text left x-coordinate."""
    img_x = cell_x0 + padding_x
    img_y = cell_y0 + (inner_h - img_size) // 2
    try:
        img.paste(portrait_resized, (img_x, img_y), portrait_resized)
    except Exception:
        pass
    return img_x + img_size + padding_x


def _draw_player_cell(
    *,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    cell_x0: int,
    cell_y0: int,
    cell_w: int,
    row_h: int,
    padding_x: int,
    padding_y: int,
    player_name: str,
    hero_name: str,
    portrait: Image.Image | None,
    font: Font,
    font_bold: Font,
) -> None:
    """Draw the Player cell with optional hero portrait and two-line text."""
    inner_h = max(1, row_h)
    inner_w = max(1, cell_w)

    text_left_x = cell_x0 + padding_x
    if portrait is not None:
        portrait_resized, img_size = _prepare_player_portrait(
            portrait, inner_h=inner_h, inner_w=inner_w, padding_y=padding_y
        )
        if portrait_resized is not None and img_size > 0:
            text_left_x = _paste_portrait_and_get_text_left(
                img=img,
                portrait_resized=portrait_resized,
                cell_x0=cell_x0,
                cell_y0=cell_y0,
                inner_h=inner_h,
                img_size=img_size,
                padding_x=padding_x,
            )

    text_right_x = cell_x0 + inner_w - padding_x
    available_w = max(1, text_right_x - text_left_x)

    player_s = _ellipsize(draw, font=font_bold, text=player_name, max_width=available_w)
    hero_s = _ellipsize(draw, font=font, text=hero_name, max_width=available_w)

    # Slightly more separation between the two lines.
    line_gap = max(4, padding_y // 2)
    player_h = _text_h(draw, font_bold, player_s or "Ag")
    hero_h = _text_h(draw, font, hero_s or "Ag")
    total_h = player_h + line_gap + hero_h
    top_y = cell_y0 + max(0, (inner_h - total_h) // 2)

    stroke_w = 1
    draw.text(
        (text_left_x, top_y),
        player_s,
        fill="white",
        font=font_bold,
        anchor="lt",
        stroke_width=stroke_w,
        stroke_fill="black",
    )
    draw.text(
        (text_left_x, top_y + player_h + line_gap),
        hero_s,
        fill="white",
        font=font,
        anchor="lt",
        stroke_width=stroke_w,
        stroke_fill="black",
    )


def render_stats_store_to_png(
    player_labels: Sequence[str],
    rows: Sequence[Tuple[str, Sequence[CellValue]]],
    *,
    title: str | None = None,
    header_max_width: int | None = None,
    header_extra: int = 15,
    layout: RenderLayoutOptions | None = None,
    team_header: TeamHeaderOptions | None = None,
    player_hero_names: Sequence[str] | None = None,
    player_hero_portrait_paths: Sequence[Path | None] | None = None,
) -> bytes:
    """Return PNG bytes rendering a stats table."""
    layout = layout or RenderLayoutOptions()
    font_size = layout.font_size
    padding_x = layout.padding_x
    padding_y = layout.padding_y
    grid = layout.grid

    table_text, table_is_numeric = _build_table_text(player_labels, rows)
    font, font_bold = _load_fonts(font_size)

    player_hero_portraits = _load_player_hero_portraits(player_hero_portrait_paths)

    player_count = len(player_labels)
    if team_header is not None:
        derived_blue_kills, derived_red_kills = _derive_team_kills(
            existing_blue_kills=team_header.team_blue_kills,
            existing_red_kills=team_header.team_red_kills,
            rows=rows,
            player_count=player_count,
        )
        derived_blue_level, derived_red_level = _derive_team_levels(
            existing_blue_level=team_header.team_blue_level,
            existing_red_level=team_header.team_red_level,
            rows=rows,
            player_count=player_count,
        )
        team_header = replace(
            team_header,
            team_blue_kills=derived_blue_kills,
            team_red_kills=derived_red_kills,
            team_blue_level=derived_blue_level,
            team_red_level=derived_red_level,
        )

    col_widths, row_height, title_h, img_w, img_h = _measure_table_layout(
        table_text,
        font=font,
        font_bold=font_bold,
        padding_x=padding_x,
        padding_y=padding_y,
        grid=grid,
        title=title,
        header_max_width=header_max_width,
        header_extra=header_extra,
    )

    # Ensure the Player column is wide enough for: portrait + player name + hero name.
    if player_hero_names:
        dummy = Image.new("RGB", (1, 1), "white")
        d = ImageDraw.Draw(dummy)

        line_gap = max(4, padding_y // 2)
        player_h = _text_h(d, font_bold, "Ag")
        hero_h = _text_h(d, font, "Ag")
        text_block_h = player_h + line_gap + hero_h
        # Ensure enough vertical room for a larger portrait while still fitting two lines.
        target_portrait = min(220, int(text_block_h * 1.5))
        needed_row_h = padding_y * 2 + max(text_block_h, target_portrait)
        row_height = max(row_height, int(needed_row_h))

        # Portrait size is chosen inside _draw_player_cell as min(cell_h - padding, cell_w//2).
        planned_portrait = max(1, min(row_height - padding_y * 2, 220))

        max_player_w = max((_text_w(d, font_bold, s) for s in player_labels), default=0)
        max_hero_w = max((_text_w(d, font, s) for s in player_hero_names), default=0)
        max_text_w = max(max_player_w, max_hero_w)

        needed_player_col_w = padding_x + planned_portrait + padding_x + max_text_w + padding_x
        if col_widths:
            col_widths[0] = max(col_widths[0], int(needed_player_col_w))

    col_widths, row_height, img_w, img_h, talent = _apply_talent_sizing(
        table_text,
        col_widths=col_widths,
        row_height=row_height,
        font_size=font_size,
        padding_x=padding_x,
        padding_y=padding_y,
        grid=grid,
        title_h=title_h,
    )

    start_y = 0
    header_layout: TeamHeaderLayout | None = None

    # Add background margins around the table, similar to the existing bottom padding.
    bottom_extra = max(padding_y * 6, row_height // 3, 8)
    side_padding = bottom_extra
    table_w = img_w
    img_w = table_w + side_padding * 2
    table_start_x = side_padding

    if _has_team_header(team_header):
        assert team_header is not None
        dummy = Image.new("RGB", (img_w, 1), "white")
        dummy_draw = ImageDraw.Draw(dummy)
        header_layout = _build_team_header_layout(
            draw=dummy_draw,
            img_w=img_w,
            font_size=font_size,
            padding_y=padding_y,
            team_header=team_header,
        )
        start_y = header_layout.title_height
        img_h += header_layout.title_height

    img = _build_background_image(img_w, img_h)
    draw = ImageDraw.Draw(img)

    if header_layout is None and title:
        draw.text((table_start_x + padding_x, padding_y), title, fill="black", font=font_bold)
        start_y = title_h

    table_ctx = _TableDrawContext(
        col_widths=col_widths,
        row_height=row_height,
        start_x=table_start_x,
        start_y=start_y,
        padding_x=padding_x,
        padding_y=padding_y,
        grid=grid,
    )
    player_opts = _PlayerCellRenderOptions(hero_names=player_hero_names, hero_portraits=player_hero_portraits)
    _draw_table(
        img=img,
        table_text=table_text,
        table_is_numeric=table_is_numeric,
        ctx=table_ctx,
        font=font,
        font_bold=font_bold,
        talent=talent,
        player=player_opts,
    )

    # Keep any remaining bottom padding as background (avoid solid fills).

    if header_layout is not None:
        _draw_team_header(img=img, layout=header_layout)

    # --- encode ---
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_player_hero_portraits(
    player_hero_portrait_paths: Sequence[Path | None] | None,
) -> list[Image.Image | None] | None:
    """Load and convert player hero portraits to RGBA, keeping failures as None."""
    if player_hero_portrait_paths is None:
        return None

    portraits: list[Image.Image | None] = []
    for p in player_hero_portrait_paths:
        if p is None:
            portraits.append(None)
            continue
        try:
            portraits.append(Image.open(p).convert("RGBA"))
        except Exception:
            portraits.append(None)
    return portraits


def _measure_table_layout(
    table_text: List[List[str]],
    *,
    font: Font,
    font_bold: Font,
    padding_x: int,
    padding_y: int,
    grid: int,
    title: str | None,
    header_max_width: int | None,
    header_extra: int,
) -> tuple[List[int], int, int, int, int]:
    """Measure/adjust table geometry, including optional header wrapping."""
    # Initial measurement without forcing header wrapping
    col_widths, row_height, title_h, img_w, img_h = _measure_layout(
        table_text,
        font=font,
        font_bold=font_bold,
        padding_x=padding_x,
        padding_y=padding_y,
        grid=grid,
        title=title,
        header_max_width=None,
    )

    if header_max_width is None:
        auto_threshold = _auto_header_wrap_threshold(col_widths)
        if _maybe_wrap_headers(table_text, font_bold=font_bold, header_max_width=auto_threshold):
            return _measure_layout(
                table_text,
                font=font,
                font_bold=font_bold,
                padding_x=padding_x,
                padding_y=padding_y,
                grid=grid,
                title=title,
                header_max_width=auto_threshold,
                header_extra=header_extra,
            )
        return col_widths, row_height, title_h, img_w, img_h

    return _measure_layout(
        table_text,
        font=font,
        font_bold=font_bold,
        padding_x=padding_x,
        padding_y=padding_y,
        grid=grid,
        title=title,
        header_max_width=header_max_width,
        header_extra=header_extra,
    )