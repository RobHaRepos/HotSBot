import logging
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .extract_replay_details import extract_details_from_file
from .extract_replay_header import extract_header_from_file
from .extract_replay_tracker_events import build_dynamic_rows, parse_tracker_events_file
from .statistic_png_renderer import TeamHeaderOptions, render_stats_store_to_png
from .web_scaper_hero_img import get_or_download_hero_image_path
from ..schemas.store import InMemoryStatsStore, Value
from .core import MissingReplayArtifactsError


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output_replays"

DEFAULT_CLI_FLAGS = ["--header", "--details", "--gameevents", "--trackerevents"]
TEAM_LEVEL_CATEGORY = "Team Level Achieved"
TEAM_KILLS_CATEGORY = "Team Deaths"

REMOVED_TABLE_CATEGORIES = {
    "Hero",
    "Creep Damage Done",
    "Game Score",
    "Less Than 4 Deaths",
    TEAM_LEVEL_CATEGORY,
    "Team Deaths",
    "Multikills",
}


def parse_replay_with_cli(path: str, flags: list[str] | None = None) -> dict[str, str]:
    """Run heroprotocol CLI to extract artifacts and return their file paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flags = flags or list(DEFAULT_CLI_FLAGS)

    results: dict[str, str] = {}
    try:
        for flag in flags:
            cmd = [sys.executable, "-m", "heroprotocol", flag, path]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if res.returncode != 0:
                logger.error("heroprotocol failed (flag=%s): %s", flag, res.stderr.strip())
                continue

            key = flag.lstrip("--")
            outfile = OUTPUT_DIR / f"{Path(path).stem}.{key}.txt"
            outfile.write_text(res.stdout, encoding="utf-8")
            results[key] = str(outfile)
    except Exception:
        logger.exception("Failed to parse replay via CLI")
        return {}

    return results


def _cleanup_paths(paths: Iterable[Path]) -> None:
    """Best-effort delete of generated artifact files."""
    for artifact_path in paths:
        try:
            artifact_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Failed to remove artifact %s", artifact_path)


def _require_replay_artifacts(artifacts: dict[str, str]) -> tuple[str, str, str]:
    """Validate and return (details, trackerevents, header) artifact paths."""
    details_path = artifacts.get("details")
    trackerevents_path = artifacts.get("trackerevents")
    header_path = artifacts.get("header")
    if not details_path or not trackerevents_path or not header_path:
        raise MissingReplayArtifactsError("Missing required replay artifacts (details/trackerevents/header)")
    return details_path, trackerevents_path, header_path


def _parse_int_values(values: Iterable[Value]) -> list[int] | None:
    """Convert a list of values to ints, returning None on conversion failure."""
    try:
        return [int(v) for v in values]
    except Exception:
        return None


def _derive_team_kills(store: InMemoryStatsStore, player_count: int) -> tuple[int | None, int | None]:
    """Derive (team_blue_kills, team_red_kills) from the TEAM_KILLS_CATEGORY row."""
    row = store.get_row(TEAM_KILLS_CATEGORY)
    if not row:
        return None, None

    values_int = _parse_int_values(row)
    if values_int is None:
        return None, None

    if len(values_int) >= 6:
        # Matches the renderer's derivation logic (keeps blue/red consistent with prior behavior).
        return values_int[5], values_int[0]

    half = max(1, player_count // 2)
    padded = values_int + [0] * max(0, player_count - len(values_int))
    return sum(padded[:half]), sum(padded[half:])


def _derive_team_levels(store: InMemoryStatsStore, player_count: int) -> tuple[int | None, int | None]:
    """Derive (team_blue_level, team_red_level) from the TEAM_LEVEL_CATEGORY row."""
    row = store.get_row(TEAM_LEVEL_CATEGORY)
    if not row:
        return None, None

    values_int = _parse_int_values(row)
    if values_int is None:
        return None, None

    half = max(1, player_count // 2)
    blue_vals = values_int[:half]
    red_vals = values_int[half:]
    return (max(blue_vals) if blue_vals else None), (max(red_vals) if red_vals else None)


def _build_visible_rows(store: InMemoryStatsStore) -> list[tuple[str, list[Value]]]:
    """Build visible (category, values) rows excluding removed categories."""
    return [(row.category, row.values) for row in store.iter_rows() if row.category not in REMOVED_TABLE_CATEGORIES]


def parse_and_build_table(replay: str) -> bytes:
    """Parse a replay file and build a statistics table."""
    artifacts = parse_replay_with_cli(replay, flags=["--details", "--trackerevents", "--header"])
    artifact_paths = [Path(p) for p in artifacts.values() if p]

    try:
        details_path, trackerevents_path, header_path = _require_replay_artifacts(artifacts)

        header = extract_header_from_file(header_path)
        table_header = extract_details_from_file(details_path, game_header=header)

        player_labels = [(player.name or "") for player in table_header.players]

        player_count = len(player_labels)

        store = InMemoryStatsStore(player_labels=player_labels)

        hero_names = [(player.hero or "") for player in table_header.players]
        store.set_row("Hero", hero_names)

        build_dynamic_rows(
            events=parse_tracker_events_file(trackerevents_path),
            player_count=len(player_labels),
            store=store,
        )

        # Keep team kills/levels in the header, but remove those categories from the visible table.
        team_blue_kills, team_red_kills = _derive_team_kills(store, player_count=player_count)
        team_blue_level, team_red_level = _derive_team_levels(store, player_count=player_count)
        rows = _build_visible_rows(store)

        hero_portrait_paths = [get_or_download_hero_image_path(h) for h in hero_names]

        png_bytes = render_stats_store_to_png(
            player_labels=store.player_labels,
            rows=rows,
            header_max_width=80,
            player_hero_names=hero_names,
            player_hero_portrait_paths=hero_portrait_paths,
            team_header=TeamHeaderOptions(
                team_blue_name="TEAM BLUE",
                team_red_name="TEAM RED",
                team_blue_level=team_blue_level,
                team_red_level=team_red_level,
                team_blue_kills=team_blue_kills,
                team_red_kills=team_red_kills,
                game_time_seconds=getattr(header, "elapsed_seconds", None),
            ),
        )
        return png_bytes
    finally:
        _cleanup_paths(artifact_paths)