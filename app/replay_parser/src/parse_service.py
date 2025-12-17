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
from ..schemas.store import InMemoryStatsStore
from .core import MissingReplayArtifactsError


logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output_replays"


def parse_replay_with_cli(path: str, flags: list[str] | None = None) -> dict[str, str]:
    """Run heroprotocol CLI to extract artifacts and return their file paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flags = flags or ["--header", "--details", "--gameevents", "--trackerevents"]

    results: dict[str, str] = {}
    try:
        for flag in flags:
            cmd = [sys.executable, "-m", "heroprotocol", flag, path]
            res = subprocess.run(cmd, capture_output=True, text=True)
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
    for artifact_path in paths:
        try:
            artifact_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Failed to remove artifact %s", artifact_path)


def parse_and_build_table(replay: str) -> bytes:
    """Parse a replay file and build a statistics table."""
    artifacts = parse_replay_with_cli(replay, flags=["--details", "--trackerevents", "--header"])
    artifact_paths = [Path(p) for p in artifacts.values() if p]
    details_path = artifacts.get("details")
    trackerevents_path = artifacts.get("trackerevents")
    header_path = artifacts.get("header")

    try:
        if not details_path or not trackerevents_path or not header_path:
            raise MissingReplayArtifactsError("Missing required replay artifacts (details/trackerevents/header)")

        header = extract_header_from_file(header_path)
        table_header = extract_details_from_file(details_path, game_header=header)
        
        player_labels = [(player.name or "") for player in table_header.players]
        
        store = InMemoryStatsStore(player_labels=player_labels)

        hero_names = [(player.hero or "") for player in table_header.players]
        store.set_row("Hero", hero_names)

        build_dynamic_rows(
            events=parse_tracker_events_file(trackerevents_path),
            player_count=len(player_labels),
            store=store,
        )

        # Keep team levels in the header, but remove the Team Level Achieved column from the table.
        team_blue_level = None
        team_red_level = None
        team_level_row = store.get_row("Team Level Achieved")
        if team_level_row:
            half = max(1, len(player_labels) // 2)
            try:
                blue_vals = [int(v) for v in team_level_row[:half]]
                red_vals = [int(v) for v in team_level_row[half:]]
                team_blue_level = max(blue_vals) if blue_vals else None
                team_red_level = max(red_vals) if red_vals else None
            except Exception:
                team_blue_level = None
                team_red_level = None

        removed_categories = {
            "Hero",
            "Creep Damage Done",
            "Game Score",
            "Team Level Achieved",
            "Multikills",
        }

        rows = [(row.category, row.values) for row in store.iter_rows() if row.category not in removed_categories]

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
                game_time_seconds=getattr(header, "elapsed_seconds", None),
            ),
        )
        return png_bytes
    finally:
        _cleanup_paths(artifact_paths)