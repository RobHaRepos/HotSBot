import ast
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..schemas.store import InMemoryStatsStore
from .parser_utils import _ensure_str, iter_python_dict_literals, load_output_file

SCORE_RESULT = "NNet.Replay.Tracker.SScoreResultEvent"
TRACKED_STATS: list[tuple[str, str]] = [
    ("Kills", "SoloKill"),
    ("Deaths", "Deaths"),
    ("Assists", "Assists"),
    ("Team Deaths", "TeamTakedowns"),
    ("Highest Kill Streak", "HighestKillStreak"),
    ("Multikills", "Multikills"),
    ("Less Than 4 Deaths", "LessThan4Deaths"),
    ("Outnumbered Deaths", "OutnumberedDeaths"),
    ("Hero Damage", "HeroDamage"),
    ("Physical Damage", "PhysicalDamage"),
    ("Spell Damage", "SpellDamage"),
    ("Teamfight Hero Damage", "TeamfightHeroDamage"),
    ("Damage Taken", "DamageTaken"),
    ("Damage Soaked", "DamageSoaked"),
    ("Teamfight Damage Taken", "TeamfightDamageTaken"),
    ("Healing Done", "Healing"),
    ("Self Healing", "SelfHealing"),
    ("Clutch Heals Performed", "ClutchHealsPerformed"),
    ("Teamfight Healing", "TeamfightHealingDone"),
    ("Protection Given", "ProtectionGivenToAllies"),
    ("Siege Damage", "SiegeDamage"),
    ("Structure Damage", "StructureDamage"),
    ("Creep Damage Done", "CreepDamageDone"),
    ("Minion Damage", "MinionDamage"),
    ("Minion Kills", "MinionKills"),
    ("Experience Contribution", "ExperienceContribution"),
    ("Time CC'ed Enemies", "TimeCCdEnemyHeroes"),
    ("Time Rooting Enemy Heroes", "TimeRootingEnemyHeroes"),
    ("Time Silencing Enemy Heroes", "TimeSilencingEnemyHeroes"),
    ("Time Stunning Enemy Heroes", "TimeStunningEnemyHeroes"),
    ("Escapes Performed", "EscapesPerformed"),
    ("Teamfight Escapes", "TeamfightEscapesPerformed"),
    ("Vengeances Performed", "VengeancesPerformed"),
    ("Game Score", "GameScore"),
    ("Team Level Achieved", "TeamLevel"),
    ("Time Spent Dead", "TimeSpentDead"),
]


def parse_tracker_events_file(path: str) -> list[dict[str, Any]]:
    """Parse the heroprotocol --trackerevents output file and return a list of events."""
    text = load_output_file(path)

    events: list[dict[str, Any]] = []
    for block in iter_python_dict_literals(text):
        block = block.strip()
        if not block:
            continue
        try:
            value = ast.literal_eval(block)
        except Exception:
            continue

        if isinstance(value, dict):
            events.append(value)

    return events


def _find_last_score_event(events: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Find the last SScoreResultEvent in the list of events."""
    last: Mapping[str, Any] | None = None
    for event in events:
        if event.get("_event") == SCORE_RESULT:
            last = event
    return last

def _latest_value(series: Sequence[Mapping[str, Any]]) -> int:
    """Get the latest value from a series of tracker event values."""
    if not series:
        return 0
    value = series[-1].get("m_value", 0)
    return int(value)

def extract_score(event: Mapping[str, Any], stat_name: str, player_count: int) -> list[int]:
    """Extract a score row from a SScoreResultEvent."""
    for instance in (event.get("m_instanceList")) or []:
        name = _ensure_str(instance.get("m_name"))
        if name != stat_name:
            continue
        
        m_values = instance.get("m_values") or []
        out: list[int] = []
        for column in range(player_count):
            series = m_values[column] if column < len(m_values) else []
            out.append(_latest_value(series))
        return out
    
    return [0] * player_count


def build_dynamic_rows(events: Iterable[Mapping[str, Any]], player_count: int, store: InMemoryStatsStore) -> None:
    """Build dynamic statistic rows from tracker events."""
    score_event = _find_last_score_event(events)
    if not score_event:
        return
    
    for tracker_name, label in TRACKED_STATS:
        row = extract_score(score_event, label, player_count)
        store.set_row(tracker_name, row)