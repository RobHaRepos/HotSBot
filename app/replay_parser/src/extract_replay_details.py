from ..schemas.schemas import GameHeader, Player
from .parser_utils import _ensure_str, load_output_dict_literal


def extract_details_from_file(path: str, game_header: GameHeader | None = None) -> GameHeader:
    """Extract replay details from the heroprotocol --details output file."""
    data = load_output_dict_literal(path)

    map_name = _ensure_str(data.get("m_title", "Unknown Map"))

    players: list[Player] = []
    for player_data in data.get("m_playerList", []):
        slot = player_data.get("m_workingSetSlotId")
        if slot is None:
            continue

        players.append(
            Player(
                name=_ensure_str(player_data.get("m_name", "Unknown")),
                hero=_ensure_str(player_data.get("m_hero", "Unknown")),
                result=player_data.get("m_result"),
                teamId=player_data.get("m_teamId"),
                slotIndex0=slot,
                playerId1=(slot + 1),
            )
        )

    header = game_header or GameHeader()
    header.map_name = map_name
    header.players = players
    return header