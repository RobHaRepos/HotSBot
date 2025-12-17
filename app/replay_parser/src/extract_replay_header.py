from ..schemas.schemas import GameHeader
from .parser_utils import load_output_dict_literal


def extract_header_from_file(path: str, game_header: GameHeader | None = None) -> GameHeader:
    """Extract replay header from the heroprotocol --header output file."""
    data = load_output_dict_literal(path)

    elapsed_loops = data.get("m_elapsedGameLoops", 0)
    elapsed_seconds = (elapsed_loops or 0) // 16

    header = game_header or GameHeader()
    if elapsed_loops is not None:
        header.elapsed_game_loops = int(elapsed_loops)
        header.elapsed_seconds = int(elapsed_seconds)

    return header