from pydantic import BaseModel, Field


class Player(BaseModel):
    name: str | None = None
    hero: str | None = None
    result: int | None = None
    teamId: int | None = None
    slotIndex0: int | None = None
    playerId1: int | None = None


class GameHeader(BaseModel):
    map_name: str | None = None
    elapsed_game_loops: int | None = None
    elapsed_seconds: int | None = None
    players: list[Player] = Field(default_factory=list)
    reached_level_team_0: int | None = None
    reached_level_team_1: int | None = None