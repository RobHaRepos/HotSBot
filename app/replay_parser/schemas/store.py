from dataclasses import dataclass
from collections.abc import Iterator, Sequence

Value = int | str


@dataclass(frozen=True)
class StatsRow:
    category: str
    values: list[Value]


class InMemoryStatsStore:
    """Simple per-run in-memory stats table (category -> 10 player values)."""

    def __init__(self, player_labels: Sequence[str]):
        self._player_labels: list[str] = list(player_labels)
        self._rows: dict[str, list[Value]] = {}

    @property
    def player_count(self) -> int:
        return len(self._player_labels)

    @property
    def player_labels(self) -> Sequence[str]:
        return self._player_labels

    def clear(self) -> None:
        self._rows.clear()

    def add(self, category: str, player_index: int, delta: int = 1) -> None:
        """Add delta to a specific player's value in the given category."""
        if not (0 <= player_index < self.player_count):
            raise IndexError(f"Invalid player index: {player_index}")

        row = self._rows.get(category)
        if row is None:
            row = [0] * self.player_count
            self._rows[category] = row
        row[player_index] = int(row[player_index]) + int(delta)

    def set_row(self, category: str, values: Sequence[Value]) -> None:
        """Set a full row of values for the given category."""
        if len(values) != self.player_count:
            raise ValueError(f"Expected {self.player_count} values, got {len(values)}")
        self._rows[str(category)] = list(values)

    def get_row(self, category: str) -> list[Value] | None:
        """Get a row by category, or None if not found."""
        row = self._rows.get(category)
        return list(row) if row is not None else None

    def iter_rows(self) -> Iterator[StatsRow]:
        """Read 'top down' in insertion order."""
        for category, values in self._rows.items():
            yield StatsRow(category=category, values=list(values))