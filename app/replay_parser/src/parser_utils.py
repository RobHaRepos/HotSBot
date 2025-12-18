from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any


def load_output_file(path: str) -> str:
    """Load the contents of a heroprotocol output file, trying multiple encodings."""
    with open(path, "rb") as fh:
        raw = fh.read()

    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            continue

    return raw.decode("latin-1", errors="ignore")


def load_output_dict_literal(path: str) -> dict[str, Any]:
    """Load a heroprotocol output file expected to contain a single dict literal."""
    value = ast.literal_eval(load_output_file(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a dict literal in {path!r}")
    return dict(value)

def _ensure_str(value: Any) -> str:
    """Ensure the value is a string, decoding bytes if necessary."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)

def iter_python_dict_literals(text: str) -> Iterable[str]:
    """Yield top-level `{...}` dict literals from a concatenated text stream."""
    depth = 0
    start: int | None = None

    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            if depth == 1 and start is not None:
                # Closing the outermost dict literal
                yield text[start:i + 1]
                start = None
                depth = 0
            elif depth > 1:
                depth -= 1