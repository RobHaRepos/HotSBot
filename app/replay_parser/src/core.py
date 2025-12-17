from __future__ import annotations

import logging
import os
from dataclasses import dataclass


def configure_logging() -> None:
    """Configure basic logging once."""
    root = logging.getLogger()
    if root.handlers:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class ReplayParserError(Exception):
    """Base exception for replay parsing failures."""


class MissingReplayArtifactsError(ReplayParserError, ValueError):
    """Raised when required replay artifacts are missing."""


class HeroprotocolCliError(ReplayParserError):
    """Raised when the heroprotocol CLI invocation fails."""


@dataclass(frozen=True)
class ParseApiError(ReplayParserError):
    """Raised when the parse API returns an error response."""

    status_code: int
    detail: str

    def __str__(self) -> str:
        return f"Parse API error ({self.status_code}): {self.detail}"
