import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .core import MissingReplayArtifactsError, configure_logging
from .parse_service import parse_and_build_table
from . import parse_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown hooks for the FastAPI app."""
    configure_logging()
    logger.info("Starting Replay Parser API...")
    yield
    logger.info("Shutting down Replay Parser API...")


app = FastAPI(title="HotS Replay Parser API", lifespan=lifespan)


class ReplayParseRequest(BaseModel):
    replay_path: str


def _to_http_exception(exc: Exception) -> HTTPException:
    """Normalize internal exceptions to an HTTPException."""
    if isinstance(exc, (MissingReplayArtifactsError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Failed to parse replay")


@app.post("/parse-replay/")
def parse_replay_endpoint(request: ReplayParseRequest):
    """Parse a replay file and return the statistics table as a PNG image."""
    replay_path = request.replay_path
    logger.info("Received request to parse replay: %s", replay_path)
    try:
        png_bytes = parse_and_build_table(replay_path)
    except Exception as exc:
        raise _to_http_exception(exc) from exc
    logger.info("Completed parsing replay: %s", replay_path)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/parse-replay/upload")
async def parse_replay_upload(file: UploadFile = File(...)):
    """Accept a replay file upload and return the parsed PNG bytes."""
    content = await file.read()
    fd, tmp_path = tempfile.mkstemp(suffix=".StormReplay")
    try:
        os.close(fd)
        await asyncio.to_thread(Path(tmp_path).write_bytes, content)

        try:
            png_bytes = await asyncio.to_thread(parse_service.parse_and_build_table, tmp_path)
        except Exception as exc:
            raise _to_http_exception(exc) from exc
        else:
            return Response(content=png_bytes, media_type="image/png")
    finally:
        try:
            await asyncio.to_thread(Path(tmp_path).unlink)
        except Exception:
            logger.exception("Failed to remove temporary uploaded replay")


@app.get("/health")
def health_check():
    """Simple health endpoint for readiness checks."""
    return {"status": "ok"}