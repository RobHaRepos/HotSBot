import asyncio
import io
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Protocol

import discord
import httpx
from discord.ext import commands

from .core import ParseApiError, configure_logging

logger = logging.getLogger(__name__)

TOKEN = os.environ.get("DISCORD_TOKEN")
PARSE_API_URL = os.environ.get("PARSE_API_URL", "http://localhost:8000/parse-replay/")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


class ResponderFn(Protocol):
    def __call__(self, content: str, *, ephemeral: bool = True) -> Awaitable[None]: ...


class ChannelSendFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


async def delete_message_if_bot(message: discord.Message, respond_fn: ResponderFn | None = None) -> bool:
    """Delete `message` only if it was authored by this bot."""
    # If bot.user is not ready, deny deletion
    if bot.user is None or not getattr(bot.user, "id", None):
        if respond_fn is not None:
            await respond_fn("Bot is not ready yet.", ephemeral=True)
        return False

    if message.author.id != bot.user.id:
        if respond_fn is not None:
            await respond_fn("I can only delete my own messages.", ephemeral=True)
        return False

    try:
        await message.delete()
        if respond_fn is not None:
            await respond_fn("Message deleted.", ephemeral=True)
        return True
    except discord.Forbidden:
        if respond_fn is not None:
            await respond_fn("I don't have permission to delete that message.", ephemeral=True)
        return False
    except Exception:
        logger.exception("Failed to delete message")
        if respond_fn is not None:
            await respond_fn("Failed to delete message.", ephemeral=True)
        return False

@bot.command()
async def parse(ctx: commands.Context) -> None:
    """Parse a replay attachment and respond with a PNG."""
    await _process_message_attachment(ctx.message, send_fn=ctx.send)


async def _process_message_attachment(
    message: discord.Message,
    send_fn: ChannelSendFn | None = None,
) -> None:
    """Save the first attachment, POST it to the parse service and send the PNG back to the origin channel."""
    attachments = getattr(message, "attachments", []) or []
    if not attachments:
        await _send_text(message, "No attachment found. Please attach a .StormReplay file.", send_fn=send_fn)
        return

    attachment = attachments[0]
    local_path = Path("temp") / attachment.filename
    local_path.parent.mkdir(exist_ok=True)
    await attachment.save(local_path)

    try:
        try:
            png_bytes = await _post_replay_file(local_path, attachment.filename)
        except ParseApiError as exc:
            logger.info("Parse failed: %s", exc)
            await _send_text(message, "Failed to parse replay.", send_fn=send_fn)
            return
        except Exception:
            logger.exception("Parse API call failed")
            await _send_text(message, "Failed to parse replay.", send_fn=send_fn)
            return

        await _send_png(message, png_bytes, send_fn=send_fn)
    finally:
        try:
            await asyncio.to_thread(local_path.unlink)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("Failed to unlink temporary replay file")


async def _post_replay_file(local_path: Path, filename: str) -> bytes:
    """POST the replay file to the parse API and return PNG bytes."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        content = await asyncio.to_thread(local_path.read_bytes)
        files = {"file": (filename, content, "application/octet-stream")}
        resp = await client.post(PARSE_API_URL, files=files)
        if resp.status_code != 200:
            raise ParseApiError(status_code=resp.status_code, detail=resp.text)
        return resp.content


async def _send_text(message: discord.Message, content: str, *, send_fn: ChannelSendFn | None) -> None:
    """Send a text message to the origin channel or via the provided send function."""
    try:
        if send_fn is not None:
            await send_fn(content)
        else:
            await message.channel.send(content)
    except discord.Forbidden:
        await _handle_forbidden_send(message, dm_content="I can't send messages in that channel. Please grant me `Send Messages` permission.")


async def _send_png(message: discord.Message, png_bytes: bytes, *, send_fn: ChannelSendFn | None) -> None:
    """Send a PNG file to the origin channel or via the provided send function."""
    try:
        file = discord.File(io.BytesIO(png_bytes), filename="replay_stats.png")
        if send_fn is not None:
            await send_fn(file=file)
        else:
            await message.channel.send(file=file)
    except discord.Forbidden:
        await _handle_forbidden_send(
            message,
            dm_content=(
                "I don't have permission to send the replay results in that channel. "
                "Please grant me the `Send Messages` and `Attach Files` permissions."
            ),
        )


async def _handle_forbidden_send(message: discord.Message, *, dm_content: str) -> None:
    """Handle a discord.Forbidden error when sending to a channel by notifying the user via DM."""
    logger.warning("Missing permission to send to channel %s", getattr(message.channel, "id", "unknown"))
    try:
        await message.add_reaction("❌")
    except Exception:
        pass
    try:
        await message.author.send(dm_content)
    except Exception:
        logger.exception("Failed to DM user about missing permissions")


@bot.event
async def on_message(message: discord.Message) -> None:
    """React and process when the bot is mentioned with an attached replay."""
    if getattr(message.author, "bot", False):
        return

    mentions = getattr(message, "mentions", []) or []
    attachments = getattr(message, "attachments", []) or []

    if bot.user in mentions and attachments:
        try:
            await message.add_reaction("👀")
        except Exception:
            pass

        await _process_message_attachment(message)

    if getattr(message, "content", None) is not None:
        await bot.process_commands(message)


@bot.event
async def on_ready() -> None:
    """Sync app commands when the bot is ready so the context menu appears."""
    try:
        await bot.tree.sync()
        logger.info("Synced app command tree")
    except Exception:
        logger.exception("Failed to sync app command tree")


@bot.tree.context_menu(name="Delete message")
async def delete_message_context(interaction: discord.Interaction, message: discord.Message):
    """Context-menu (right-click -> Apps -> Delete message) handler."""
    async def responder(content: str, *, ephemeral: bool = True) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=ephemeral)
            return
        await interaction.response.send_message(content, ephemeral=ephemeral)

    try:
        await delete_message_if_bot(message, respond_fn=responder)
    except Exception:
        logger.exception("Error handling delete message context command")


def main() -> None:
    """Run the Discord bot. This function is safe to import without starting the bot."""
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable not set")

    configure_logging()
    logger.info("Starting Discord bot")
    bot.run(TOKEN)


if __name__ == "__main__":  # pragma: no cover
    main()