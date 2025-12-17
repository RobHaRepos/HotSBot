from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.replay_parser.src import discord as discord_module


class _DummyResponse:
    def __init__(self, status_code: int, content: bytes, text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text


class _DummyAsyncClient:
    def __init__(self, status_code: int):
        self._status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, _url: str, json=None, files=None, **kwargs):
        # Give the function an awaitable to satisfy linters/analysis tools
        await asyncio.sleep(0)
        # If the client is given files (multipart), ensure any file-like objects are closed to avoid locks on Windows
        if files:
            for part in files.values():
                # part may be (filename, fileobj, content_type) or (filename, bytes, content_type)
                if isinstance(part, tuple) and len(part) >= 2:
                    fileobj = part[1]
                    try:
                        # If it's a bytes-like object, it won't have close()
                        fileobj.close()
                    except Exception:
                        pass

        return _DummyResponse(
            status_code=self._status_code,
            content=b"\x89PNG\r\n\x1a\n" + b"data",
            text="error" if self._status_code != 200 else "",
        )


class _Ctx:
    def __init__(self, attachment: SimpleNamespace):
        self.message = SimpleNamespace(attachments=[attachment])
        self.sent: list[tuple[tuple, dict]] = []

    async def send(self, *args, **kwargs):
        await asyncio.sleep(0)
        self.sent.append((args, kwargs))


class _Attachment:
    def __init__(self, tmp_path: Path, filename: str):
        self.filename = filename
        self._tmp_path = tmp_path / filename

    async def save(self, path: Path):
        await asyncio.sleep(0)
        path.write_text("replay", encoding="utf-8")


def test_parse_command_removes_temp_file_on_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    att = _Attachment(tmp_path, "example.StormReplay")
    ctx = _Ctx(att)  # type: ignore[arg-type]

    monkeypatch.setattr(
        discord_module,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kw: _DummyAsyncClient(status_code=200)),
    )
    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(File=lambda *args, **kwargs: (args, kwargs)))
    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))

    asyncio.run(discord_module.parse(ctx))  # type: ignore[arg-type]

    temp_file = tmp_path / "temp" / "example.StormReplay"
    assert not temp_file.exists()
    file_args = ctx.sent[0][1]["file"][0]
    assert isinstance(file_args[0], io.BytesIO)


def test_parse_command_reports_error_and_still_cleans(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    att = _Attachment(tmp_path, "fail.StormReplay")
    ctx = _Ctx(att)  # type: ignore[arg-type]

    monkeypatch.setattr(
        discord_module,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kw: _DummyAsyncClient(status_code=500)),
    )
    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(File=lambda *args, **kwargs: (args, kwargs)))
    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))

    asyncio.run(discord_module.parse(ctx))  # type: ignore[arg-type]

    temp_file = tmp_path / "temp" / "fail.StormReplay"
    assert not temp_file.exists()
    assert "Failed to parse replay" in ctx.sent[0][0][0]

def test_bot_mention_triggers_reaction_and_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    att = _Attachment(tmp_path, "mention.StormReplay")

    class Channel:
        def __init__(self):
            self.sent = []

        async def send(self, *args, **kwargs):
            await asyncio.sleep(0)
            self.sent.append((args, kwargs))

    class Message:
        def __init__(self, attachments, channel, mentions):
            self.attachments = attachments
            self.channel = channel
            self.mentions = mentions
            self.author = SimpleNamespace(bot=False)
            self.reactions = []

        async def add_reaction(self, emoji):
            await asyncio.sleep(0)
            self.reactions.append(emoji)

    channel = Channel()
    message = Message([att], channel, [discord_module.bot.user])

    monkeypatch.setattr(
        discord_module,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kw: _DummyAsyncClient(status_code=200)),
    )
    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(File=lambda *args, **kwargs: (args, kwargs)))
    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))

    # Run the on_message handler
    asyncio.run(discord_module.on_message(message)) # type: ignore

    # Reaction should have been added
    assert "👀" in message.reactions

    # Channel send should have been invoked with a file
    assert channel.sent
    file_args = channel.sent[0][1]["file"][0]
    assert isinstance(file_args[0], io.BytesIO)


def test_missing_permissions_falls_back_to_dm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    att = _Attachment(tmp_path, "no_perms.StormReplay")

    sent_to_author = []

    class Channel:
        async def send(self, *args, **kwargs):
            await asyncio.sleep(0)
            # Simulate missing permissions
            raise discord_module.discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), {})  # type: ignore

    class Author:
        async def send(self, *args, **kwargs):
            await asyncio.sleep(0)
            sent_to_author.append((args, kwargs))

    class Message:
        def __init__(self):
            self.attachments = [att]
            self.channel = Channel()
            self.mentions = [discord_module.bot.user]
            self.author = Author()

        async def add_reaction(self, emoji):
            await asyncio.sleep(0)
            # record reaction attempt (no-op)
            # no-op intentionally

    message = Message()

    monkeypatch.setattr(
        discord_module,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kw: _DummyAsyncClient(status_code=200)),
    )
    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))

    asyncio.run(discord_module.on_message(message)) # type: ignore

    # DM fallback should have been attempted
    assert sent_to_author


def test_delete_message_if_bot_success(monkeypatch: pytest.MonkeyPatch):
    deleted = {}
    # bot.user is a read-only property; set internal _connection.user for testing
    monkeypatch.setattr(discord_module.bot, "_connection", SimpleNamespace(user=SimpleNamespace(id=9999)), raising=False)

    class Message:
        def __init__(self):
            self.author = SimpleNamespace(id=getattr(discord_module.bot.user, "id", 9999))

        async def delete(self):
            await asyncio.sleep(0)
            deleted['ok'] = True

    async def responder(content: str, ephemeral: bool = True):
        await asyncio.sleep(0)
        deleted['resp'] = (content, ephemeral)

    msg = Message()
    res = asyncio.run(discord_module.delete_message_if_bot(msg, respond_fn=responder)) # type: ignore
    assert res is True
    assert deleted.get('ok')
    assert deleted.get('resp') and deleted['resp'][0] == 'Message deleted.'


def test_delete_message_if_bot_denies_when_not_bot(monkeypatch: pytest.MonkeyPatch):
    called = {}
    # bot.user is a read-only property; set internal _connection.user for testing
    monkeypatch.setattr(discord_module.bot, "_connection", SimpleNamespace(user=SimpleNamespace(id=9999)), raising=False)

    class Message:
        def __init__(self):
            self.author = SimpleNamespace(id=12345)

        async def delete(self):
            await asyncio.sleep(0)
            called['deleted'] = True

    async def responder(content: str, ephemeral: bool = True):
        await asyncio.sleep(0)
        called['resp'] = (content, ephemeral)

    msg = Message()
    res = asyncio.run(discord_module.delete_message_if_bot(msg, respond_fn=responder)) # type: ignore
    assert res is False
    assert 'deleted' not in called
    assert called.get('resp') and 'only delete' in called['resp'][0]


def test_bot_mention_ignored_when_no_attachment(monkeypatch: pytest.MonkeyPatch):
    class Message:
        def __init__(self):
            self.attachments = []
            self.channel = SimpleNamespace(sent=[])
            self.mentions = [discord_module.bot.user]
            self.author = SimpleNamespace(bot=False)

        async def add_reaction(self, emoji):
            raise AssertionError("Should not react when no attachment")

    message = Message()
    # Should not raise
    asyncio.run(discord_module.on_message(message)) # type: ignore


def test_process_message_attachment_reports_missing_attachment(monkeypatch: pytest.MonkeyPatch):
    sent: list[str] = []

    class Channel:
        async def send(self, content: str):
            await asyncio.sleep(0)
            sent.append(content)

    class Message:
        def __init__(self):
            self.attachments = []
            self.channel = Channel()
            self.author = SimpleNamespace(send=lambda *_a, **_k: asyncio.sleep(0))

    msg = Message()
    asyncio.run(discord_module._process_message_attachment(msg))  # type: ignore[arg-type]
    assert sent and ".StormReplay" in sent[0]


def test_send_png_uses_channel_send_when_send_fn_none(monkeypatch: pytest.MonkeyPatch):
    sent = {}

    class Channel:
        async def send(self, *, file=None):
            await asyncio.sleep(0)
            sent["file"] = file

    class Message:
        channel = Channel()
        author = SimpleNamespace(send=lambda *_a, **_k: asyncio.sleep(0))

    asyncio.run(discord_module._send_png(Message(), b"\x89PNG\r\n\x1a\n", send_fn=None))  # type: ignore[arg-type]
    assert sent.get("file") is not None


def test_on_ready_sync_success(monkeypatch: pytest.MonkeyPatch):
    called = {"n": 0}

    async def sync():
        await asyncio.sleep(0)
        called["n"] += 1

    monkeypatch.setattr(discord_module.bot.tree, "sync", sync)
    asyncio.run(discord_module.on_ready())
    assert called["n"] == 1


def test_on_ready_sync_failure_is_caught(monkeypatch: pytest.MonkeyPatch):
    async def sync():
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    monkeypatch.setattr(discord_module.bot.tree, "sync", sync)
    # Should not raise
    asyncio.run(discord_module.on_ready())


def test_delete_message_context_responder_paths(monkeypatch: pytest.MonkeyPatch):
    called: list[tuple[str, bool]] = []

    async def fake_delete(_message, respond_fn=None):
        if respond_fn is not None:
            await respond_fn("ok", ephemeral=True)
        return True

    monkeypatch.setattr(discord_module, "delete_message_if_bot", fake_delete)

    class Response:
        def __init__(self, done: bool):
            self._done = done

        def is_done(self) -> bool:
            return self._done

        async def send_message(self, content: str, *, ephemeral: bool = True):
            await asyncio.sleep(0)
            called.append((content, ephemeral))

    class Followup:
        async def send(self, content: str, *, ephemeral: bool = True):
            await asyncio.sleep(0)
            called.append((content, ephemeral))

    class Interaction:
        def __init__(self, done: bool):
            self.response = Response(done)
            self.followup = Followup()

    class Message:
        pass

    callback = getattr(discord_module.delete_message_context, "callback")
    asyncio.run(callback(Interaction(done=False), Message()))  # type: ignore[arg-type]
    asyncio.run(callback(Interaction(done=True), Message()))  # type: ignore[arg-type]
    assert ("ok", True) in called


def test_main_raises_when_token_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(discord_module, "TOKEN", "")
    with pytest.raises(RuntimeError):
        discord_module.main()


def test_delete_message_if_bot_denies_when_bot_not_ready(monkeypatch: pytest.MonkeyPatch):
    # Simulate bot.user not being ready.
    monkeypatch.setattr(discord_module.bot, "_connection", SimpleNamespace(user=None), raising=False)

    called = {}

    async def responder(content: str, ephemeral: bool = True):
        await asyncio.sleep(0)
        called["msg"] = (content, ephemeral)

    class Message:
        author = SimpleNamespace(id=123)

        async def delete(self):
            raise AssertionError("Should not delete")

    res = asyncio.run(discord_module.delete_message_if_bot(Message(), respond_fn=responder))  # type: ignore[arg-type]
    assert res is False
    assert called.get("msg") and "not ready" in called["msg"][0].lower()


def test_process_message_attachment_handles_post_exception_and_file_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sent: list[str] = []

    class Channel:
        async def send(self, content: str = "", **_kwargs):
            await asyncio.sleep(0)
            sent.append(content)

    class Attachment:
        filename = "missing.StormReplay"

        async def save(self, _path: Path):
            await asyncio.sleep(0)
            # Intentionally do not create the file.

    class Message:
        attachments = [Attachment()]
        channel = Channel()
        author = SimpleNamespace(send=lambda *_a, **_k: asyncio.sleep(0))

    # Force temp directory into tmp_path.
    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))

    asyncio.run(discord_module._process_message_attachment(Message()))  # type: ignore[arg-type]
    assert sent and "Failed" in sent[0]


def test_process_message_attachment_logs_unlink_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class Attachment:
        filename = "unlink.StormReplay"

        async def save(self, path: Path):
            await asyncio.sleep(0)
            path.write_text("x", encoding="utf-8")

    class Channel:
        async def send(self, *args, **kwargs):
            await asyncio.sleep(0)

    class Message:
        attachments = [Attachment()]
        channel = Channel()
        author = SimpleNamespace(send=lambda *_a, **_k: asyncio.sleep(0))

    monkeypatch.setattr(discord_module, "Path", lambda *parts: Path(tmp_path, *parts))
    monkeypatch.setattr(discord_module, "_post_replay_file", lambda *_a, **_k: asyncio.sleep(0, result=b"\x89PNG\r\n\x1a\n"))
    monkeypatch.setattr(discord_module, "_send_png", lambda *_a, **_k: asyncio.sleep(0))

    orig_unlink = Path.unlink

    def unlink_raises(self: Path):
        if self.name == "unlink.StormReplay":
            raise OSError("locked")
        return orig_unlink(self)

    monkeypatch.setattr(Path, "unlink", unlink_raises)
    # Should not raise
    asyncio.run(discord_module._process_message_attachment(Message()))  # type: ignore[arg-type]


def test_send_text_forbidden_add_reaction_and_dm_failures(monkeypatch: pytest.MonkeyPatch):
    class Channel:
        async def send(self, _content: str):
            await asyncio.sleep(0)
            raise discord_module.discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), {})  # type: ignore

    class Author:
        async def send(self, _content: str):
            await asyncio.sleep(0)
            raise RuntimeError("dm fail")

    class Message:
        channel = Channel()
        author = Author()

        async def add_reaction(self, _emoji: str):
            await asyncio.sleep(0)
            raise RuntimeError("no react")

    # Should not raise
    asyncio.run(discord_module._send_text(Message(), "hi", send_fn=None))  # type: ignore[arg-type]


def test_on_message_early_return_when_author_is_bot(monkeypatch: pytest.MonkeyPatch):
    called = {"n": 0}

    async def process_commands(_message):
        called["n"] += 1

    monkeypatch.setattr(discord_module.bot, "process_commands", process_commands)

    class Message:
        author = SimpleNamespace(bot=True)

    asyncio.run(discord_module.on_message(Message()))  # type: ignore[arg-type]
    assert called["n"] == 0


def test_on_message_processes_commands_when_content_present(monkeypatch: pytest.MonkeyPatch):
    called = {"n": 0}

    async def process_commands(_message):
        await asyncio.sleep(0)
        called["n"] += 1

    monkeypatch.setattr(discord_module.bot, "process_commands", process_commands)

    class Message:
        attachments = []
        mentions = []
        author = SimpleNamespace(bot=False)
        content = "hello"

    asyncio.run(discord_module.on_message(Message()))  # type: ignore[arg-type]
    assert called["n"] == 1


def test_on_message_reaction_failure_is_ignored(monkeypatch: pytest.MonkeyPatch):
    async def process_commands(_message):
        await asyncio.sleep(0)

    monkeypatch.setattr(discord_module.bot, "process_commands", process_commands)

    class Attachment:
        filename = "x.StormReplay"

        async def save(self, path: Path):
            await asyncio.sleep(0)
            path.write_text("replay", encoding="utf-8")

    class Message:
        attachments = [Attachment()]
        mentions = [discord_module.bot.user]
        author = SimpleNamespace(bot=False)
        content = "hello"

        async def add_reaction(self, _emoji: str):
            await asyncio.sleep(0)
            raise RuntimeError("no perms")

    monkeypatch.setattr(discord_module, "_process_message_attachment", lambda *_a, **_k: asyncio.sleep(0))
    asyncio.run(discord_module.on_message(Message()))  # type: ignore[arg-type]


def test_delete_message_context_exception_is_caught(monkeypatch: pytest.MonkeyPatch):
    async def fake_delete(_message, respond_fn=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(discord_module, "delete_message_if_bot", fake_delete)

    class Response:
        def is_done(self) -> bool:
            return False

        async def send_message(self, *_a, **_k):
            await asyncio.sleep(0)

    class Interaction:
        response = Response()
        followup = SimpleNamespace(send=lambda *_a, **_k: asyncio.sleep(0))

    callback = getattr(discord_module.delete_message_context, "callback")
    asyncio.run(callback(Interaction(), SimpleNamespace()))  # type: ignore[arg-type]


def test_main_runs_bot_when_token_present(monkeypatch: pytest.MonkeyPatch):
    called = {"token": None}

    monkeypatch.setattr(discord_module, "TOKEN", "token")
    monkeypatch.setattr(discord_module, "configure_logging", lambda: None)
    monkeypatch.setattr(discord_module.bot, "run", lambda t: called.__setitem__("token", t))

    discord_module.main()
    assert called["token"] == "token"
