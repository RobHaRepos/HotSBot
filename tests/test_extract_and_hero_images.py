from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.replay_parser.src.extract_replay_details import extract_details_from_file
from app.replay_parser.src.extract_replay_header import extract_header_from_file
from app.replay_parser.src.parser_utils import load_output_dict_literal
from app.replay_parser.src import web_scaper_hero_img as hero_img


def test_load_output_dict_literal_rejects_non_dict(tmp_path: Path):
    path = tmp_path / "bad.txt"
    path.write_text("['not a dict']", encoding="utf-8")

    with pytest.raises(ValueError):
        load_output_dict_literal(str(path))


def test_extract_header_from_file_computes_elapsed_seconds(tmp_path: Path):
    path = tmp_path / "header.txt"
    path.write_text("{'m_elapsedGameLoops': 160}", encoding="utf-8")

    header = extract_header_from_file(str(path))
    assert header.elapsed_game_loops == 160
    assert header.elapsed_seconds == 10


def test_extract_header_from_file_keeps_none_elapsed(tmp_path: Path):
    path = tmp_path / "header.txt"
    path.write_text("{'m_elapsedGameLoops': None}", encoding="utf-8")

    header = extract_header_from_file(str(path))
    assert header.elapsed_game_loops is None
    assert header.elapsed_seconds is None


def test_extract_details_from_file_builds_players(tmp_path: Path):
    path = tmp_path / "details.txt"
    payload = {
        "m_title": "Silver City",
        "m_playerList": [
            {
                "m_workingSetSlotId": 0,
                "m_name": "Alpha",
                "m_hero": "Nova",
                "m_result": 1,
                "m_teamId": 0,
            },
            {
                "m_workingSetSlotId": 1,
                "m_name": b"Bravo",
                "m_hero": "Raynor",
                "m_result": 2,
                "m_teamId": 1,
            },
            {"m_workingSetSlotId": None, "m_name": "Skip"},
        ],
    }
    path.write_text(repr(payload), encoding="utf-8")

    header = extract_details_from_file(str(path))
    assert header.map_name == "Silver City"
    assert len(header.players) == 2
    assert header.players[0].name == "Alpha"
    assert header.players[0].hero == "Nova"
    assert header.players[0].playerId1 == 1
    assert header.players[1].name == "Bravo"
    assert header.players[1].playerId1 == 2


def test_extract_hero_cards_from_html_parses_two_cards():
    html = """
    <blz-hero-card hero-name=\"Abathur\" icon=\"https://cdn/x_card_icon.webp\">
      <blz-image slot=\"image\" src=\"https://cdn/abathur_card_portrait.webp\"></blz-image>
    </blz-hero-card>
    <blz-hero-card hero-name=\"Lt. Morales\" icon=\"https://cdn/morales_card_portrait.webp\">
      <blz-image slot=\"image\" src=\"\"></blz-image>
    </blz-hero-card>
    """

    cards = hero_img._extract_hero_cards_from_html(html)
    assert {c["name"] for c in cards} == {"Abathur", "Lt. Morales"}


def test_extract_hero_cards_skips_cards_without_hero_name():
    html = "<blz-hero-card icon=\"x\"></blz-hero-card>"
    assert hero_img._extract_hero_cards_from_html(html) == []


def test_fetch_html_success(monkeypatch: pytest.MonkeyPatch):
    class Resp:
        text = "<html/>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(hero_img.requests, "get", lambda *_a, **_k: Resp())
    assert hero_img._fetch_html("https://example") == "<html/>"


def test_scrape_cards_via_playwright_happy_path_via_fake_module(monkeypatch: pytest.MonkeyPatch):
    html = "<blz-hero-card hero-name=\"Abathur\"><blz-image slot=\"image\" src=\"x_card_portrait.webp\"></blz-image></blz-hero-card>"

    class Page:
        def goto(self, *_a, **_k):
            return None

        def content(self):
            return html

    class Ctx:
        def new_page(self):
            return Page()

    class Browser:
        def new_context(self, **_kw):
            return Ctx()

        def close(self):
            return None

    class Chromium:
        def launch(self, **_kw):
            return Browser()

    class Playwright:
        chromium = Chromium()

    class _CM:
        def __enter__(self):
            return Playwright()

        def __exit__(self, exc_type, exc, tb):
            return False

    # Patch sys.modules so `from playwright.sync_api import sync_playwright` works.
    import sys

    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=lambda: _CM()))

    cards = hero_img._scrape_cards_via_playwright("https://example", headless=True)
    assert cards and cards[0]["name"] == "Abathur"


def test_build_hero_map_filters_invalid_entries():
    results = [
        {"name": "", "portrait": "x_card_portrait.webp", "icon": ""},
        {"name": "Abathur", "portrait": "", "icon": ""},
        {"name": "Abathur", "portrait": "x_card_portrait.webp", "icon": ""},
    ]
    m = hero_img._build_hero_map(results)
    assert hero_img._normalize_hero_name("Abathur") in m


def test_scrape_hero_image_map_falls_back_to_playwright_and_writes_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(hero_img, "_scrape_cards_via_requests", lambda *_a, **_k: [])
    monkeypatch.setattr(
        hero_img,
        "_scrape_cards_via_playwright",
        lambda *_a, **_k: [{"name": "Abathur", "portrait": "x_card_portrait.webp", "icon": ""}],
    )

    cache = tmp_path / "cache.json"
    m = hero_img.scrape_hero_image_map(cache_path=cache)
    assert m
    assert cache.exists()


def test_download_hero_image_raises_for_unknown_hero(tmp_path: Path):
    with pytest.raises(ValueError):
        hero_img.download_hero_image("Unknown", {}, out_dir=tmp_path)


def test_download_hero_image_handles_cache_read_and_write_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Create a cached file and url file so it attempts to read it.
    out = tmp_path / "Abathur.png"
    out.write_bytes(b"x")
    url = out.with_suffix(out.suffix + ".url")
    url.write_text("old", encoding="utf-8")

    buf = io.BytesIO()
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class Resp:
        content = png_bytes

        def raise_for_status(self):
            return None

    monkeypatch.setattr(hero_img.requests, "get", lambda *_a, **_k: Resp())

    orig_read_text = Path.read_text
    orig_write_text = Path.write_text

    def read_text_raises(self: Path, *a, **k):
        if self.suffix == ".url":
            raise OSError("no read")
        return orig_read_text(self, *a, **k)

    def write_text_raises(self: Path, *a, **k):
        if self.suffix == ".url":
            raise OSError("no write")
        return orig_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text_raises)
    monkeypatch.setattr(Path, "write_text", write_text_raises)

    hero_map = {hero_img._normalize_hero_name("Abathur"): hero_img.HeroAsset("Abathur", "https://cdn/x_card_portrait.webp")}
    # Should still return a path (may overwrite the PNG).
    p = hero_img.download_hero_image("Abathur", hero_map, out_dir=tmp_path, force_redownload=True)
    assert p.exists()


def test_get_or_download_returns_none_on_internal_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(hero_img, "scrape_hero_image_map", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    hero_img._HERO_MAP_MEMO = None
    assert hero_img.get_or_download_hero_image_path("Abathur", out_dir=tmp_path, allow_network=True) is None


def test_cut_bottom_square_crop_handles_wide_images():
    img = Image.new("RGBA", (200, 50), (255, 0, 0, 255))
    cropped = hero_img.cut_bottom_to_quadrilateral(img, cut_ratio=0.0, square=True)
    assert cropped.size[0] == cropped.size[1]


def test_pick_best_image_url_prefers_portrait_then_icon():
    assert hero_img._pick_best_image_url("x_card_portrait.webp", "y_card_icon.webp") == "x_card_portrait.webp"
    assert hero_img._pick_best_image_url("", "y_card_portrait.webp") == "y_card_portrait.webp"
    assert hero_img._pick_best_image_url("", "") == ""


def test_load_cached_hero_map_accepts_only_portrait_urls(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(
        '{"abathur": {"name": "Abathur", "img_url": "https://cdn/abathur_card_portrait.webp"}}',
        encoding="utf-8",
    )
    data = hero_img._load_cached_hero_map(cache)
    assert data is not None
    assert hero_img._normalize_hero_name("Abathur") in data


def test_load_cached_hero_map_rejects_icon_only_cache(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(
        '{"abathur": {"name": "Abathur", "img_url": "https://cdn/abathur_card_icon.webp"}}',
        encoding="utf-8",
    )
    assert hero_img._load_cached_hero_map(cache) is None


def test_scrape_cards_via_requests_returns_empty_on_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hero_img, "_fetch_html", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert hero_img._scrape_cards_via_requests("https://example") == []


def test_scrape_cards_via_playwright_returns_empty_when_unavailable():
    # Playwright is not a runtime dependency for tests; function should fail gracefully.
    assert hero_img._scrape_cards_via_playwright("https://example", headless=True) == []


def test_scrape_hero_image_map_uses_valid_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(
        '{"abathur": {"name": "Abathur", "img_url": "https://cdn/abathur_card_portrait.webp"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(hero_img, "_scrape_cards_via_requests", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    hero_map = hero_img.scrape_hero_image_map(cache_path=cache)
    assert hero_img._normalize_hero_name("Abathur") in hero_map


def test_safe_filename_normalizes_expected_characters():
    assert hero_img._safe_filename("Lt. Morales") == "Lt_Morales"
    assert hero_img._safe_filename("  ") == "hero"


def test_download_hero_image_saves_png_and_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Generate a small input image and return it as "downloaded" bytes.
    src_img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    buf = io.BytesIO()
    src_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    calls = {"n": 0}

    class _Resp:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(_url: str, **_kw):
        calls["n"] += 1
        return _Resp(png_bytes)

    monkeypatch.setattr(hero_img.requests, "get", fake_get)

    hero_map = {
        hero_img._normalize_hero_name("Abathur"): hero_img.HeroAsset(
            name="Abathur",
            img_url="https://cdn/abathur_card_portrait.webp",
        )
    }

    out = hero_img.download_hero_image("Abathur", hero_map, out_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.with_suffix(out.suffix + ".url").exists()

    # Second call should hit the cache and avoid re-downloading.
    out2 = hero_img.download_hero_image("Abathur", hero_map, out_dir=tmp_path)
    assert out2 == out
    assert calls["n"] == 1


def test_get_or_download_returns_cached_when_present(tmp_path: Path):
    cached = tmp_path / "Abathur.png"
    cached.write_bytes(b"x")

    assert hero_img.get_or_download_hero_image_path("Abathur", out_dir=tmp_path, allow_network=False) == cached


def test_get_or_download_returns_none_when_network_disallowed(tmp_path: Path):
    assert hero_img.get_or_download_hero_image_path("Abathur", out_dir=tmp_path, allow_network=False) is None


def test_get_or_download_skips_network_during_pytest(tmp_path: Path):
    # PYTEST_CURRENT_TEST is set during runs; this should force a None return.
    assert hero_img.get_or_download_hero_image_path("Abathur", out_dir=tmp_path, allow_network=True) is None


def test_get_or_download_network_path_uses_memo_and_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Make the function think it's not running under pytest so we can cover the network path.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    hero_img._HERO_MAP_MEMO = None

    hero_map = {
        hero_img._normalize_hero_name("Abathur"): hero_img.HeroAsset(
            name="Abathur",
            img_url="https://cdn/abathur_card_portrait.webp",
        )
    }

    calls = {"scrape": 0, "download": 0}

    def fake_scrape(*_args, **_kwargs):
        calls["scrape"] += 1
        return hero_map

    def fake_download(_hero: str, _map, *, out_dir: Path, **_kw):
        calls["download"] += 1
        if calls["download"] == 1:
            raise ValueError("missing")
        out = out_dir / "Abathur.png"
        out.write_bytes(b"png")
        return out

    monkeypatch.setattr(hero_img, "scrape_hero_image_map", fake_scrape)
    monkeypatch.setattr(hero_img, "download_hero_image", fake_download)

    out = hero_img.get_or_download_hero_image_path("Abathur", out_dir=tmp_path, allow_network=True)
    assert out is not None and out.exists()
    assert calls["scrape"] == 2
    assert calls["download"] == 2
