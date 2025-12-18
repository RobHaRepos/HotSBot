"""Scrape and cache Heroes of the Storm hero portrait images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import io
import json
import logging
import os
from pathlib import Path
import re
import unicodedata

import requests

from PIL import Image, ImageDraw

HEROES_URL_EN_GB = "https://heroesofthestorm.blizzard.com/en-gb/heroes"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "images" / "heroes"
CACHE_PATH = REPO_ROOT / "data" / "hero_image_map_cache.json"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.36"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeroAsset:
    name: str
    img_url: str


HeroMap = dict[str, HeroAsset]


_HERO_MAP_MEMO: HeroMap | None = None


class _BlzHeroesParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._in_card: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.casefold()
        attrs_dict = {k.casefold(): (v or "") for k, v in attrs}

        if t == "blz-hero-card":
            self._in_card = True
            self._current = {
                "name": attrs_dict.get("hero-name", "").strip(),
                "icon": attrs_dict.get("icon", "").strip(),
                "portrait": "",
            }
            return

        if self._in_card and self._current is not None and t == "blz-image":
            if attrs_dict.get("slot", "") != "image":
                return
            src = attrs_dict.get("src", "").strip()
            if src and _is_portrait_url(src):
                self._current["portrait"] = src

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "blz-hero-card":
            return
        if self._current is not None and self._current.get("name"):
            self.results.append(self._current)
        self._current = None
        self._in_card = False


def _is_portrait_url(url: str) -> bool:
    """Return True if url looks like a hero portrait URL."""
    return "card_portrait" in (url or "")


def _read_json_dict(path: Path) -> dict[str, object] | None:
    """Read a JSON object from disk, returning None on failure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_cached_hero_map(cache_path: Path) -> HeroMap | None:
    """Load hero map from cache if it appears valid."""
    data = _read_json_dict(cache_path)
    if not data:
        return None

    cached: HeroMap = {}
    for hero, info in data.items():
        if not isinstance(hero, str) or not isinstance(info, dict):
            continue
        name = info.get("name")
        img_url = info.get("img_url")
        if isinstance(name, str) and isinstance(img_url, str):
            cached[hero] = HeroAsset(name=name, img_url=img_url)

    # Older cache versions stored role icons; only trust portrait URLs.
    if any(_is_portrait_url(v.img_url) for v in cached.values()):
        return cached
    return None


def _pick_best_image_url(portrait: str, icon: str) -> str:
    """Pick the most portrait-like URL from portrait/icon candidates."""
    portrait = (portrait or "").strip()
    icon = (icon or "").strip()
    if _is_portrait_url(portrait):
        return portrait
    if _is_portrait_url(icon):
        return icon
    return ""


def _extract_hero_cards_from_html(html: str) -> list[dict[str, str]]:
    """Extract hero-name and portrait/icon URLs from the heroes page HTML."""
    parser = _BlzHeroesParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # If the markup is malformed, return whatever was collected.
        logger.debug("HTMLParser failed while extracting hero cards", exc_info=True)
    return parser.results


def _fetch_html(url: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Fetch HTML from the given URL."""
    response = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=timeout)
    response.raise_for_status()
    return response.text


def _scrape_cards_via_requests(heroes_url: str) -> list[dict[str, str]]:
    """Scrape hero cards by requesting raw HTML."""
    try:
        return _extract_hero_cards_from_html(_fetch_html(heroes_url))
    except Exception:
        logger.exception("Failed to scrape hero cards via requests")
        return []


def _scrape_cards_via_playwright(heroes_url: str, *, headless: bool) -> list[dict[str, str]]:
    """Scrape hero cards via Playwright rendering (fallback)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            try:
                context = browser.new_context(user_agent=DEFAULT_UA, locale="de-DE")
                try:
                    page = context.new_page()
                    page.goto(heroes_url, wait_until="networkidle", timeout=60_000)
                    html = page.content()
                finally:
                    context.close()
            finally:
                browser.close()
        return _extract_hero_cards_from_html(html)
    except Exception:
        logger.exception("Failed to scrape hero cards via Playwright")
        return []


def _build_hero_map(results: list[dict[str, str]]) -> HeroMap:
    """Build a normalized-name->asset map from extracted entries."""
    hero_map: HeroMap = {}
    for entry in results:
        name = (entry.get("name") or "").strip()
        portrait = (entry.get("portrait") or "").strip()
        icon = (entry.get("icon") or "").strip()

        if not name:
            continue
        img_url = _pick_best_image_url(portrait, icon)
        if not img_url:
            continue

        hero_map[_normalize_hero_name(name)] = HeroAsset(name=name, img_url=img_url)
    return hero_map


def _normalize_hero_name(hero_name: str) -> str:
    """Normalize a hero name for matching."""
    hero_name = hero_name.strip().casefold()
    hero_name = unicodedata.normalize("NFKD", hero_name)
    hero_name = "".join(char for char in hero_name if not unicodedata.combining(char))
    hero_name = re.sub(r"[^a-z0-9]", " ", hero_name).strip()
    return hero_name


def _safe_filename(s: str) -> str:
    """Convert a string to a safe filename."""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    return s or "hero"


def scrape_hero_image_map(
    heroes_url: str = HEROES_URL_EN_GB,
    *,
    cache_path: Path | None = CACHE_PATH,
    headless: bool = True,
) -> HeroMap:
    """Scrape hero names and image URLs."""
    if cache_path and cache_path.exists():
        cached = _load_cached_hero_map(cache_path)
        if cached is not None:
            return cached

    results = _scrape_cards_via_requests(heroes_url)
    if not results:
        results = _scrape_cards_via_playwright(heroes_url, headless=headless)

    hero_map = _build_hero_map(results)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {hero: asdict(hero_asset) for hero, hero_asset in hero_map.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return hero_map


def download_hero_image(
    hero_name: str,
    hero_map: HeroMap,
    *,
    out_dir: Path = OUTPUT_DIR,
    timeout: int = 25,
    force_redownload: bool = False,
) -> Path:
    """Download a hero image and save it locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hero = _normalize_hero_name(hero_name)
    if hero not in hero_map:
        raise ValueError(f"Hero '{hero_name}' not found in hero map.")

    asset = hero_map[hero]
    file_name = _safe_filename(asset.name) + ".png"
    output_path = out_dir / file_name
    source_url_path = output_path.with_suffix(output_path.suffix + ".url")

    if output_path.exists() and not force_redownload and source_url_path.exists():
        try:
            if source_url_path.read_text(encoding="utf-8").strip() == asset.img_url:
                return output_path
        except Exception:
            logger.debug("Failed to read %s", source_url_path, exc_info=True)

    response = requests.get(asset.img_url, headers={"User-Agent": DEFAULT_UA}, timeout=timeout)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content)).convert("RGBA")
    img = cut_bottom_to_quadrilateral(img, cut_ratio=0.18)
    img = round_corners(img, radius_ratio=0.10)
    img.save(output_path, format="PNG")

    try:
        source_url_path.write_text(asset.img_url, encoding="utf-8")
    except Exception:
        logger.debug("Failed to write %s", source_url_path, exc_info=True)

    return output_path


def get_cached_hero_image_path(hero_name: str, *, out_dir: Path = OUTPUT_DIR) -> Path | None:
    """Return the local cached hero portrait path if present."""
    file_name = _safe_filename(hero_name) + ".png"
    path = out_dir / file_name
    return path if path.exists() else None


def get_or_download_hero_image_path(
    hero_name: str,
    *,
    out_dir: Path = OUTPUT_DIR,
    allow_network: bool = True,
) -> Path | None:
    """Return a cached portrait path or download it when allowed."""
    cached = get_cached_hero_image_path(hero_name, out_dir=out_dir)
    if cached is not None:
        return cached

    if not allow_network:
        return None
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None

    global _HERO_MAP_MEMO
    try:
        if _HERO_MAP_MEMO is None:
            _HERO_MAP_MEMO = scrape_hero_image_map()
        try:
            return download_hero_image(hero_name, _HERO_MAP_MEMO, out_dir=out_dir)
        except ValueError:
            # Hero might be missing due to locale/cached data; retry with en-gb.
            _HERO_MAP_MEMO = scrape_hero_image_map(heroes_url=HEROES_URL_EN_GB, cache_path=None)
            return download_hero_image(hero_name, _HERO_MAP_MEMO, out_dir=out_dir)
    except Exception:
        logger.exception("Failed to resolve hero image for %s", hero_name)
        return None


def round_corners(img: Image.Image, *, radius_ratio: float = 0.10) -> Image.Image:
    """Round the corners of an RGBA image by applying an alpha mask."""
    img = img.convert("RGBA")
    w, h = img.size
    radius = max(1, int(min(w, h) * float(radius_ratio)))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)

    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def cut_bottom_to_quadrilateral(
    img: Image.Image,
    cut_ratio: float = 0.18,
    *,
    top_crop_ratio: float = 0.02,
    bottom_crop_ratio: float = 0.08,
    square: bool = True,
    square_bias_bottom: float = 0.60,
) -> Image.Image:
    """Crop and square a portrait to a consistent rectangular tile."""
    img = img.convert("RGBA")
    w, h = img.size

    side_cut = max(0, int(w * cut_ratio))
    top_cut = max(0, int(h * top_crop_ratio))
    bottom_cut = max(0, int(h * bottom_crop_ratio))

    # Ensure we always return a non-empty crop.
    left = min(side_cut, w - 2)
    right = max(left + 1, w - side_cut)
    top = min(top_cut, h - 2)
    bottom = max(top + 1, h - bottom_cut)

    cropped = img.crop((left, top, right, bottom))

    # Prefer cutting more off the bottom than the top.
    if square:
        cw, ch = cropped.size
        target = min(cw, ch)
        if ch > target:
            excess = ch - target
            bias = min(0.95, max(0.05, float(square_bias_bottom)))
            bottom_remove = int(round(excess * bias))
            top_remove = excess - bottom_remove
            cropped = cropped.crop((0, top_remove, cw, ch - bottom_remove))
        elif cw > target:
            # Defensive: should not happen with current pipeline, but keep stable.
            excess = cw - target
            left_remove = excess // 2
            right_remove = excess - left_remove
            cropped = cropped.crop((left_remove, 0, cw - right_remove, ch))

    return cropped
    
if __name__ == "__main__":  # pragma: no cover
    hero_map = scrape_hero_image_map(headless=True)
    print(f"Scraped {len(hero_map)} heroes.")
    for probe in ["Abathur", "Alarak", "Zeratul", "Lt. Morales"]:
        hero = _normalize_hero_name(probe)
        if hero in hero_map:
            print(probe, "->", hero_map[hero].img_url)