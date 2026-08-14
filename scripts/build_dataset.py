"""
Assemble a small, legitimately-licensed emergency vs. non-emergency vehicle
image dataset from Wikimedia Commons category pages.

This is NOT the original Kaggle dataset used in the LinkedIn-described
coursework project. There is no Kaggle API credential configured on this
machine, so this script builds an honest, smaller, CPU-trainable substitute
by pulling images directly from Wikimedia Commons categories, which publish
structured metadata (license, author, source URL) per file.

Positive class (emergency vehicles): Police vehicles, Ambulances, Fire engines
Negative class (non-emergency vehicles): Sedans, SUVs, Pickup trucks, Vans

Usage:
    python scripts/build_dataset.py --target-per-class 200

Writes:
    data/raw/emergency/*.jpg
    data/raw/non_emergency/*.jpg
    data/manifest.csv   (filename, class, wikimedia title, source url,
                          license short name, artist/credit, page url)
"""
from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

import requests
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "emergency-vehicle-cnn-dataset-builder/1.0 "
        "(portfolio project; contact: praneeth.bojanala555@gmail.com)"
    )
}

POSITIVE_CATEGORIES = ["Police vehicles", "Ambulances", "Fire engines"]
NEGATIVE_CATEGORIES = ["Sedans", "SUVs", "Pickup trucks", "Vans"]

VALID_EXT = {".jpg", ".jpeg", ".png"}
THUMB_WIDTH = 512  # fetch a resized thumbnail directly from Commons to save bandwidth
MIN_SIDE = 150      # skip tiny icons/diagrams


def _list_members(category: str, cmtype: str, limit: int, session: requests.Session) -> list[str]:
    titles: list[str] = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": cmtype,
            "cmlimit": min(50, limit - len(titles)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue or not members:
            break
        time.sleep(0.2)
    return titles[:limit]


def list_category_files(category: str, limit: int) -> list[str]:
    """Return up to `limit` file titles from a Commons category (files only)."""
    session = requests.Session()
    return _list_members(category, "file", limit, session)


def list_category_files_recursive(
    category: str, limit: int, max_depth: int = 2, max_subcats: int = 40
) -> list[str]:
    """Many Commons vehicle categories (e.g. "Category:Sedans") hold almost no
    files directly -- the files live under subcategories like "by brand" ->
    "<Manufacturer> <Model>". BFS through subcategories (bounded depth/fanout)
    collecting file titles until `limit` is reached or the frontier is
    exhausted.
    """
    session = requests.Session()
    titles: list[str] = []
    seen_cats: set[str] = {category}
    frontier: list[tuple[str, int]] = [(category, 0)]

    while frontier and len(titles) < limit:
        cat, depth = frontier.pop(0)
        direct = _list_members(cat, "file", limit - len(titles), session)
        titles.extend(direct)
        if len(titles) >= limit:
            break
        if depth >= max_depth:
            continue
        subcats = _list_members(cat, "subcat", max_subcats, session)
        for sc in subcats:
            name = sc.split(":", 1)[1] if ":" in sc else sc
            if name not in seen_cats:
                seen_cats.add(name)
                frontier.append((name, depth + 1))

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:limit]


def fetch_imageinfo(titles: list[str]) -> dict:
    """Batch-fetch imageinfo (url, license, artist) for up to 50 titles at a time."""
    out = {}
    session = requests.Session()
    for i in range(0, len(titles), 50):
        batch = titles[i : i + 50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": THUMB_WIDTH,
            "format": "json",
        }
        r = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = page.get("title")
            infos = page.get("imageinfo")
            if not title or not infos:
                continue
            out[title] = infos[0]
        time.sleep(0.2)
    return out


def download_and_save(url: str, dest: Path, max_retries: int = 5) -> tuple[int, int] | None:
    for attempt in range(max_retries):
        try:
            time.sleep(0.6)  # proactive throttle to avoid Commons rate limits
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  429 rate-limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            break
        except Exception as e:  # noqa: BLE001 - best-effort scraping, skip failures
            if attempt == max_retries - 1:
                print(f"  skip (download/decode failed): {url} ({e})")
                return None
            time.sleep(2 * (attempt + 1))
    else:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    img.save(dest, quality=90)
    time.sleep(0.5)  # be polite to Commons servers
    return img.size


def build_class(
    class_name: str,
    categories: list[str],
    target: int,
    out_dir: Path,
    manifest_rows: list[dict],
    start_index: int = 0,
    existing_titles: set[str] | None = None,
) -> int:
    """Download up to `target` NEW images for this class, numbered starting at
    `start_index` (so re-running to top up a class doesn't clobber existing
    files). Returns the number of new images saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cat = max(20, (target * 3) // max(1, len(categories)))  # over-fetch, some will fail/filter
    seen_titles: set[str] = set(existing_titles or set())
    n_saved = 0
    idx = start_index
    for cat in categories:
        if n_saved >= target:
            break
        print(f"[{class_name}] category: {cat}")
        titles = list_category_files_recursive(cat, per_cat)
        infos = fetch_imageinfo(titles)
        for title, info in infos.items():
            if n_saved >= target:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            ext = ".jpg" if "jpeg" in mime else ".png"
            fname = f"{class_name}_{idx:04d}{ext}"
            dest = out_dir / fname
            size = download_and_save(url, dest)
            if size is None:
                continue
            meta = info.get("extmetadata", {})
            license_short = meta.get("LicenseShortName", {}).get("value", "unknown")
            artist = meta.get("Artist", {}).get("value", "unknown")
            # strip any HTML from artist field crudely
            import re

            artist = re.sub("<[^<]+?>", "", artist or "")
            manifest_rows.append(
                {
                    "filename": fname,
                    "class": class_name,
                    "wikimedia_title": title,
                    "source_url": info.get("descriptionurl", ""),
                    "image_url": url,
                    "license": license_short,
                    "artist": artist.strip()[:200],
                    "width": size[0],
                    "height": size[1],
                    "category": cat,
                }
            )
            n_saved += 1
            idx += 1
            if n_saved % 20 == 0:
                print(f"  ...{n_saved}/{target} new images saved")
    print(f"[{class_name}] done: {n_saved}/{target} new images saved (next index {idx})")
    return n_saved


FIELDNAMES = [
    "filename",
    "class",
    "wikimedia_title",
    "source_url",
    "image_url",
    "license",
    "artist",
    "width",
    "height",
    "category",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-class", type=int, default=200)
    ap.add_argument("--out-root", type=str, default="data/raw")
    ap.add_argument("--manifest", type=str, default="data/manifest.csv")
    ap.add_argument(
        "--classes",
        type=str,
        default="emergency,non_emergency",
        help="comma-separated subset of classes to (re)fetch, e.g. 'non_emergency' to top up one class",
    )
    ap.add_argument(
        "--top-up",
        action="store_true",
        help="append new images to an existing manifest/data dir instead of overwriting",
    )
    args = ap.parse_args()

    root = Path(args.out_root)
    manifest_path = Path(args.manifest)
    manifest_rows: list[dict] = []
    existing_by_class: dict[str, tuple[int, set[str]]] = {}

    if args.top_up and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_rows.append(row)
        for cls in ("emergency", "non_emergency"):
            rows = [r for r in manifest_rows if r["class"] == cls]
            titles = {r["wikimedia_title"] for r in rows}
            max_idx = -1
            for r in rows:
                try:
                    n = int(Path(r["filename"]).stem.rsplit("_", 1)[1])
                    max_idx = max(max_idx, n)
                except (IndexError, ValueError):
                    pass
            existing_by_class[cls] = (max_idx + 1, titles)

    requested = set(args.classes.split(","))

    if "emergency" in requested:
        start_idx, existing_titles = existing_by_class.get("emergency", (0, set()))
        needed = max(0, args.target_per_class - start_idx)
        build_class(
            "emergency",
            POSITIVE_CATEGORIES,
            needed,
            root / "emergency",
            manifest_rows,
            start_index=start_idx,
            existing_titles=existing_titles,
        )
    if "non_emergency" in requested:
        start_idx, existing_titles = existing_by_class.get("non_emergency", (0, set()))
        needed = max(0, args.target_per_class - start_idx)
        build_class(
            "non_emergency",
            NEGATIVE_CATEGORIES,
            needed,
            root / "non_emergency",
            manifest_rows,
            start_index=start_idx,
            existing_titles=existing_titles,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
