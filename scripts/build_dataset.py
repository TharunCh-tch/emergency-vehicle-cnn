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


def list_category_files(category: str, limit: int) -> list[str]:
    """Return up to `limit` file titles from a Commons category (files only)."""
    titles: list[str] = []
    cmcontinue = None
    session = requests.Session()
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "file",
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cat = max(1, (target * 2) // len(categories))  # over-fetch, some will fail/filter
    seen_titles: set[str] = set()
    n_saved = 0
    for cat in categories:
        if n_saved >= target:
            break
        print(f"[{class_name}] category: {cat}")
        titles = list_category_files(cat, per_cat)
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
            fname = f"{class_name}_{n_saved:04d}{ext}"
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
            if n_saved % 20 == 0:
                print(f"  ...{n_saved}/{target} saved")
    print(f"[{class_name}] done: {n_saved}/{target} saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-class", type=int, default=200)
    ap.add_argument("--out-root", type=str, default="data/raw")
    ap.add_argument("--manifest", type=str, default="data/manifest.csv")
    args = ap.parse_args()

    root = Path(args.out_root)
    manifest_rows: list[dict] = []

    build_class(
        "emergency",
        POSITIVE_CATEGORIES,
        args.target_per_class,
        root / "emergency",
        manifest_rows,
    )
    build_class(
        "non_emergency",
        NEGATIVE_CATEGORIES,
        args.target_per_class,
        root / "non_emergency",
        manifest_rows,
    )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written: {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
