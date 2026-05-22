"""Populate the Nobel Physics + Fields Medalist corpora from Wikipedia.

Fetches plain-text extracts of curated articles, saves them to the dataset
cache (``~/.cache/blowfish/corpora/<name>/``), computes SHA-256s, and rewrites
the manifest JSONs at ``blowfish/datasets/data/{nobel_physics,fields_medalists}.json``
so the corpus loaders find pre-verified files on disk.

Wikipedia extracts are licensed CC-BY-SA 4.0; attribution is preserved by
recording the source URL + ``oldid`` (immutable revision id) in each manifest
entry. The downloaded text is NOT committed to the repo — the cache directory
is gitignored.

Usage::

    pip install -e ".[datasets]"
    python scripts/bootstrap_corpora.py

To rotate the corpus to real lecture transcripts (the Wikipedia articles are
a proxy seed), edit the ``CURATED_*`` lists below and re-run, or replace the
manifest entries by hand with ``{doc_id, title, mirror_urls, sha256, license}``
pointing at stable lecture archives.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests

from blowfish.datasets.cache import cache_root


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "blowfish" / "datasets" / "data"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
WIKINEWS_API = "https://en.wikinews.org/w/api.php"
USER_AGENT = "open-blowfish bootstrap (https://github.com/thomasbarillot/open-blowfish)"


# Curated seed lists. Each tuple is (doc_id, wikipedia title, year, scholar).
CURATED_NOBEL_PHYSICS: list[tuple[str, str, int, str]] = [
    ("einstein_1921", "Albert Einstein", 1921, "Albert Einstein"),
    ("dirac_1933", "Paul Dirac", 1933, "Paul Dirac"),
    ("feynman_1965", "Richard Feynman", 1965, "Richard Feynman"),
    ("chandrasekhar_1983", "Subrahmanyan Chandrasekhar", 1983, "Subrahmanyan Chandrasekhar"),
    ("weinberg_1979", "Steven Weinberg", 1979, "Steven Weinberg"),
    ("penrose_2020", "Roger Penrose", 2020, "Roger Penrose"),
    ("ghez_2020", "Andrea M. Ghez", 2020, "Andrea Ghez"),
    ("strickland_2018", "Donna Strickland", 2018, "Donna Strickland"),
]

CURATED_FIELDS_MEDALISTS: list[tuple[str, str, int, str]] = [
    ("tao_2006", "Terence Tao", 2006, "Terence Tao"),
    ("perelman_2006", "Grigori Perelman", 2006, "Grigori Perelman"),
    ("mirzakhani_2014", "Maryam Mirzakhani", 2014, "Maryam Mirzakhani"),
    ("villani_2010", "Cédric Villani", 2010, "Cédric Villani"),
    ("bhargava_2014", "Manjul Bhargava", 2014, "Manjul Bhargava"),
    ("venkatesh_2018", "Akshay Venkatesh", 2018, "Akshay Venkatesh"),
    ("birkar_2018", "Caucher Birkar", 2018, "Caucher Birkar"),
    ("viazovska_2022", "Maryna Viazovska", 2022, "Maryna Viazovska"),
]


# Britannica 1911 — public-domain articles transcribed on Wikisource. The
# article title convention is ``1911 Encyclopædia Britannica/<Article>``.
CURATED_BRITANNICA_1911: list[tuple[str, str, str]] = [
    ("brit1911_algebra", "1911 Encyclopædia Britannica/Algebra", "Algebra"),
    ("brit1911_astronomy", "1911 Encyclopædia Britannica/Astronomy", "Astronomy"),
    ("brit1911_geology", "1911 Encyclopædia Britannica/Geology", "Geology"),
    ("brit1911_logic", "1911 Encyclopædia Britannica/Logic", "Logic"),
    ("brit1911_mathematics", "1911 Encyclopædia Britannica/Mathematics", "Mathematics"),
    ("brit1911_physics", "1911 Encyclopædia Britannica/Physics", "Physics"),
    ("brit1911_newton", "1911 Encyclopædia Britannica/Newton, Sir Isaac", "Newton, Sir Isaac"),
    ("brit1911_aeronautics", "1911 Encyclopædia Britannica/Aeronautics", "Aeronautics"),
]


# Wikinews — open news (CC-BY 2.5). Fetched by recent category membership rather
# than hardcoded titles, since news article URLs are not as stable as Wikipedia
# bios; we pin via per-fetch oldid so subsequent re-fetches verify the same revision.
WIKINEWS_CATEGORIES_AND_TARGET_COUNTS: list[tuple[str, int]] = [
    ("Economy_and_business", 4),
    ("Politics_and_conflicts", 4),
]


# Chronicling America (LoC) — open API; public-domain US newspapers 1900–1925.
# Closest open analog to "old WSJ / FT" — picks period financial pages.
CHRONICLING_AMERICA_API = "https://chroniclingamerica.loc.gov/search/pages/results/"
CHRONICLING_AMERICA_QUERY = "stocks securities banking investors"
CHRONICLING_AMERICA_DATE_RANGE = (1900, 1925)
CHRONICLING_AMERICA_TARGET_COUNT = 8


# cc_news — HuggingFace dataset, modern news, filtered to financial outlets.
CC_NEWS_TARGET_COUNT = 8
CC_NEWS_DOMAIN_FILTERS = ("wsj.com", "ft.com", "bloomberg.com", "reuters.com")


def fetch_extract(api_url: str, title: str) -> tuple[str, int]:
    """Fetch plain-text extract + current oldid for an article on any MediaWiki host."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|revisions",
        "explaintext": "1",
        "exsectionformat": "plain",
        "rvprop": "ids",
        "redirects": "1",
    }
    response = requests.get(
        api_url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page or not page["extract"].strip():
        raise RuntimeError(f"no extract returned for {title!r} at {api_url}: {page!r}")
    extract = page["extract"]
    oldid = int(page["revisions"][0]["revid"])
    return extract, oldid


def _strip_wikitext(text: str) -> str:
    """Best-effort wikitext → readable plain text. Removes templates, file/
    category links, formatting markers, and HTML tags. Suited for Wikisource
    transcribed articles where the ``prop=extracts`` API returns empty."""
    # Drop {{...}} templates (run twice to handle one level of nesting).
    for _ in range(2):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Drop wikitable {| ... |}.
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # Drop ref / noinclude / pages markup.
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"<noinclude>.*?</noinclude>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?pages[^>]*/?>", "", text)
    # File and Category links.
    text = re.sub(r"\[\[(?:Category|File|Image):[^\[\]]*\]\]", "", text)
    # [[Target|Display]] → Display; [[Target]] → Target.
    text = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    # ''' bold ''' and '' italic ''.
    text = re.sub(r"'''([^']*)'''", r"\1", text)
    text = re.sub(r"''([^']*)''", r"\1", text)
    # Remaining HTML tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse 3+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_wikitext_cleaned(api_url: str, title: str) -> tuple[str, int]:
    """Fetch raw wikitext for ``title`` and strip markup. Used for sites where
    ``prop=extracts`` returns empty (e.g. Wikisource transcribed pages)."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content|ids",
        "rvslots": "main",
        "redirects": "1",
    }
    response = requests.get(
        api_url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "revisions" not in page:
        raise RuntimeError(f"no revisions for {title!r} at {api_url}: {page!r}")
    rev = page["revisions"][0]
    wikitext = rev["slots"]["main"]["*"]
    cleaned = _strip_wikitext(wikitext)
    if len(cleaned) < 200:
        raise RuntimeError(
            f"cleaned wikitext too short for {title!r} ({len(cleaned)} chars)"
        )
    return cleaned, int(rev["revid"])


def list_category_members(api_url: str, category: str, limit: int = 20) -> list[str]:
    """Return article titles under ``Category:<category>`` (most recent first)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": str(limit),
        "cmtype": "page",
        "format": "json",
        "cmsort": "timestamp",
        "cmdir": "desc",
    }
    response = requests.get(
        api_url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    response.raise_for_status()
    return [m["title"] for m in response.json()["query"]["categorymembers"]]


def stable_url(api_url: str, title: str, oldid: int) -> str:
    safe_title = title.replace(" ", "_")
    return (
        f"{api_url}?action=query&format=json&prop=extracts&explaintext=1"
        f"&titles={safe_title}&oldid={oldid}"
    )


def _write_manifest(corpus_name: str, manifest_entries: list[dict]) -> None:
    manifest_path = MANIFEST_DIR / f"{corpus_name}.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["documents"] = manifest_entries
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  → wrote {len(manifest_entries)} entries to {manifest_path}")


def bootstrap_wikipedia_scholars(
    corpus_name: str,
    entries: list[tuple[str, str, int, str]],
    *,
    scholar_field_name: str,
) -> None:
    target_dir = cache_root() / "corpora" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    print(f"\n=== bootstrapping {corpus_name} ({len(entries)} entries; source: Wikipedia) ===")
    for doc_id, title, year, scholar in entries:
        print(f"  {title!r}...", end=" ", flush=True)
        try:
            text, oldid = fetch_extract(WIKIPEDIA_API, title)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (target_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest_entries.append(
            {
                "doc_id": doc_id,
                "title": title,
                "year": year,
                scholar_field_name: scholar,
                "mirror_urls": [stable_url(WIKIPEDIA_API, title, oldid)],
                "sha256": sha,
                "license": "CC-BY-SA-4.0",
                "source": "wikipedia",
                "oldid": oldid,
            }
        )
        print(f"OK ({len(text):,} chars, sha {sha[:8]})")
    _write_manifest(corpus_name, manifest_entries)
    print(f"  → text cached at {target_dir}")


def bootstrap_britannica_1911() -> None:
    corpus_name = "britannica_1911"
    target_dir = cache_root() / "corpora" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    print(f"\n=== bootstrapping {corpus_name} (source: Wikisource — public domain) ===")
    for doc_id, title, article in CURATED_BRITANNICA_1911:
        print(f"  {title!r}...", end=" ", flush=True)
        try:
            text, oldid = fetch_wikitext_cleaned(WIKISOURCE_API, title)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            continue
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (target_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest_entries.append(
            {
                "doc_id": doc_id,
                "title": title,
                "article": article,
                "mirror_urls": [stable_url(WIKISOURCE_API, title, oldid)],
                "sha256": sha,
                "license": "public-domain",
                "source": "wikisource",
                "edition": "1911 Encyclopædia Britannica",
                "oldid": oldid,
            }
        )
        print(f"OK ({len(text):,} chars, sha {sha[:8]})")
    _write_manifest(corpus_name, manifest_entries)
    print(f"  → text cached at {target_dir}")


def bootstrap_wikinews() -> None:
    corpus_name = "wikinews"
    target_dir = cache_root() / "corpora" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    print(f"\n=== bootstrapping {corpus_name} (source: Wikinews — CC-BY 2.5) ===")
    seen_ids: set[str] = set()
    for category, target_count in WIKINEWS_CATEGORIES_AND_TARGET_COUNTS:
        print(f"  category {category!r}: discovering up to {target_count} articles...")
        try:
            titles = list_category_members(WIKINEWS_API, category, limit=target_count * 3)
        except Exception as exc:  # noqa: BLE001
            print(f"    discovery FAILED: {exc}")
            continue
        fetched = 0
        for title in titles:
            if fetched >= target_count:
                break
            doc_id = (
                "wn_"
                + category.lower().replace("_", "")
                + "_"
                + "".join(c if c.isalnum() else "_" for c in title)[:48].strip("_").lower()
            )
            if doc_id in seen_ids:
                continue
            print(f"    {title!r}...", end=" ", flush=True)
            try:
                text, oldid = fetch_extract(WIKINEWS_API, title)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {exc}")
                continue
            if len(text) < 200:
                print("SKIP (<200 chars)")
                continue
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            (target_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            manifest_entries.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "mirror_urls": [stable_url(WIKINEWS_API, title, oldid)],
                    "sha256": sha,
                    "license": "CC-BY-2.5",
                    "source": "wikinews",
                    "oldid": oldid,
                }
            )
            seen_ids.add(doc_id)
            fetched += 1
            print(f"OK ({len(text):,} chars, sha {sha[:8]})")
    _write_manifest(corpus_name, manifest_entries)
    print(f"  → text cached at {target_dir}")


def bootstrap_chronicling_america_finance() -> None:
    corpus_name = "chronicling_america_finance"
    target_dir = cache_root() / "corpora" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    print(
        f"\n=== bootstrapping {corpus_name} (source: LoC Chronicling America "
        f"{CHRONICLING_AMERICA_DATE_RANGE[0]}-{CHRONICLING_AMERICA_DATE_RANGE[1]}) ==="
    )
    params = {
        "proxtext": CHRONICLING_AMERICA_QUERY,
        "format": "json",
        "dateFilterType": "yearRange",
        "date1": str(CHRONICLING_AMERICA_DATE_RANGE[0]),
        "date2": str(CHRONICLING_AMERICA_DATE_RANGE[1]),
        "rows": str(CHRONICLING_AMERICA_TARGET_COUNT * 3),
    }
    try:
        response = requests.get(
            CHRONICLING_AMERICA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception as exc:  # noqa: BLE001
        print(f"  search FAILED: {exc}")
        _write_manifest(corpus_name, [])
        return
    fetched = 0
    for item in items:
        if fetched >= CHRONICLING_AMERICA_TARGET_COUNT:
            break
        text = (item.get("ocr_eng") or "").strip()
        if len(text) < 500:
            continue
        lccn = item.get("lccn", "unknown")
        date = item.get("date", "unknown")
        seq = item.get("sequence", 0)
        doc_id = f"loc_{lccn}_{date}_{seq:03d}".replace("/", "_")[:80]
        title = item.get("title_normal") or item.get("title") or "Unknown"
        item_url = item.get("url") or item.get("id", "")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (target_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest_entries.append(
            {
                "doc_id": doc_id,
                "title": f"{title} ({date}, p.{seq})",
                "lccn": lccn,
                "date": date,
                "sequence": seq,
                "mirror_urls": [item_url] if item_url else [],
                "sha256": sha,
                "license": "public-domain",
                "source": "chronicling_america",
            }
        )
        fetched += 1
        print(f"  {doc_id}... OK ({len(text):,} chars, sha {sha[:8]})")
    _write_manifest(corpus_name, manifest_entries)
    print(f"  → text cached at {target_dir}")


def bootstrap_cc_news_finance() -> None:
    corpus_name = "cc_news_finance"
    target_dir = cache_root() / "corpora" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== bootstrapping {corpus_name} (source: HuggingFace cc_news, "
        f"filtered to {CC_NEWS_DOMAIN_FILTERS}) ==="
    )
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "  SKIP: requires HuggingFace `datasets` library. "
            "Install with `pip install datasets` to bootstrap this corpus."
        )
        _write_manifest(corpus_name, [])
        return
    try:
        ds = load_dataset("cc_news", streaming=True, split="train")
    except Exception as exc:  # noqa: BLE001
        print(f"  load_dataset FAILED: {exc}")
        _write_manifest(corpus_name, [])
        return
    manifest_entries: list[dict] = []
    fetched = 0
    for i, example in enumerate(ds):
        if fetched >= CC_NEWS_TARGET_COUNT:
            break
        if i > 200_000:  # streaming safety cap
            print(f"  scanned {i:,} examples, stopping with {fetched} matches")
            break
        url = (example.get("url") or "").lower()
        if not any(d in url for d in CC_NEWS_DOMAIN_FILTERS):
            continue
        text = (example.get("text") or "").strip()
        if len(text) < 500:
            continue
        title = (example.get("title") or "").strip() or f"cc_news entry #{i}"
        domain = next(d for d in CC_NEWS_DOMAIN_FILTERS if d in url)
        doc_id = (
            f"ccn_{domain.replace('.com', '')}_{i:06d}_"
            + "".join(c if c.isalnum() else "_" for c in title)[:48].strip("_").lower()
        )
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (target_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest_entries.append(
            {
                "doc_id": doc_id,
                "title": title,
                "domain": domain,
                "url": example.get("url", ""),
                "date": example.get("date") or example.get("date_publish") or "",
                "mirror_urls": [example.get("url", "")] if example.get("url") else [],
                "sha256": sha,
                "license": "varies",
                "source": "cc_news",
            }
        )
        fetched += 1
        print(f"  [{i:6d}] {domain} :: {title[:60]!r}... OK ({len(text):,} chars)")
    _write_manifest(corpus_name, manifest_entries)
    print(f"  → text cached at {target_dir}")


def main() -> int:
    bootstrap_wikipedia_scholars(
        "nobel_physics", CURATED_NOBEL_PHYSICS, scholar_field_name="laureate"
    )
    bootstrap_wikipedia_scholars(
        "fields_medalists", CURATED_FIELDS_MEDALISTS, scholar_field_name="medalist"
    )
    bootstrap_britannica_1911()
    bootstrap_wikinews()
    bootstrap_chronicling_america_finance()
    bootstrap_cc_news_finance()
    print(
        "\nDone. Verify with:\n"
        "  python -c 'from blowfish.datasets import list_corpora, Corpus;"
        " print({n: len(list(Corpus(n).iter_documents())) for n in list_corpora()})'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
