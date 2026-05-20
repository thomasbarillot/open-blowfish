from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .. import config

logger = logging.getLogger(__name__)


class CourtListenerError(RuntimeError):
    pass


def _headers():
    token = os.environ.get("COURTLISTENER_API_TOKEN", "af68b3e6f2b5512fb8eb2a1563738cee123c6657")
    h = {"User-Agent": "avantia-legal-rag-repro/0.1"}
    if token:
        h["Authorization"] = f"Token {token}"
    return h


_MAX_RETRIES = 8
_BASE_BACKOFF = 2.0


def _request(method: str, url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("headers", _headers())
    kwargs.setdefault("timeout", 90)
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            delay = min(_BASE_BACKOFF ** attempt, 120)
            logger.info("Connection error, retrying in %.1fs (attempt %d/%d): %s",
                        delay, attempt + 1, _MAX_RETRIES, e)
            time.sleep(delay)
            continue
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if r.status_code == 502:
                delay = 1.0
            else:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = _BASE_BACKOFF ** attempt
                else:
                    delay = _BASE_BACKOFF ** attempt
                delay = min(delay, 120)
            logger.info("Rate-limited (%s), retrying in %.1fs (attempt %d/%d)",
                        r.status_code, delay, attempt + 1, _MAX_RETRIES)
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r
    if last_exc:
        raise CourtListenerError(f"Repeated failures on {url}") from last_exc
    raise CourtListenerError(f"Repeated transient failures on {url}")


def _get(url: str, params: dict | None = None) -> requests.Response:
    return _request("GET", url, params=params)


def _post(url: str, json: dict | None = None) -> requests.Response:
    return _request("POST", url, json=json)


def lookup_citation(citation: str) -> int | None:
    """Look up a normalized citation via the v4 POST text endpoint."""
    url = f"{config.COURTLISTENER_API_BASE}/citation-lookup/"
    try:
        r = _post(url, json={"text": citation})
    except (requests.HTTPError, CourtListenerError) as e:
        logger.debug("citation-lookup failed for %s: %s", citation, e)
        return None

    results = r.json()
    if not results:
        return None

    for entry in results:
        for cluster in entry.get("clusters") or []:
            sub_opinions = cluster.get("sub_opinions") or []
            if sub_opinions:
                first_op = sub_opinions[0]
                op_id = _id_from_url(first_op) if isinstance(first_op, str) else first_op.get("id")
                if op_id is not None:
                    return int(op_id)
    return None


def _id_from_url(url: str) -> int | None:
    parts = url.rstrip("/").split("/")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


def fetch_opinion(opinion_id: int) -> dict | None:
    url = f"{config.COURTLISTENER_API_BASE}/opinions/{opinion_id}/"
    r = _get(url)
    body = r.json()
    text = body.get("plain_text") or body.get("html_with_citations") or ""
    if not text.strip():
        return None

    court = ""
    date_filed = ""
    all_cits: list[str] = []

    cluster_ref = body.get("cluster")
    if isinstance(cluster_ref, str):
        try:
            cluster = _get(cluster_ref).json()
        except requests.HTTPError:
            cluster = {}
    elif isinstance(cluster_ref, dict):
        cluster = cluster_ref
    else:
        cluster = {}

    date_filed = cluster.get("date_filed", "")

    docket_ref = cluster.get("docket")
    if isinstance(docket_ref, str):
        try:
            docket = _get(docket_ref).json()
        except requests.HTTPError:
            docket = {}
    elif isinstance(docket_ref, dict):
        docket = docket_ref
    else:
        docket = {}
    court = docket.get("court_id") or docket.get("court", "")

    citations = cluster.get("citations") or []
    for c in citations:
        if isinstance(c, str):
            try:
                c = _get(c).json()
            except requests.HTTPError:
                continue
        vol = c.get("volume", "")
        rep = c.get("reporter", "")
        pg = c.get("page", "")
        if vol and rep and pg:
            all_cits.append(f"{vol} {rep} {pg}")

    primary = all_cits[0] if all_cits else ""
    return {
        "opinion_id": str(body["id"]),
        "text": text,
        "court": court,
        "date_filed": date_filed,
        "primary_citation": primary,
        "normalized_citation": primary,
        "all_citations": all_cits,
    }


def fetch_bulk_court_opinions(court_id: str, max_records: int) -> list[dict]:
    """Use the /search/ endpoint (type=o, court=) to find opinions by court,
    then fetch full text for each via the opinions detail endpoint.

    Returns a materialized list. Bulk entries omit citation strings since
    they are used as distractors in the retrieval corpus, never as gold citations.
    """
    checkpoint_path = _bulk_checkpoint_path(court_id)
    results, next_url = _load_bulk_checkpoint(checkpoint_path)
    if results:
        logger.info("Resuming bulk %s from checkpoint: %d opinions cached",
                    court_id, len(results))

    if next_url is None:
        url = f"{config.COURTLISTENER_API_BASE}/search/"
        params: dict | None = {"type": "o", "court": court_id, "page_size": 10}
    else:
        url = next_url
        params = None

    pbar = tqdm(
        total=max_records, desc=f"bulk {court_id}",
        leave=False, initial=min(len(results), max_records),
    )
    try:
        while url and len(results) < max_records:
            try:
                r = _get(url, params=params)
            except (requests.HTTPError, CourtListenerError) as e:
                logger.warning("Skipping court %s: %s", court_id, e)
                return results
            body = r.json()
            for cluster_hit in body.get("results", []):
                if len(results) >= max_records:
                    break
                for op_entry in cluster_hit.get("opinions", []):
                    if len(results) >= max_records:
                        break
                    op_id = op_entry.get("id")
                    if op_id is None:
                        continue
                    op = _fetch_opinion_minimal(op_id, court_id, cluster_hit)
                    if op:
                        results.append(op)
                        pbar.update(1)
            url = body.get("next")
            params = None
            _save_bulk_checkpoint(checkpoint_path, results, url)
    finally:
        pbar.close()

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return results


def _bulk_checkpoint_path(court_id: str) -> Path:
    d = config.CORPUS_DIR / "bulk_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{court_id}.json"


def _load_bulk_checkpoint(path: Path) -> tuple[list[dict], str | None]:
    if not path.exists():
        return [], None
    try:
        data = json.loads(path.read_text())
        return data.get("results", []), data.get("next_url")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Bulk checkpoint %s unreadable, starting fresh: %s", path, e)
        return [], None


def _save_bulk_checkpoint(path: Path, results: list[dict], next_url: str | None) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"results": results, "next_url": next_url}))
    tmp.replace(path)


def _fetch_opinion_minimal(opinion_id: int, court_id: str, cluster_hit: dict) -> dict | None:
    try:
        r = _get(f"{config.COURTLISTENER_API_BASE}/opinions/{opinion_id}/")
    except (requests.HTTPError, CourtListenerError):
        return None
    body = r.json()
    text = body.get("plain_text") or body.get("html_with_citations") or ""
    if not text.strip():
        return None

    raw_cites = cluster_hit.get("citation") or []
    all_cits = [c for c in raw_cites if isinstance(c, str) and c.strip()]
    primary = all_cits[0] if all_cits else f"distractor::{opinion_id}"

    return {
        "opinion_id": str(opinion_id),
        "text": text,
        "court": court_id,
        "date_filed": cluster_hit.get("dateFiled", ""),
        "primary_citation": primary,
        "normalized_citation": primary,
        "all_citations": all_cits,
    }
