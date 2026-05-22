"""Generic, manifest-driven Corpus.

A corpus is a JSON manifest under ``blowfish/datasets/data/<name>.json``
plus a content cache under ``$BLOWFISH_CACHE_DIR/corpora/<name>/``. Everything
corpus-specific (source, license, document list, fetch URLs) lives in the
manifest as **metadata**; the ``Corpus`` class is intentionally generic so
users can add a new corpus by dropping in a new manifest with no Python
changes.

Manifest schema (v2)::

    {
        "name": "<corpus name>",
        "version": "v1",
        "description": "...",
        "license": "...",
        "source": "...",
        "bootstrap_recipe": "<recipe id in scripts/bootstrap_corpora.py>",
        "documents": [
            {
                "doc_id": "...",
                "title": "...",
                "mirror_urls": ["..."],
                "sha256": "...",
                "license": "...",
                ...  # arbitrary per-document metadata
            },
            ...
        ]
    }

Adding a new corpus:

1. Write ``blowfish/datasets/data/<name>.json`` following the schema above.
2. (Optional) Add a bootstrap recipe to ``scripts/bootstrap_corpora.py`` so
   the manifest can be populated automatically; otherwise hand-edit
   ``documents[]`` with stable URLs + sha256.
3. The corpus is now visible via ``list_corpora()`` and loadable via
   ``Corpus("<name>")`` — no subclass needed.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional, Protocol, runtime_checkable

from blowfish.datasets.cache import cache_root
from blowfish.datasets.download import sha256_file
from blowfish.datasets.fetchers import FETCHERS, get_fetcher
from blowfish.datasets.types import Document


_DATA_DIR = Path(__file__).parent / "data"
_LEGACY_TOP_LEVEL_KEYS = ("_schema_version", "_corpus_name", "_description", "_license_note")


@runtime_checkable
class CorpusProtocol(Protocol):
    """Structural interface so adapters can plug in non-manifest sources later."""

    name: str
    version: str

    def iter_documents(self) -> Iterator[Document]: ...


class Corpus:
    """Manifest-driven corpus. One class, many manifests."""

    def __init__(
        self,
        name: str,
        *,
        cache_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        self.name = name
        self._data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (cache_root() / "corpora" / name)
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()

    @property
    def manifest_path(self) -> Path:
        return self._data_dir / f"{self.name}.json"

    @property
    def version(self) -> str:
        return str(self._manifest.get("version", "v1"))

    @property
    def license(self) -> str:
        return str(self._manifest.get("license", "unknown"))

    @property
    def source(self) -> str:
        return str(self._manifest.get("source", "unknown"))

    @property
    def description(self) -> str:
        return str(self._manifest.get("description", ""))

    @property
    def documents(self) -> list[dict[str, Any]]:
        return list(self._manifest.get("documents", []))

    def metadata(self) -> dict[str, Any]:
        """Top-level manifest fields excluding the document list."""
        return {k: v for k, v in self._manifest.items() if k != "documents"}

    def iter_documents(self, *, strict_sha: bool = False) -> Iterator[Document]:
        """Yield each document; downloads at runtime if the cache is cold.

        For each manifest entry: if the local cache has a file with matching
        sha256, use it. Otherwise dispatch to the source-specific fetcher
        (``entry["source"]``) to re-materialize the text from its archive URL.

        Archive content drifts (Wikipedia articles get edited, news pages get
        revised). By default the loader is **drift-tolerant**: it emits a
        ``UserWarning`` if the fetched SHA differs from the manifest's, then
        uses the fetched content. Pass ``strict_sha=True`` to raise instead —
        useful for reproducibility-critical experiment runs.
        """
        for entry in self.documents:
            dest = self._cache_dir / f"{entry['doc_id']}.txt"
            expected_sha = entry.get("sha256")
            if dest.exists() and (
                expected_sha is None or sha256_file(dest) == expected_sha
            ):
                text = dest.read_text(encoding="utf-8")
            else:
                source = entry.get("source", "direct")
                fetcher = get_fetcher(source)
                text = fetcher.fetch(entry)
                if expected_sha:
                    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if actual != expected_sha:
                        msg = (
                            f"{self.name}/{entry['doc_id']}: archive content drifted "
                            f"from manifest sha256 ({expected_sha[:12]}… → {actual[:12]}…)."
                        )
                        if strict_sha:
                            raise RuntimeError(
                                msg + " strict_sha=True; re-run "
                                "scripts/bootstrap_corpora.py to refresh."
                            )
                        warnings.warn(
                            msg + " Using fetched content. Run "
                            "scripts/bootstrap_corpora.py to refresh manifest.",
                            UserWarning,
                            stacklevel=2,
                        )
                dest.write_text(text, encoding="utf-8")
            doc_metadata = {
                k: v
                for k, v in entry.items()
                if k not in ("doc_id", "title", "mirror_urls", "sha256")
            }
            yield Document(
                doc_id=entry["doc_id"],
                title=entry["title"],
                text=text,
                metadata=doc_metadata,
            )

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest for corpus {self.name!r} at {self.manifest_path}. "
                f"List available corpora with blowfish.datasets.list_corpora()."
            )
        with self.manifest_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return _normalize_manifest(raw)


def _normalize_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept v1 (underscored ``_corpus_name``, ``_description``) and v2 (clean
    top-level ``name``/``description``) manifests; canonicalize to v2.

    ``_license_note`` is descriptive prose, NOT a license identifier — it maps
    to ``license_note``, leaving ``license`` free to carry the SPDX-ish id.
    When the manifest does not declare a corpus-level ``license`` but every
    document agrees on one, the document-majority value is promoted.
    """
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in _LEGACY_TOP_LEVEL_KEYS:
            cleaned = k.lstrip("_")
            if cleaned == "corpus_name":
                cleaned = "name"
            out.setdefault(cleaned, v)
        else:
            out[k] = v
    if "license" not in out:
        doc_licenses = [
            d.get("license") for d in out.get("documents", []) if d.get("license")
        ]
        if doc_licenses and len(set(doc_licenses)) == 1:
            out["license"] = doc_licenses[0]
    return out


def list_corpora(*, data_dir: Optional[Path] = None) -> list[str]:
    """List corpus names by scanning the data directory for ``*.json`` manifests."""
    data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
    if not data_dir.exists():
        return []
    return sorted(p.stem for p in data_dir.glob("*.json") if p.is_file())


def load_corpus(name: str, **kwargs: Any) -> Corpus:
    """Convenience factory."""
    return Corpus(name, **kwargs)


# Back-compat factory functions for the named corpora we ship. New corpora
# should be discovered via ``list_corpora()`` / ``load_corpus(name)`` instead
# of being added here.
def NobelPhysics(**kwargs: Any) -> Corpus:
    return Corpus("nobel_physics", **kwargs)


def FieldsMedalists(**kwargs: Any) -> Corpus:
    return Corpus("fields_medalists", **kwargs)


def Britannica1911(**kwargs: Any) -> Corpus:
    return Corpus("britannica_1911", **kwargs)


def Wikinews(**kwargs: Any) -> Corpus:
    return Corpus("wikinews", **kwargs)


def ChroniclingAmericaFinance(**kwargs: Any) -> Corpus:
    return Corpus("chronicling_america_finance", **kwargs)


def CCNewsFinance(**kwargs: Any) -> Corpus:
    return Corpus("cc_news_finance", **kwargs)


class CorpusHooks:
    """Lookup-by-name registry. ``getattr(CorpusHooks, name)()`` returns a Corpus."""

    nobel_physics: ClassVar = staticmethod(NobelPhysics)
    fields_medalists: ClassVar = staticmethod(FieldsMedalists)
    britannica_1911: ClassVar = staticmethod(Britannica1911)
    wikinews: ClassVar = staticmethod(Wikinews)
    chronicling_america_finance: ClassVar = staticmethod(ChroniclingAmericaFinance)
    cc_news_finance: ClassVar = staticmethod(CCNewsFinance)


ALL_CORPUS_NAMES = (
    "nobel_physics",
    "fields_medalists",
    "britannica_1911",
    "wikinews",
    "chronicling_america_finance",
    "cc_news_finance",
)
