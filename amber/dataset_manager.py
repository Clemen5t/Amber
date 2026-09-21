from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import re
import shutil
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

TRAIN_FILE = DATA_DIR / "train.txt"
PREPARED_TRAIN_FILE = PROCESSED_DIR / "train.txt"
PREPARED_VALIDATION_FILE = PROCESSED_DIR / "validation.txt"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.json"

SUPPORTED_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".text",
    ".jsonl",
    ".xml",
    ".gz",
    ".bz2",
}

WIKIMEDIA_LICENSE = "Wikimedia - CC BY-SA / GFDL selon le projet et le contenu"
WIKIMEDIA_LICENSE_URL = (
    "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/fr"
)

SOURCE_CATALOG = {
    "wikipedia_fr": {
        "name": "Wikipedia FR - articles",
        "url": (
            "https://dumps.wikimedia.org/frwiki/latest/"
            "frwiki-latest-pages-articles-multistream.xml.bz2"
        ),
        "filename": "frwiki-latest-pages-articles-multistream.xml.bz2",
        "language": "fr",
        "license": WIKIMEDIA_LICENSE,
        "license_url": WIKIMEDIA_LICENSE_URL,
        "note": "Encyclopédie généraliste. Dump officiel très volumineux.",
    },
    "wiktionary_fr": {
        "name": "Wiktionnaire FR - articles",
        "url": (
            "https://dumps.wikimedia.org/frwiktionary/latest/"
            "frwiktionary-latest-pages-articles-multistream.xml.bz2"
        ),
        "filename": (
            "frwiktionary-latest-pages-articles-multistream.xml.bz2"
        ),
        "language": "fr",
        "license": WIKIMEDIA_LICENSE,
        "license_url": WIKIMEDIA_LICENSE_URL,
        "note": "Vocabulaire, définitions et exemples linguistiques.",
    },
    "wikisource_fr": {
        "name": "Wikisource FR - textes",
        "url": (
            "https://dumps.wikimedia.org/frwikisource/latest/"
            "frwikisource-latest-pages-articles-multistream.xml.bz2"
        ),
        "filename": (
            "frwikisource-latest-pages-articles-multistream.xml.bz2"
        ),
        "language": "fr",
        "license": WIKIMEDIA_LICENSE,
        "license_url": WIKIMEDIA_LICENSE_URL,
        "note": "Textes et ouvrages. Vérifier la licence de chaque contenu si nécessaire.",
    },
    "wikibooks_fr": {
        "name": "Wikilivres FR - articles",
        "url": (
            "https://dumps.wikimedia.org/frwikibooks/latest/"
            "frwikibooks-latest-pages-articles-multistream.xml.bz2"
        ),
        "filename": "frwikibooks-latest-pages-articles-multistream.xml.bz2",
        "language": "fr",
        "license": WIKIMEDIA_LICENSE,
        "license_url": WIKIMEDIA_LICENSE_URL,
        "note": "Contenu pédagogique et livres collaboratifs.",
    },
    "wikiquote_fr": {
        "name": "Wikiquote FR - articles",
        "url": (
            "https://dumps.wikimedia.org/frwikiquote/latest/"
            "frwikiquote-latest-pages-articles-multistream.xml.bz2"
        ),
        "filename": "frwikiquote-latest-pages-articles-multistream.xml.bz2",
        "language": "fr",
        "license": WIKIMEDIA_LICENSE,
        "license_url": WIKIMEDIA_LICENSE_URL,
        "note": "Citations et contexte. Corpus plus petit que Wikipedia.",
    },
}


class DownloadCancelled(RuntimeError):
    pass


def ensure_layout() -> None:
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        TOKENIZER_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def format_bytes(size: int | float) -> str:
    value = float(size)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0

    return f"{value:.1f} TB"


def format_tokens(tokens: int | float) -> str:
    value = float(tokens)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"

    if value >= 1_000:
        return f"{value / 1_000:.1f} K"

    return f"{int(value):,}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def file_stats(
    path: Path,
    *,
    detailed_limit_bytes: int = 64 * 1024 * 1024,
) -> dict:
    path = Path(path)

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_bytes": 0,
            "size_human": "0 B",
            "characters": 0,
            "utf8_bytes": 0,
            "lines": 0,
            "words": 0,
            "sha256": None,
            "modified": None,
            "details_skipped": False,
        }

    stat = path.stat()
    size = stat.st_size

    if size > detailed_limit_bytes:
        return {
            "exists": True,
            "path": str(path),
            "size_bytes": size,
            "size_human": format_bytes(size),
            "characters": None,
            "utf8_bytes": size,
            "lines": None,
            "words": None,
            "sha256": None,
            "modified": datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(timespec="seconds"),
            "details_skipped": True,
        }

    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")

    return {
        "exists": True,
        "path": str(path),
        "size_bytes": len(raw),
        "size_human": format_bytes(len(raw)),
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "modified": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(timespec="seconds"),
        "details_skipped": False,
    }


def dataset_stats(path: Path = TRAIN_FILE) -> dict:
    return file_stats(path)


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".source.json")


def _write_source_metadata(path: Path, metadata: dict) -> None:
    metadata = dict(metadata)

    metadata.setdefault(
        "added_at",
        datetime.now().isoformat(timespec="seconds")
    )

    metadata["local_path"] = str(path)

    _metadata_path(path).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def list_raw_sources() -> list[dict]:
    ensure_layout()
    sources = []

    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.name.endswith(".source.json"):
            continue

        if path.name.endswith(".part"):
            continue

        metadata = {}

        meta_path = _metadata_path(path)

        if meta_path.exists():
            try:
                metadata = json.loads(
                    meta_path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                metadata = {}

        sources.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_human": format_bytes(
                    path.stat().st_size
                ),
                "sha256": metadata.get("sha256"),
                "source_type": metadata.get(
                    "source_type",
                    "unknown"
                ),
                "source_url": metadata.get("source_url"),
                "license": metadata.get("license"),
                "license_url": metadata.get("license_url"),
                "language": metadata.get("language"),
            }
        )

    return sources


def import_local_file(
    source_path: str | Path,
    *,
    language: str | None = None,
    license_name: str | None = None,
) -> dict:
    ensure_layout()

    source = Path(source_path).expanduser().resolve()

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {source}"
        )

    lower_name = source.name.lower()

    if not any(
        lower_name.endswith(suffix)
        for suffix in SUPPORTED_TEXT_SUFFIXES
    ):
        raise ValueError(
            "Format non pris en charge. "
            "Utilise TXT, MD, JSONL, XML, GZ ou BZ2."
        )

    destination = RAW_DIR / source.name

    if destination.exists():
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        destination = RAW_DIR / (
            f"{source.stem}_{stamp}{source.suffix}"
        )

    shutil.copy2(
        source,
        destination
    )

    sha256 = _sha256_file(
        destination
    )

    metadata = {
        "source_type": "local_import",
        "original_path": str(source),
        "language": language,
        "license": license_name,
        "sha256": sha256,
    }

    _write_source_metadata(
        destination,
        metadata
    )

    return {
        "path": str(destination),
        "name": destination.name,
        "size_bytes": destination.stat().st_size,
        "size_human": format_bytes(
            destination.stat().st_size
        ),
        "sha256": sha256,
    }


def probe_remote_size(
    url: str,
    *,
    timeout: int = 25,
) -> int | None:
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None

    headers = {
        "User-Agent": (
            "AmberDatasetManager/0.0.7 "
            "(local research project)"
        )
    }

    request = urllib.request.Request(
        url,
        headers=headers,
        method="HEAD"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:
            length = response.headers.get(
                "Content-Length"
            )

            if length:
                return int(length)

    except Exception:
        pass

    request = urllib.request.Request(
        url,
        headers={
            **headers,
            "Range": "bytes=0-0",
        }
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:
            content_range = response.headers.get(
                "Content-Range"
            )

            if content_range and "/" in content_range:
                total = content_range.rsplit(
                    "/",
                    1
                )[-1]

                if total.isdigit():
                    return int(total)

            length = response.headers.get(
                "Content-Length"
            )

            if length:
                return int(length)

    except Exception:
        return None

    return None


def estimate_source_plan(
    selected_keys: list[str],
    *,
    target_tokens: int = 100_000_000,
) -> dict:
    ensure_layout()

    items = []
    remote_total = 0
    unknown = 0

    for key in selected_keys:
        source = SOURCE_CATALOG.get(key)

        if not source:
            continue

        size = probe_remote_size(
            source["url"]
        )

        if size is None:
            unknown += 1
        else:
            remote_total += size

        destination = RAW_DIR / source["filename"]
        part = destination.with_suffix(
            destination.suffix + ".part"
        )

        existing = 0

        if destination.exists():
            existing = destination.stat().st_size
        elif part.exists():
            existing = part.stat().st_size

        remaining = (
            max(0, size - existing)
            if size is not None
            else None
        )

        items.append(
            {
                "key": key,
                "name": source["name"],
                "size_bytes": size,
                "size_human": (
                    format_bytes(size)
                    if size is not None
                    else "inconnue"
                ),
                "existing_bytes": existing,
                "remaining_bytes": remaining,
                "remaining_human": (
                    format_bytes(remaining)
                    if remaining is not None
                    else "inconnue"
                ),
                "license": source.get("license"),
                "license_url": source.get("license_url"),
                "note": source.get("note"),
            }
        )

    target_tokens = max(
        1,
        int(target_tokens)
    )

    # Estimation volontairement prudente avant le vrai tokenization.
    # En français, un tokenizer BPE 32k produit souvent quelques caractères
    # par token. On réserve ~4 octets de texte/token + marge de travail.
    estimated_text_bytes = target_tokens * 4
    prepared_working_bytes = estimated_text_bytes * 2
    safety_bytes = 2 * 1024**3

    known_remaining = sum(
        item["remaining_bytes"] or 0
        for item in items
        if item["remaining_bytes"] is not None
    )

    recommended_free = (
        known_remaining
        + prepared_working_bytes
        + safety_bytes
    )

    usage = shutil.disk_usage(
        DATA_DIR
    )

    return {
        "items": items,
        "selected_count": len(items),
        "remote_total_bytes": remote_total,
        "remote_total_human": format_bytes(remote_total),
        "known_remaining_bytes": known_remaining,
        "known_remaining_human": format_bytes(known_remaining),
        "unknown_sizes": unknown,
        "target_tokens": target_tokens,
        "target_tokens_human": format_tokens(target_tokens),
        "estimated_text_bytes": estimated_text_bytes,
        "estimated_text_human": format_bytes(estimated_text_bytes),
        "recommended_free_bytes": recommended_free,
        "recommended_free_human": format_bytes(recommended_free),
        "disk_free_bytes": usage.free,
        "disk_free_human": format_bytes(usage.free),
        "disk_ok": usage.free >= recommended_free,
    }


def download_source(
    url: str,
    *,
    filename: str | None = None,
    language: str | None = None,
    license_name: str | None = None,
    license_url: str | None = None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
    resume: bool = True,
) -> dict:
    ensure_layout()

    parsed = urllib.parse.urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https"
    }:
        raise ValueError(
            "Seules les URL HTTP/HTTPS sont acceptées."
        )

    if not filename:
        filename = Path(
            urllib.parse.unquote(
                parsed.path
            )
        ).name

    if not filename:
        filename = (
            "download_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".txt"
        )

    destination = RAW_DIR / filename
    temp = destination.with_suffix(
        destination.suffix + ".part"
    )

    if destination.exists():
        return {
            "path": str(destination),
            "name": destination.name,
            "size_bytes": destination.stat().st_size,
            "size_human": format_bytes(
                destination.stat().st_size
            ),
            "sha256": None,
            "resumed": False,
            "already_present": True,
        }

    resume_from = (
        temp.stat().st_size
        if resume and temp.exists()
        else 0
    )

    headers = {
        "User-Agent": (
            "AmberDatasetManager/0.0.7 "
            "(local research project)"
        )
    }

    if resume_from > 0:
        headers["Range"] = (
            f"bytes={resume_from}-"
        )

    request = urllib.request.Request(
        url,
        headers=headers
    )

    downloaded = resume_from
    total = None
    resumed = resume_from > 0

    try:
        with urllib.request.urlopen(
            request,
            timeout=60
        ) as response:
            status = getattr(
                response,
                "status",
                None
            )

            content_range = response.headers.get(
                "Content-Range"
            )

            if resumed and status != 206:
                resume_from = 0
                downloaded = 0
                resumed = False

            if content_range and "/" in content_range:
                tail = content_range.rsplit(
                    "/",
                    1
                )[-1]

                if tail.isdigit():
                    total = int(tail)

            if total is None:
                length = response.headers.get(
                    "Content-Length"
                )

                if length:
                    length = int(length)
                    total = (
                        length + resume_from
                        if resumed
                        else length
                    )

            mode = (
                "ab"
                if resumed
                else "wb"
            )

            with temp.open(mode) as handle:
                while True:
                    if (
                        cancel_event is not None
                        and cancel_event.is_set()
                    ):
                        raise DownloadCancelled(
                            "Téléchargement annulé. "
                            "Le fichier .part est conservé pour reprendre plus tard."
                        )

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    handle.write(
                        chunk
                    )

                    downloaded += len(
                        chunk
                    )

                    if progress_callback:
                        progress_callback(
                            downloaded,
                            total
                        )

        temp.replace(
            destination
        )

    except DownloadCancelled:
        raise

    except Exception:
        # On conserve volontairement le .part pour permettre une reprise.
        raise

    sha256 = _sha256_file(
        destination
    )

    metadata = {
        "source_type": "download",
        "source_url": url,
        "language": language,
        "license": license_name,
        "license_url": license_url,
        "sha256": sha256,
    }

    _write_source_metadata(
        destination,
        metadata
    )

    return {
        "path": str(destination),
        "name": destination.name,
        "size_bytes": destination.stat().st_size,
        "size_human": format_bytes(
            destination.stat().st_size
        ),
        "sha256": sha256,
        "resumed": resumed,
        "already_present": False,
    }


def download_catalog_source(
    key: str,
    *,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    source = SOURCE_CATALOG.get(
        key
    )

    if not source:
        raise KeyError(
            f"Source inconnue : {key}"
        )

    return download_source(
        source["url"],
        filename=source["filename"],
        language=source.get("language"),
        license_name=source.get("license"),
        license_url=source.get("license_url"),
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        resume=True,
    )


def download_catalog_sources(
    keys: list[str],
    *,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> list[dict]:
    results = []

    for index, key in enumerate(keys, start=1):
        if (
            cancel_event is not None
            and cancel_event.is_set()
        ):
            raise DownloadCancelled(
                "Téléchargement annulé."
            )

        source = SOURCE_CATALOG.get(
            key
        )

        if not source:
            continue

        def progress(
            downloaded,
            total,
            *,
            current_key=key,
            current_name=source["name"],
            current_index=index,
        ):
            if progress_callback:
                progress_callback(
                    {
                        "key": current_key,
                        "name": current_name,
                        "index": current_index,
                        "count": len(keys),
                        "downloaded": downloaded,
                        "total": total,
                    }
                )

        result = download_catalog_source(
            key,
            progress_callback=progress,
            cancel_event=cancel_event,
        )

        results.append(
            result
        )

    return results


def _normalize_text(
    text: str
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace(
        "\x00",
        " "
    )

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    lines = []

    for line in text.splitlines():
        line = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        if line:
            lines.append(
                line
            )

    return "\n".join(
        lines
    ).strip()


def _strip_wiki_markup(
    text: str
) -> str:
    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"<ref\b[^>]*>.*?</ref>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<ref\b[^>]*/>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
        r"\1",
        text
    )

    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text
    )

    text = re.sub(
        r"\[https?://[^\]]+\]",
        " ",
        text
    )

    text = re.sub(
        r"\{\{[^{}]{0,2000}\}\}",
        " ",
        text
    )

    text = re.sub(
        r"={2,}\s*(.*?)\s*={2,}",
        r"\1",
        text
    )

    text = text.replace(
        "'''",
        ""
    ).replace(
        "''",
        ""
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    return _normalize_text(
        text
    )


def _open_binary(
    path: Path
):
    name = path.name.lower()

    if name.endswith(".bz2"):
        return bz2.open(
            path,
            "rb"
        )

    if name.endswith(".gz"):
        return gzip.open(
            path,
            "rb"
        )

    return path.open(
        "rb"
    )


def _open_text(
    path: Path
):
    binary = _open_binary(
        path
    )

    return io.TextIOWrapper(
        binary,
        encoding="utf-8",
        errors="replace"
    )


def _local_name(
    tag: str
) -> str:
    if "}" in tag:
        return tag.rsplit(
            "}",
            1
        )[-1]

    return tag


def _iter_mediawiki_xml(
    path: Path
):
    with _open_binary(path) as handle:
        context = ET.iterparse(
            handle,
            events=("end",)
        )

        for _, elem in context:
            if _local_name(
                elem.tag
            ) != "page":
                continue

            namespace = None
            title = ""
            body = ""

            for child in elem.iter():
                name = _local_name(
                    child.tag
                )

                if name == "ns" and namespace is None:
                    namespace = (
                        child.text or ""
                    ).strip()

                elif name == "title" and not title:
                    title = (
                        child.text or ""
                    ).strip()

                elif name == "text":
                    body = child.text or ""

            if namespace in {
                None,
                "0"
            }:
                cleaned = _strip_wiki_markup(
                    body
                )

                if title and cleaned:
                    yield (
                        title
                        + "\n"
                        + cleaned
                    )

            elem.clear()


def _iter_jsonl(
    path: Path
):
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )
            except Exception:
                continue

            if isinstance(item, str):
                text = item

            elif isinstance(item, dict):
                text = None

                for key in (
                    "text",
                    "content",
                    "body",
                    "article",
                    "document",
                ):
                    value = item.get(
                        key
                    )

                    if isinstance(value, str):
                        text = value
                        break

                if text is None:
                    continue

            else:
                continue

            normalized = _normalize_text(
                text
            )

            if normalized:
                yield normalized


def _iter_plain_text(
    path: Path
):
    with _open_text(path) as handle:
        buffer = []

        for line in handle:
            line = line.rstrip(
                "\n"
            )

            if line.strip():
                buffer.append(
                    line
                )

            elif buffer:
                text = _normalize_text(
                    "\n".join(
                        buffer
                    )
                )

                buffer = []

                if text:
                    yield text

        if buffer:
            text = _normalize_text(
                "\n".join(
                    buffer
                )
            )

            if text:
                yield text


def iter_documents(
    path: str | Path
):
    path = Path(
        path
    )

    name = path.name.lower()

    if (
        name.endswith(".xml")
        or name.endswith(".xml.bz2")
        or name.endswith(".xml.gz")
    ):
        yield from _iter_mediawiki_xml(
            path
        )
        return

    if (
        name.endswith(".jsonl")
        or name.endswith(".jsonl.bz2")
        or name.endswith(".jsonl.gz")
    ):
        yield from _iter_jsonl(
            path
        )
        return

    yield from _iter_plain_text(
        path
    )


def prepare_dataset(
    *,
    validation_ratio: float = 0.02,
    min_chars: int = 80,
    max_chars: int = 20000,
    include_seed: bool = True,
    target_tokens: int | None = None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    ensure_layout()

    if not 0.0 <= validation_ratio < 0.5:
        raise ValueError(
            "validation_ratio doit être compris entre 0 et 0.5."
        )

    sources = []

    if include_seed and TRAIN_FILE.exists():
        sources.append(
            TRAIN_FILE
        )

    sources.extend(
        [
            Path(item["path"])
            for item in list_raw_sources()
        ]
    )

    if not sources:
        raise RuntimeError(
            "Aucune source disponible."
        )

    seen = set()

    train_docs = 0
    validation_docs = 0
    skipped = 0
    total_chars = 0
    stopped_on_target = False

    target_chars = (
        int(target_tokens) * 4
        if target_tokens
        else None
    )

    with PREPARED_TRAIN_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as train_handle, PREPARED_VALIDATION_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as validation_handle:

        source_index = 0

        for source in sources:
            source_index += 1

            if progress_callback:
                progress_callback(
                    {
                        "stage": "source",
                        "source": source.name,
                        "source_index": source_index,
                        "source_count": len(sources),
                        "train_docs": train_docs,
                        "validation_docs": validation_docs,
                        "skipped": skipped,
                        "approx_tokens": total_chars // 4,
                    }
                )

            for document in iter_documents(
                source
            ):
                if (
                    cancel_event is not None
                    and cancel_event.is_set()
                ):
                    raise DownloadCancelled(
                        "Préparation annulée."
                    )

                document = _normalize_text(
                    document
                )

                length = len(
                    document
                )

                if length < min_chars:
                    skipped += 1
                    continue

                if length > max_chars:
                    document = document[
                        :max_chars
                    ]

                digest = hashlib.sha256(
                    document.encode(
                        "utf-8"
                    )
                ).hexdigest()

                if digest in seen:
                    skipped += 1
                    continue

                seen.add(
                    digest
                )

                bucket = int(
                    digest[:8],
                    16
                ) / 0xFFFFFFFF

                if bucket < validation_ratio:
                    handle = validation_handle
                    validation_docs += 1
                else:
                    handle = train_handle
                    train_docs += 1

                handle.write(
                    document
                )

                handle.write(
                    "\n\n"
                )

                total_chars += len(
                    document
                )

                if (
                    target_chars is not None
                    and total_chars >= target_chars
                ):
                    stopped_on_target = True
                    break

                if (
                    progress_callback
                    and (
                        train_docs
                        + validation_docs
                    ) % 1000 == 0
                ):
                    progress_callback(
                        {
                            "stage": "documents",
                            "source": source.name,
                            "train_docs": train_docs,
                            "validation_docs": validation_docs,
                            "skipped": skipped,
                            "approx_tokens": total_chars // 4,
                        }
                    )

            if stopped_on_target:
                break

    approx_tokens = total_chars // 4

    result = {
        "train_docs": train_docs,
        "validation_docs": validation_docs,
        "skipped": skipped,
        "unique_docs": len(seen),
        "characters": total_chars,
        "approx_tokens": approx_tokens,
        "approx_tokens_human": format_tokens(
            approx_tokens
        ),
        "target_tokens": target_tokens,
        "target_reached": (
            stopped_on_target
            if target_tokens
            else None
        ),
        "train_file": str(
            PREPARED_TRAIN_FILE
        ),
        "validation_file": str(
            PREPARED_VALIDATION_FILE
        ),
        "train_size": (
            PREPARED_TRAIN_FILE.stat().st_size
            if PREPARED_TRAIN_FILE.exists()
            else 0
        ),
        "validation_size": (
            PREPARED_VALIDATION_FILE.stat().st_size
            if PREPARED_VALIDATION_FILE.exists()
            else 0
        ),
    }

    write_manifest(
        prepared=result
    )

    return result


def write_manifest(
    path: Path = TRAIN_FILE,
    *,
    prepared: dict | None = None,
    planner: dict | None = None,
) -> dict:
    ensure_layout()

    seed_stats = dataset_stats(
        path
    )

    raw_sources = list_raw_sources()

    if prepared is None:
        prepared = {
            "train": file_stats(
                PREPARED_TRAIN_FILE
            ),
            "validation": file_stats(
                PREPARED_VALIDATION_FILE
            ),
        }

    manifest = {
        "amber_dataset_manifest_version": 3,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "seed_corpus": seed_stats,
        "raw_sources": raw_sources,
        "prepared": prepared,
        "planner": planner,
        "tokenizer_target": {
            "name": "AmberBPETokenizer",
            "vocab_size": 32000,
            "training_status": (
                "ready"
                if PREPARED_TRAIN_FILE.exists()
                else "dataset_not_prepared"
            ),
        },
        "status": (
            "prepared"
            if PREPARED_TRAIN_FILE.exists()
            and PREPARED_TRAIN_FILE.stat().st_size > 0
            else "collecting"
        ),
        "notes": (
            "Amber 0.0.7 conserve Amber Seed séparément. "
            "Les objectifs en tokens sont des estimations avant tokenization exacte."
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return manifest
