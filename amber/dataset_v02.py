from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

from amber.dataset_manager import (
    DATA_DIR,
    DownloadCancelled,
    format_bytes,
    format_tokens,
    iter_documents,
    list_raw_sources,
)


ROOT = Path(__file__).resolve().parents[1]
V02_DIR = DATA_DIR / "v02"
PROCESSED_DIR = V02_DIR / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.txt"
VALIDATION_FILE = PROCESSED_DIR / "validation.txt"
MANIFEST_FILE = V02_DIR / "manifest.json"

DEFAULT_TARGET_TOKENS = 500_000_000

# Amber 0.1 a montré une forte contamination par Wiktionnaire/Wikiquote.
# Amber 0.2 limite donc volontairement ces sources et privilégie la prose.
SOURCE_CAPS = {
    "wikipedia": 0.78,
    "wikisource": 0.12,
    "wikibooks": 0.06,
    "wiktionary": 0.025,
    "wikiquote": 0.015,
    "other": 0.05,
}

SOURCE_ORDER = {
    "wikipedia": 0,
    "wikisource": 1,
    "wikibooks": 2,
    "other": 3,
    "wiktionary": 4,
    "wikiquote": 5,
}

BAD_LINE_PATTERNS = [
    re.compile(r"^\s*[|!].*$"),
    re.compile(r"^\s*\{\|.*$"),
    re.compile(r"^\s*\|\}.*$"),
    re.compile(r"^\s*\|-.*$"),
    re.compile(r"^\s*(?:catégorie|category)\s*:", re.I),
    re.compile(r"^\s*(?:fichier|file|image)\s*:", re.I),
    re.compile(r"^\s*(?:source|réf|ref|référence)\s*=", re.I),
    re.compile(r"^\s*[#*:;]+\s*$"),
    re.compile(r"^\s*[-–—_=]{3,}\s*$"),
]

RESIDUAL_MARKUP = re.compile(
    r"(\{\{|\}\}|\[\[|\]\]|<ref\b|</ref>|\|\s*source\s*=|"
    r"\b(?:thumb|upright|lang|nocat)\s*=)",
    re.I,
)


def _source_kind(path: Path) -> str:
    name = path.name.lower()

    if "frwiki-" in name or "wikipedia" in name:
        return "wikipedia"

    if "wikisource" in name:
        return "wikisource"

    if "wikibooks" in name or "wikilivres" in name:
        return "wikibooks"

    if "wiktionary" in name or "wiktionnaire" in name:
        return "wiktionary"

    if "wikiquote" in name:
        return "wikiquote"

    return "other"


def _collapse_templates(text: str) -> str:
    # Plusieurs passes retirent aussi une bonne partie des templates imbriqués.
    for _ in range(8):
        new_text = re.sub(
            r"\{\{[^{}]*\}\}",
            " ",
            text,
            flags=re.DOTALL,
        )

        if new_text == text:
            break

        text = new_text

    return text


def _clean_links(text: str) -> str:
    text = re.sub(
        r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
        r"\1",
        text,
    )

    text = re.sub(
        r"\[https?://[^\s\]]+\s+([^\]]+)\]",
        r"\1",
        text,
    )

    text = re.sub(
        r"\[https?://[^\]]+\]",
        " ",
        text,
    )

    return text


def _normalize_line(line: str) -> str:
    line = unicodedata.normalize(
        "NFKC",
        line,
    )

    line = line.replace(
        "\x00",
        " ",
    )

    line = re.sub(
        r"<!--.*?-->",
        " ",
        line,
        flags=re.DOTALL,
    )

    line = re.sub(
        r"<ref\b[^>]*>.*?</ref>",
        " ",
        line,
        flags=re.I | re.DOTALL,
    )

    line = re.sub(
        r"<ref\b[^>]*/>",
        " ",
        line,
        flags=re.I,
    )

    line = _collapse_templates(
        line
    )

    line = _clean_links(
        line
    )

    line = re.sub(
        r"<[^>]+>",
        " ",
        line,
    )

    line = re.sub(
        r"={2,}\s*(.*?)\s*={2,}",
        r"\1",
        line,
    )

    # On conserve le texte d'une puce naturelle, mais pas son marqueur Wiki.
    line = re.sub(
        r"^\s*[#*:;]+\s*",
        "",
        line,
    )

    line = re.sub(
        r"^\s*[-•]\s+",
        "",
        line,
    )

    line = line.replace(
        "'''",
        "",
    ).replace(
        "''",
        "",
    )

    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    ).strip()

    return line


def _looks_like_noise(line: str) -> bool:
    if not line:
        return True

    for pattern in BAD_LINE_PATTERNS:
        if pattern.match(
            line
        ):
            return True

    lower = line.lower()

    if (
        "|source=" in lower
        or "| source=" in lower
        or "{{" in line
        or "}}" in line
        or "{|" in line
        or "|}" in line
    ):
        return True

    if len(line) <= 4 and not any(
        char.isalnum()
        for char in line
    ):
        return True

    return False


def _quality_ok(text: str) -> tuple[bool, str]:
    if len(text) < 140:
        return False, "too_short"

    words = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿŒœ0-9'-]+",
        text,
    )

    if len(words) < 24:
        return False, "too_few_words"

    if RESIDUAL_MARKUP.search(
        text
    ):
        return False, "residual_markup"

    visible = [
        char
        for char in text
        if not char.isspace()
    ]

    if visible:
        alpha_numeric = sum(
            char.isalnum()
            for char in visible
        )

        if (
            alpha_numeric
            / len(visible)
            < 0.55
        ):
            return False, "low_text_ratio"

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        short_symbol_lines = sum(
            1
            for line in lines
            if len(line) < 8
            and sum(
                char.isalnum()
                for char in line
            ) < 3
        )

        if (
            short_symbol_lines
            / len(lines)
            > 0.12
        ):
            return False, "symbol_lines"

        counts = Counter(
            lines
        )

        repeats = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        if (
            len(lines) >= 4
            and repeats
            / len(lines)
            > 0.20
        ):
            return False, "repeated_lines"

    return True, "ok"


def clean_document(
    document: str,
) -> tuple[str | None, str]:
    document = unicodedata.normalize(
        "NFKC",
        document,
    )

    document = _collapse_templates(
        document
    )

    lines = []

    for raw_line in document.splitlines():
        if _looks_like_noise(
            raw_line
        ):
            continue

        line = _normalize_line(
            raw_line
        )

        if _looks_like_noise(
            line
        ):
            continue

        if not line:
            continue

        lines.append(
            line
        )

    cleaned = "\n".join(
        lines
    ).strip()

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    cleaned = re.sub(
        r"(?:\n\s*[:*#|]\s*){2,}",
        "\n",
        cleaned,
    )

    ok, reason = _quality_ok(
        cleaned
    )

    if not ok:
        return None, reason

    return cleaned, "ok"


def _chunks(
    text: str,
    max_chars: int = 16_000,
):
    if len(text) <= max_chars:
        yield text
        return

    lines = text.splitlines()
    buffer = []
    size = 0

    for line in lines:
        extra = (
            len(line)
            + 1
        )

        if (
            buffer
            and size + extra > max_chars
        ):
            chunk = "\n".join(
                buffer
            ).strip()

            if chunk:
                yield chunk

            buffer = []
            size = 0

        buffer.append(
            line
        )

        size += extra

    if buffer:
        chunk = "\n".join(
            buffer
        ).strip()

        if chunk:
            yield chunk


def prepare_v02_dataset(
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    validation_ratio: float = 0.01,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    target_tokens = int(
        target_tokens
    )

    if target_tokens <= 0:
        raise ValueError(
            "target_tokens doit être > 0."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_sources = [
        Path(
            item["path"]
        )
        for item in list_raw_sources()
    ]

    if not raw_sources:
        raise RuntimeError(
            "Aucune source brute disponible."
        )

    raw_sources.sort(
        key=lambda path: (
            SOURCE_ORDER[
                _source_kind(path)
            ],
            path.name.lower(),
        )
    )

    target_chars = (
        target_tokens
        * 4
    )

    source_budgets = {
        kind: int(
            target_chars
            * fraction
        )
        for kind, fraction in SOURCE_CAPS.items()
    }

    source_chars = Counter()
    accepted_by_source = Counter()
    rejected = Counter()
    seen = set()

    total_chars = 0
    train_docs = 0
    validation_docs = 0

    with TRAIN_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as train_handle, VALIDATION_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as validation_handle:

        for source_index, source in enumerate(
            raw_sources,
            start=1,
        ):
            kind = _source_kind(
                source
            )

            budget = source_budgets[
                kind
            ]

            if progress_callback:
                progress_callback(
                    {
                        "stage": "source",
                        "source": source.name,
                        "kind": kind,
                        "source_index": source_index,
                        "source_count": len(raw_sources),
                        "accepted_docs": (
                            train_docs
                            + validation_docs
                        ),
                        "approx_tokens": (
                            total_chars
                            // 4
                        ),
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
                        "Préparation Amber 0.2 annulée."
                    )

                if (
                    source_chars[
                        kind
                    ]
                    >= budget
                ):
                    break

                cleaned, reason = clean_document(
                    document
                )

                if cleaned is None:
                    rejected[
                        reason
                    ] += 1
                    continue

                for chunk in _chunks(
                    cleaned
                ):
                    ok, reason = _quality_ok(
                        chunk
                    )

                    if not ok:
                        rejected[
                            reason
                        ] += 1
                        continue

                    digest = hashlib.sha256(
                        chunk.encode(
                            "utf-8"
                        )
                    ).hexdigest()

                    if digest in seen:
                        rejected[
                            "duplicate"
                        ] += 1
                        continue

                    seen.add(
                        digest
                    )

                    if (
                        source_chars[
                            kind
                        ]
                        + len(chunk)
                        > budget
                    ):
                        break

                    bucket = (
                        int(
                            digest[:8],
                            16,
                        )
                        / 0xFFFFFFFF
                    )

                    handle = (
                        validation_handle
                        if bucket
                        < validation_ratio
                        else train_handle
                    )

                    handle.write(
                        chunk
                    )

                    handle.write(
                        "\n\n"
                    )

                    if handle is validation_handle:
                        validation_docs += 1
                    else:
                        train_docs += 1

                    chars = len(
                        chunk
                    )

                    total_chars += chars
                    source_chars[
                        kind
                    ] += chars
                    accepted_by_source[
                        kind
                    ] += 1

                    if (
                        progress_callback
                        and (
                            train_docs
                            + validation_docs
                        )
                        % 1000
                        == 0
                    ):
                        progress_callback(
                            {
                                "stage": "documents",
                                "source": source.name,
                                "kind": kind,
                                "accepted_docs": (
                                    train_docs
                                    + validation_docs
                                ),
                                "rejected_docs": sum(
                                    rejected.values()
                                ),
                                "approx_tokens": (
                                    total_chars
                                    // 4
                                ),
                            }
                        )

                if total_chars >= target_chars:
                    break

            if total_chars >= target_chars:
                break

    approx_tokens = (
        total_chars
        // 4
    )

    result = {
        "version": "0.2.0",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "target_tokens": target_tokens,
        "target_tokens_human": format_tokens(
            target_tokens
        ),
        "approx_tokens": approx_tokens,
        "approx_tokens_human": format_tokens(
            approx_tokens
        ),
        "target_reached": (
            approx_tokens
            >= int(
                target_tokens
                * 0.95
            )
        ),
        "train_docs": train_docs,
        "validation_docs": validation_docs,
        "unique_docs": len(
            seen
        ),
        "rejected": dict(
            rejected
        ),
        "accepted_by_source": dict(
            accepted_by_source
        ),
        "characters_by_source": dict(
            source_chars
        ),
        "source_caps": SOURCE_CAPS,
        "raw_sources": [
            {
                "name": source.name,
                "kind": _source_kind(
                    source
                ),
                "size_bytes": source.stat().st_size,
                "size_human": format_bytes(
                    source.stat().st_size
                ),
            }
            for source in raw_sources
        ],
        "train_file": str(
            TRAIN_FILE
        ),
        "validation_file": str(
            VALIDATION_FILE
        ),
        "train_size_bytes": (
            TRAIN_FILE.stat().st_size
            if TRAIN_FILE.exists()
            else 0
        ),
        "validation_size_bytes": (
            VALIDATION_FILE.stat().st_size
            if VALIDATION_FILE.exists()
            else 0
        ),
        "notes": (
            "Amber 0.2 utilise le tokenizer 32k existant pour rester "
            "compatible avec les poids Amber 0.1. Wiktionnaire et Wikiquote "
            "sont fortement plafonnés; Wikipedia/Wikisource sont privilégiés."
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        (
            "[V02 DATASET COMPLETE] "
            f"~{approx_tokens:,} tokens | "
            f"train_docs={train_docs:,} | "
            f"val_docs={validation_docs:,} | "
            f"rejected={sum(rejected.values()):,}"
        ),
        flush=True,
    )

    if not result[
        "target_reached"
    ]:
        print(
            (
                "[V02 DATASET WARNING] Cible non atteinte. "
                "Ajoute en priorité Wikipedia FR puis Wikisource FR "
                "dans l'onglet Dataset."
            ),
            flush=True,
        )

    return result


def load_v02_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return {}

    try:
        return json.loads(
            MANIFEST_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare",
        action="store_true",
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
    )

    args = parser.parse_args()

    if args.prepare:
        def progress(info):
            if info.get(
                "stage"
            ) == "source":
                print(
                    (
                        "[V02 DATASET] "
                        f"{info['source_index']}/{info['source_count']} "
                        f"{info['source']} ({info['kind']}) | "
                        f"~{info['approx_tokens']:,} tokens"
                    ),
                    flush=True,
                )

            elif (
                info.get(
                    "accepted_docs",
                    0
                )
                % 10_000
                == 0
            ):
                print(
                    (
                        "[V02 DATASET] "
                        f"docs={info['accepted_docs']:,} | "
                        f"rejetés={info.get('rejected_docs', 0):,} | "
                        f"~{info['approx_tokens']:,} tokens"
                    ),
                    flush=True,
                )

        prepare_v02_dataset(
            target_tokens=args.target_tokens,
            progress_callback=progress,
        )

        return

    print(
        json.dumps(
            load_v02_manifest(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
