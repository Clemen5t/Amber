from __future__ import annotations

import argparse
import json
import time
from array import array
from datetime import datetime
from pathlib import Path

import numpy as np

from amber.dataset_manager import (
    PREPARED_TRAIN_FILE,
    PREPARED_VALIDATION_FILE,
    format_bytes,
    format_tokens,
)
from tokenizer.amber_bpe import AmberBPETokenizer


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "v01_cache"

TRAIN_BIN = CACHE_DIR / "train.uint16.bin"
VALIDATION_BIN = CACHE_DIR / "validation.uint16.bin"
CACHE_META = CACHE_DIR / "meta.json"


def _encode_file(
    source: Path,
    destination: Path,
    tokenizer: AmberBPETokenizer,
    label: str,
):
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset préparé introuvable : {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_bytes = source.stat().st_size
    consumed_bytes = 0
    token_count = 0
    document_count = 0

    started = time.time()

    with source.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as inp, destination.open(
        "wb"
    ) as out:
        buffer = []

        for line in inp:
            encoded_line = line.encode(
                "utf-8",
                errors="replace"
            )

            consumed_bytes += len(
                encoded_line
            )

            if line.strip():
                buffer.append(
                    line.rstrip("\n")
                )

            elif buffer:
                text = "\n".join(
                    buffer
                ).strip()

                buffer = []

                if not text:
                    continue

                ids = tokenizer.encode(
                    text,
                    add_bos=False,
                    add_eos=True
                )

                array(
                    "H",
                    ids
                ).tofile(out)

                token_count += len(ids)
                document_count += 1

                if document_count % 1000 == 0:
                    elapsed = (
                        time.time()
                        - started
                    )

                    percent = (
                        consumed_bytes
                        / total_bytes
                        * 100
                        if total_bytes
                        else 0
                    )

                    speed = (
                        token_count
                        / elapsed
                        if elapsed > 0
                        else 0
                    )

                    print(
                        f"[CACHE {label}] "
                        f"{percent:6.2f}% | "
                        f"docs={document_count:,} | "
                        f"tokens={token_count:,} | "
                        f"{speed:,.0f} tok/s",
                        flush=True
                    )

        if buffer:
            text = "\n".join(
                buffer
            ).strip()

            if text:
                ids = tokenizer.encode(
                    text,
                    add_bos=False,
                    add_eos=True
                )

                array(
                    "H",
                    ids
                ).tofile(out)

                token_count += len(ids)
                document_count += 1

    return {
        "source": str(source),
        "destination": str(destination),
        "documents": document_count,
        "tokens": token_count,
        "tokens_human": format_tokens(
            token_count
        ),
        "source_size_bytes": total_bytes,
        "source_size_human": format_bytes(
            total_bytes
        ),
        "cache_size_bytes": (
            destination.stat().st_size
        ),
        "cache_size_human": format_bytes(
            destination.stat().st_size
        ),
    }


def build_v01_cache():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = AmberBPETokenizer()

    if tokenizer.vocab_size > 65535:
        raise RuntimeError(
            "Le cache uint16 nécessite vocab_size <= 65535."
        )

    print(
        "=" * 72,
        flush=True
    )

    print(
        "AMBER 0.1 TOKEN CACHE BUILDER",
        flush=True
    )

    print(
        f"Tokenizer : {tokenizer.vocab_size:,} tokens",
        flush=True
    )

    print(
        "=" * 72,
        flush=True
    )

    train = _encode_file(
        PREPARED_TRAIN_FILE,
        TRAIN_BIN,
        tokenizer,
        "TRAIN"
    )

    validation = _encode_file(
        PREPARED_VALIDATION_FILE,
        VALIDATION_BIN,
        tokenizer,
        "VAL"
    )

    metadata = {
        "version": "0.1.0",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "train": train,
        "validation": validation,
        "total_tokens": (
            train["tokens"]
            + validation["tokens"]
        ),
    }

    CACHE_META.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "",
        flush=True
    )

    print(
        f"TRAIN tokens      : {train['tokens']:,}",
        flush=True
    )

    print(
        f"VALIDATION tokens : {validation['tokens']:,}",
        flush=True
    )

    print(
        f"TOTAL tokens      : "
        f"{metadata['total_tokens']:,}",
        flush=True
    )

    print(
        f"Cache train       : "
        f"{train['cache_size_human']}",
        flush=True
    )

    print(
        "[CACHE COMPLETE]",
        flush=True
    )

    return metadata


def load_cache_metadata():
    if not CACHE_META.exists():
        return None

    try:
        return json.loads(
            CACHE_META.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


class TokenBin:
    def __init__(
        self,
        path: Path
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                str(self.path)
            )

        self.data = np.memmap(
            self.path,
            dtype=np.uint16,
            mode="r"
        )

    def __len__(self):
        return int(
            self.data.shape[0]
        )

    def batch(
        self,
        batch_size: int,
        sequence_length: int,
        rng: np.random.Generator,
    ):
        max_start = (
            len(self)
            - sequence_length
            - 1
        )

        if max_start <= 0:
            raise RuntimeError(
                "Cache tokens trop petit."
            )

        starts = rng.integers(
            0,
            max_start,
            size=batch_size
        )

        x = np.empty(
            (
                batch_size,
                sequence_length
            ),
            dtype=np.int64
        )

        y = np.empty_like(
            x
        )

        for row, start in enumerate(
            starts
        ):
            block = np.asarray(
                self.data[
                    start:
                    start + sequence_length + 1
                ],
                dtype=np.int64
            )

            x[row] = block[:-1]
            y[row] = block[1:]

        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        action="store_true"
    )

    args = parser.parse_args()

    if args.build:
        build_v01_cache()
        return

    meta = load_cache_metadata()

    if meta:
        print(
            json.dumps(
                meta,
                ensure_ascii=False,
                indent=2
            )
        )
    else:
        print(
            "Aucun cache Amber 0.1."
        )


if __name__ == "__main__":
    main()
