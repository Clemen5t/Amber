from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_FILE = DATA_DIR / "train.txt"
MANIFEST_FILE = DATA_DIR / "dataset_manifest.json"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def dataset_stats(path: Path = TRAIN_FILE) -> dict:
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
        }

    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    stat = path.stat()

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
    }


def write_manifest(path: Path = TRAIN_FILE) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stats = dataset_stats(path)

    manifest = {
        "amber_dataset_manifest_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tokenizer": {
            "name": "AmberByteTokenizer",
            "active_vocab_size": 259,
            "model_vocab_size": 4096,
        },
        "training_file": stats,
        "status": (
            "ready"
            if stats["exists"] and stats["utf8_bytes"] > 0
            else "missing"
        ),
        "next_target": (
            "Amber 0.1: tokenizer entraîne et corpus multi-source"
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
