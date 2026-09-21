from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from amber.dataset_manager import (
    PREPARED_TRAIN_FILE,
    TOKENIZER_DIR,
)


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILE = TOKENIZER_DIR / "amber_tokenizer.json"
TOKENIZER_META = TOKENIZER_DIR / "amber_tokenizer_meta.json"


def train_amber_tokenizer(
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    input_files: list[str | Path] | None = None,
    progress_callback=None,
) -> dict:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.normalizers import NFKC
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Le package 'tokenizers' n'est pas installé."
        ) from exc

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if input_files is None:
        input_files = [
            PREPARED_TRAIN_FILE
        ]

    files = [
        Path(path)
        for path in input_files
        if Path(path).exists()
        and Path(path).stat().st_size > 0
    ]

    if not files:
        raise RuntimeError(
            "Aucun dataset préparé. Lance d'abord "
            "'Préparer dataset' dans Amber."
        )

    if vocab_size < 512:
        raise ValueError(
            "Le vocabulaire doit contenir au moins 512 tokens."
        )

    if progress_callback:
        progress_callback(
            f"Entraînement tokenizer BPE sur {len(files)} fichier(s)..."
        )

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    tokenizer.normalizer = NFKC()

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=[
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
        ],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(
        [
            str(path)
            for path in files
        ],
        trainer
    )

    tokenizer.save(
        str(TOKENIZER_FILE)
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    metadata = {
        "name": "AmberBPETokenizer",
        "version": "0.1",
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": int(actual_vocab_size),
        "min_frequency": int(min_frequency),
        "special_tokens": {
            token: tokenizer.token_to_id(token)
            for token in [
                "<pad>",
                "<bos>",
                "<eos>",
                "<unk>",
            ]
        },
        "input_files": [
            str(path)
            for path in files
        ],
        "engine": "Hugging Face tokenizers library",
        "pretrained_weights_used": False,
        "pretrained_vocab_used": False,
    }

    TOKENIZER_META.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    if progress_callback:
        progress_callback(
            f"Tokenizer terminé : {actual_vocab_size:,} tokens."
        )

    return metadata


def tokenizer_status() -> dict:
    if not TOKENIZER_FILE.exists():
        return {
            "exists": False,
            "path": str(TOKENIZER_FILE),
            "vocab_size": 0,
            "metadata": {},
        }

    metadata = {}

    if TOKENIZER_META.exists():
        try:
            metadata = json.loads(
                TOKENIZER_META.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            metadata = {}

    return {
        "exists": True,
        "path": str(TOKENIZER_FILE),
        "vocab_size": int(
            metadata.get(
                "actual_vocab_size",
                0
            )
        ),
        "metadata": metadata,
    }
