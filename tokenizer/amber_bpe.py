from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "data" / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "amber_tokenizer.json"
TOKENIZER_META = TOKENIZER_DIR / "amber_tokenizer_meta.json"


class AmberBPETokenizer:
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer_path: str | Path = TOKENIZER_FILE):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'tokenizers' n'est pas installé. "
                "Installe Amber via l'Updater 0.0.6."
            ) from exc

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer Amber introuvable : {tokenizer_path}"
            )

        self.path = tokenizer_path
        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.pad_id = self.tokenizer.token_to_id(
            self.PAD_TOKEN
        )
        self.bos_id = self.tokenizer.token_to_id(
            self.BOS_TOKEN
        )
        self.eos_id = self.tokenizer.token_to_id(
            self.EOS_TOKEN
        )
        self.unk_id = self.tokenizer.token_to_id(
            self.UNK_TOKEN
        )

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.tokenizer.encode(
            text
        ).ids

        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids

        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]

        return ids

    def decode(
        self,
        ids
    ) -> str:
        return self.tokenizer.decode(
            [int(x) for x in ids],
            skip_special_tokens=True
        )

    @classmethod
    def metadata(cls) -> dict:
        if not TOKENIZER_META.exists():
            return {}

        try:
            return json.loads(
                TOKENIZER_META.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return {}
