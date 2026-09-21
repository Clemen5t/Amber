from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from training.data_v01 import (
    VALIDATION_BIN,
    TokenBin,
)
from tokenizer.amber_bpe import AmberBPETokenizer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "amber_v01_latest.pt"


def find_amber_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm / GPU non disponible."
        )

    for index in range(
        torch.cuda.device_count()
    ):
        name = torch.cuda.get_device_name(
            index
        )

        if "7900 XT" in name.upper():
            return torch.device(
                f"cuda:{index}"
            )

    raise RuntimeError(
        "AMD Radeon RX 7900 XT introuvable."
    )


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint Amber 0.1 introuvable."
        )

    device = find_amber_gpu()

    print(
        "[V01 EVAL] Chargement checkpoint...",
        flush=True
    )

    try:
        payload = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

    config_data = dict(
        payload.get(
            "config",
            {}
        )
    )

    config_data[
        "gradient_checkpointing"
    ] = False

    config = AmberV01Config(
        **config_data
    )

    model = AmberV01Model(
        config
    )

    model.load_state_dict(
        payload[
            "model_state"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    metadata = {
        "step": int(
            payload.get(
                "step",
                0
            )
        ),
        "tokens_seen": int(
            payload.get(
                "tokens_seen",
                0
            )
        ),
        "stored_train_loss": payload.get(
            "train_loss"
        ),
        "stored_val_loss": payload.get(
            "val_loss"
        ),
    }

    del payload

    return (
        model,
        device,
        metadata,
    )


@torch.no_grad()
def evaluate(
    *,
    model,
    device,
    batches=50,
    sequence_length=512,
    batch_size=4,
):
    validation = TokenBin(
        VALIDATION_BIN
    )

    if len(
        validation
    ) <= sequence_length + 1:
        raise RuntimeError(
            "Validation cache trop petit."
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    rng = np.random.default_rng(
        20260921
    )

    losses = []
    evaluated_tokens = 0

    started = time.perf_counter()

    for index in range(
        int(batches)
    ):
        x_np, y_np = validation.batch(
            batch_size,
            sequence_length,
            rng
        )

        x = torch.from_numpy(
            x_np
        ).to(
            device=device,
            dtype=torch.long
        )

        y = torch.from_numpy(
            y_np
        ).to(
            device=device,
            dtype=torch.long
        )

        with torch.autocast(
            device_type="cuda",
            dtype=dtype
        ):
            _, loss = model(
                x,
                y
            )

        losses.append(
            float(
                loss.item()
            )
        )

        evaluated_tokens += (
            batch_size
            * sequence_length
        )

        if (
            index == 0
            or (index + 1) % 10 == 0
            or index + 1 == batches
        ):
            print(
                (
                    "[V01 EVAL] "
                    f"{index + 1}/{batches} batches | "
                    f"loss moyenne="
                    f"{sum(losses) / len(losses):.4f}"
                ),
                flush=True
            )

    mean_loss = (
        sum(losses)
        / len(losses)
    )

    perplexity = math.exp(
        min(
            mean_loss,
            20.0
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    return {
        "validation_loss": mean_loss,
        "perplexity": perplexity,
        "evaluated_tokens": evaluated_tokens,
        "batches": int(batches),
        "batch_size": int(batch_size),
        "sequence_length": int(
            sequence_length
        ),
        "seconds": elapsed,
        "tokens_per_second": (
            evaluated_tokens
            / elapsed
            if elapsed > 0
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batches",
        type=int,
        default=50
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    args = parser.parse_args()

    tokenizer = AmberBPETokenizer()

    model, device, metadata = (
        load_model()
    )

    result = evaluate(
        model=model,
        device=device,
        batches=max(
            1,
            args.batches
        ),
        sequence_length=min(
            max(
                64,
                args.sequence_length
            ),
            model.config.context_length
        ),
        batch_size=max(
            1,
            args.batch_size
        ),
    )

    result[
        "vocab_size"
    ] = tokenizer.vocab_size

    result.update(
        metadata
    )

    print(
        "[V01 EVAL JSON] "
        + json.dumps(
            result,
            ensure_ascii=False
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
