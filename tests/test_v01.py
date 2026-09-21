from __future__ import annotations

import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from tokenizer.amber_bpe import AmberBPETokenizer


def find_amber_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm / GPU non disponible."
        )

    for i in range(
        torch.cuda.device_count()
    ):
        name = torch.cuda.get_device_name(i)

        if "7900 XT" in name.upper():
            return torch.device(
                f"cuda:{i}"
            )

    raise RuntimeError(
        "AMD Radeon RX 7900 XT introuvable."
    )


def main():
    print(
        "=" * 68
    )

    print(
        "AMBER 0.1.0 - MODEL SMOKE TEST"
    )

    print(
        "=" * 68
    )

    tokenizer = AmberBPETokenizer()

    config = AmberV01Config(
        vocab_size=tokenizer.vocab_size,
        gradient_checkpointing=False,
    )

    device = find_amber_gpu()

    model = AmberV01Model(
        config
    ).to(
        device
    )

    params = model.parameter_count()

    print(
        "GPU        :",
        torch.cuda.get_device_name(
            device
        )
    )

    print(
        f"Vocab      : {tokenizer.vocab_size:,}"
    )

    print(
        f"Parameters : {params:,}"
    )

    print(
        f"Context    : {config.context_length}"
    )

    print(
        f"Layers     : {config.n_layers}"
    )

    print(
        f"Model dim  : {config.d_model}"
    )

    print(
        f"Heads      : {config.n_heads}"
    )

    print(
        f"KV heads   : {config.n_kv_heads}"
    )

    x = torch.randint(
        0,
        tokenizer.vocab_size,
        (
            1,
            64
        ),
        device=device
    )

    y = torch.randint(
        0,
        tokenizer.vocab_size,
        (
            1,
            64
        ),
        device=device
    )

    model.train()

    logits, loss = model(
        x,
        y
    )

    loss.backward()

    allocated = (
        torch.cuda.memory_allocated(
            device
        )
        / 1024**3
    )

    print(
        f"Logits     : {tuple(logits.shape)}"
    )

    print(
        f"Loss       : {loss.item():.4f}"
    )

    print(
        f"VRAM       : {allocated:.2f} GB"
    )

    print(
        ""
    )

    print(
        "V01 FORWARD  : OK"
    )

    print(
        "V01 LOSS     : OK"
    )

    print(
        "V01 BACKPROP : OK"
    )

    print(
        "=" * 68
    )


if __name__ == "__main__":
    main()
