from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from training.data_v01 import (
    TRAIN_BIN,
    TokenBin,
    load_cache_metadata,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "amber_v01_latest.pt"
AUTOTUNE_FILE = CHECKPOINT_DIR / "amber_v01_autotune.json"

SEQUENCE_LENGTH = 1024
TOKENS_PER_UPDATE = 16_384
SAFETY_RATIO = 0.90

CANDIDATES = [
    (True, 1, 16),
    (True, 2, 8),
    (True, 4, 4),
    (True, 8, 2),
    (True, 16, 1),
    (False, 1, 16),
    (False, 2, 8),
    (False, 4, 4),
    (False, 8, 2),
    (False, 16, 1),
]


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


def amp_dtype():
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass

    return torch.float16


def format_gb(value):
    return (
        float(value)
        / 1024**3
    )


def build_model(
    *,
    vocab_size,
    device,
):
    config = AmberV01Config(
        vocab_size=vocab_size,
        context_length=SEQUENCE_LENGTH,
        gradient_checkpointing=True,
    )

    model = AmberV01Model(
        config
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    if CHECKPOINT.exists():
        print(
            "[AUTOTUNE] Chargement du checkpoint courant...",
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

        model.load_state_dict(
            payload["model_state"]
        )

        if "optimizer_state" in payload:
            optimizer.load_state_dict(
                payload["optimizer_state"]
            )

        print(
            (
                "[AUTOTUNE] Checkpoint chargé : "
                f"{int(payload.get('tokens_seen', 0)):,} tokens."
            ),
            flush=True
        )

    else:
        print(
            "[AUTOTUNE] Aucun checkpoint : benchmark sur poids initiaux.",
            flush=True
        )

    return (
        model,
        optimizer,
        config,
    )


def run_update(
    *,
    model,
    optimizer,
    train_data,
    device,
    dtype,
    rng,
    micro_batch,
    accumulation,
):
    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0

    for _ in range(
        accumulation
    ):
        x_np, y_np = train_data.batch(
            micro_batch,
            SEQUENCE_LENGTH,
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

            scaled_loss = (
                loss
                / accumulation
            )

        scaled_loss.backward()

        total_loss += float(
            loss.detach().item()
        )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    return (
        total_loss
        / accumulation
    )


def main():
    metadata = load_cache_metadata()

    if not metadata:
        raise RuntimeError(
            "Cache Amber 0.1 absent."
        )

    device = find_amber_gpu()

    torch.cuda.set_device(
        device
    )

    total_memory = torch.cuda.get_device_properties(
        device
    ).total_memory

    memory_limit = int(
        total_memory
        * SAFETY_RATIO
    )

    dtype = amp_dtype()

    print(
        "=" * 76,
        flush=True
    )

    print(
        "AMBER 0.1.5 - RX 7900 XT AUTO-TUNER",
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )

    print(
        f"[AUTOTUNE] GPU : {torch.cuda.get_device_name(device)}",
        flush=True
    )

    print(
        (
            f"[AUTOTUNE] VRAM totale : "
            f"{format_gb(total_memory):.2f} GB"
        ),
        flush=True
    )

    print(
        (
            f"[AUTOTUNE] Limite sécurité : "
            f"{format_gb(memory_limit):.2f} GB "
            f"({SAFETY_RATIO * 100:.0f} %)"
        ),
        flush=True
    )

    print(
        (
            f"[AUTOTUNE] Tokens/update conservés : "
            f"{TOKENS_PER_UPDATE:,}"
        ),
        flush=True
    )

    model, optimizer, config = build_model(
        vocab_size=int(
            metadata["vocab_size"]
        ),
        device=device,
    )

    train_data = TokenBin(
        TRAIN_BIN
    )

    rng = np.random.default_rng(
        20260921
    )

    results = []

    for (
        checkpointing,
        micro_batch,
        accumulation
    ) in CANDIDATES:
        label = (
            f"mb={micro_batch} "
            f"accum={accumulation} "
            f"checkpoint={'ON' if checkpointing else 'OFF'}"
        )

        print(
            f"[AUTOTUNE] Test {label}...",
            flush=True
        )

        model.config.gradient_checkpointing = (
            checkpointing
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

        try:
            # Warm-up non chronométré.
            run_update(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                device=device,
                dtype=dtype,
                rng=rng,
                micro_batch=micro_batch,
                accumulation=accumulation,
            )

            torch.cuda.synchronize(
                device
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

            started = time.perf_counter()

            loss = run_update(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                device=device,
                dtype=dtype,
                rng=rng,
                micro_batch=micro_batch,
                accumulation=accumulation,
            )

            torch.cuda.synchronize(
                device
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            peak_allocated = torch.cuda.max_memory_allocated(
                device
            )

            peak_reserved = torch.cuda.max_memory_reserved(
                device
            )

            speed = (
                TOKENS_PER_UPDATE
                / elapsed
            )

            safe = (
                peak_reserved
                <= memory_limit
            )

            result = {
                "checkpointing": bool(
                    checkpointing
                ),
                "micro_batch": int(
                    micro_batch
                ),
                "gradient_accumulation": int(
                    accumulation
                ),
                "tokens_per_update": TOKENS_PER_UPDATE,
                "seconds_per_update": elapsed,
                "tokens_per_second": speed,
                "loss": loss,
                "peak_allocated_bytes": int(
                    peak_allocated
                ),
                "peak_reserved_bytes": int(
                    peak_reserved
                ),
                "safe": bool(
                    safe
                ),
            }

            results.append(
                result
            )

            print(
                (
                    f"[AUTOTUNE] OK {label} | "
                    f"{speed:,.0f} tok/s | "
                    f"alloc={format_gb(peak_allocated):.2f} GB | "
                    f"reserved={format_gb(peak_reserved):.2f} GB | "
                    f"{'SAFE' if safe else 'TOO HIGH'}"
                ),
                flush=True
            )

        except torch.cuda.OutOfMemoryError:
            print(
                f"[AUTOTUNE] OOM {label}",
                flush=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            torch.cuda.empty_cache()

        except RuntimeError as exc:
            text = str(
                exc
            ).lower()

            if (
                "out of memory" in text
                or "memory" in text
                and "alloc" in text
            ):
                print(
                    f"[AUTOTUNE] OOM {label}",
                    flush=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                torch.cuda.empty_cache()

            else:
                raise

    safe_results = [
        item
        for item in results
        if item["safe"]
    ]

    if not safe_results:
        raise RuntimeError(
            "Aucun profil Auto-Tuner sûr."
        )

    best = max(
        safe_results,
        key=lambda item: item[
            "tokens_per_second"
        ]
    )

    output = {
        "version": "0.1.5",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "gpu": torch.cuda.get_device_name(
            device
        ),
        "total_vram_bytes": int(
            total_memory
        ),
        "safety_ratio": SAFETY_RATIO,
        "sequence_length": SEQUENCE_LENGTH,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "best": best,
        "results": results,
    }

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AUTOTUNE_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "=" * 76,
        flush=True
    )

    print(
        (
            "[AUTOTUNE BEST] "
            f"micro_batch={best['micro_batch']} | "
            f"grad_accum={best['gradient_accumulation']} | "
            f"checkpoint={'ON' if best['checkpointing'] else 'OFF'} | "
            f"speed={best['tokens_per_second']:,.0f} tok/s | "
            f"reserved={format_gb(best['peak_reserved_bytes']):.2f} GB"
        ),
        flush=True
    )

    print(
        (
            "[AUTOTUNE] Profil sauvegardé : "
            f"{AUTOTUNE_FILE.name}"
        ),
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )


if __name__ == "__main__":
    main()
