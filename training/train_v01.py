from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from training.data_v01 import (
    TRAIN_BIN,
    VALIDATION_BIN,
    TokenBin,
    load_cache_metadata,
)
from training.governor import TrainingGovernor


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT = CHECKPOINT_DIR / "amber_v01_latest.pt"
STATUS_FILE = CHECKPOINT_DIR / "amber_v01_status.json"
TEMP_CHECKPOINT = CHECKPOINT_DIR / "amber_v01_latest.tmp"
STOP_FILE = ROOT / "training" / ".v01_stop_requested"
PAUSE_FILE = ROOT / "training" / ".v01_pause_requested"


PROFILES = {
    "eco": {
        "sequence_length": 512,
        "micro_batch": 1,
        "gradient_accumulation": 8,
        "learning_rate": 2.5e-4,
    },
    "balanced": {
        "sequence_length": 768,
        "micro_batch": 1,
        "gradient_accumulation": 12,
        "learning_rate": 3.0e-4,
    },
    "full": {
        "sequence_length": 1024,
        "micro_batch": 1,
        "gradient_accumulation": 16,
        "learning_rate": 3.0e-4,
    },
}


def clear_file(path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


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


def amp_setup():
    bf16_supported = False

    try:
        bf16_supported = bool(
            torch.cuda.is_bf16_supported()
        )
    except Exception:
        bf16_supported = False

    amp_dtype = (
        torch.bfloat16
        if bf16_supported
        else torch.float16
    )

    scaler_enabled = (
        amp_dtype == torch.float16
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled
        )

    return (
        amp_dtype,
        scaler
    )


def learning_rate(
    *,
    tokens_seen,
    target_tokens,
    peak_lr,
):
    if target_tokens <= 0:
        return peak_lr

    progress = min(
        max(
            tokens_seen / target_tokens,
            0.0
        ),
        1.0
    )

    warmup = 0.02

    if progress < warmup:
        return peak_lr * (
            progress
            / warmup
        )

    decay_progress = (
        progress - warmup
    ) / (
        1.0 - warmup
    )

    min_lr = peak_lr * 0.10

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi
            * decay_progress
        )
    )

    return (
        min_lr
        + (
            peak_lr
            - min_lr
        )
        * cosine
    )


def write_status(
    *,
    step,
    tokens_seen,
    train_loss,
    val_loss,
    profile,
    target_tokens=None,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        "version": "0.1.3",
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "train_loss": (
            float(train_loss)
            if train_loss is not None
            else None
        ),
        "val_loss": (
            float(val_loss)
            if val_loss is not None
            else None
        ),
        "profile": profile,
        "target_tokens": (
            int(target_tokens)
            if target_tokens is not None
            else None
        ),
    }

    STATUS_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def save_checkpoint(
    *,
    model,
    optimizer,
    scaler,
    config,
    step,
    tokens_seen,
    train_loss,
    val_loss,
    profile,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if int(step) <= 0 or int(tokens_seen) <= 0:
        write_status(
            step=0,
            tokens_seen=0,
            train_loss=None,
            val_loss=None,
            profile=profile,
        )

        print(
            "[V01 CHECKPOINT] Aucun token entraîné : checkpoint lourd non sauvegardé.",
            flush=True
        )
        return

    payload = {
        "amber_version": "0.1.3",
        "config": config.to_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "train_loss": (
            float(train_loss)
            if train_loss is not None
            else None
        ),
        "val_loss": (
            float(val_loss)
            if val_loss is not None
            else None
        ),
        "profile": profile,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
    }

    torch.save(
        payload,
        TEMP_CHECKPOINT
    )

    TEMP_CHECKPOINT.replace(
        CHECKPOINT
    )

    write_status(
        step=step,
        tokens_seen=tokens_seen,
        train_loss=train_loss,
        val_loss=val_loss,
        profile=profile,
    )

    print(
        (
            f"[V01 CHECKPOINT] "
            f"step={step} "
            f"tokens={tokens_seen:,}"
        ),
        flush=True
    )


def load_checkpoint(
    *,
    model,
    optimizer,
    scaler,
    device,
):
    if not CHECKPOINT.exists():
        print(
            "[V01] Aucun checkpoint existant : démarrage depuis zéro.",
            flush=True
        )

        return {
            "step": 0,
            "tokens_seen": 0,
            "train_loss": None,
            "val_loss": None,
        }

    # Le sidecar JSON est minuscule : on le lit avant le gros fichier .pt.
    status = {}

    if STATUS_FILE.exists():
        try:
            status = json.loads(
                STATUS_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            status = {}

    status_step = int(
        status.get(
            "step",
            -1
        )
    )

    status_tokens = int(
        status.get(
            "tokens_seen",
            -1
        )
    )

    if status_step == 0 and status_tokens == 0:
        size_mb = (
            CHECKPOINT.stat().st_size
            / 1024**2
        )

        print(
            (
                "[V01] Checkpoint vide détecté "
                f"({size_mb:.1f} MB, 0 token). "
                "Il est inutile et sera ignoré."
            ),
            flush=True
        )

        try:
            CHECKPOINT.unlink()
            print(
                "[V01] Ancien checkpoint vide supprimé.",
                flush=True
            )
        except Exception as exc:
            print(
                f"[V01] Impossible de supprimer le checkpoint vide : {exc}",
                flush=True
            )

        return {
            "step": 0,
            "tokens_seen": 0,
            "train_loss": None,
            "val_loss": None,
        }

    size_mb = (
        CHECKPOINT.stat().st_size
        / 1024**2
    )

    print(
        (
            "[V01] Chargement checkpoint "
            f"({size_mb:.1f} MB) vers la RAM..."
        ),
        flush=True
    )

    load_started = time.time()
    load_done = threading.Event()

    def checkpoint_heartbeat():
        while not load_done.wait(5):
            elapsed = (
                time.time()
                - load_started
            )

            print(
                (
                    "[V01] Chargement checkpoint toujours en cours... "
                    f"{elapsed:.0f}s"
                ),
                flush=True
            )

    heartbeat_thread = threading.Thread(
        target=checkpoint_heartbeat,
        daemon=True
    )

    heartbeat_thread.start()

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

    finally:
        load_done.set()

    print(
        (
            "[V01] Checkpoint lu en "
            f"{time.time() - load_started:.1f}s. "
            "Chargement des poids sur le GPU..."
        ),
        flush=True
    )

    model.load_state_dict(
        payload["model_state"]
    )

    print(
        "[V01] Poids modèle restaurés.",
        flush=True
    )

    if "optimizer_state" in payload:
        print(
            "[V01] Restauration optimizer...",
            flush=True
        )

        optimizer.load_state_dict(
            payload["optimizer_state"]
        )

        print(
            "[V01] Optimizer restauré.",
            flush=True
        )

    if "scaler_state" in payload:
        try:
            scaler.load_state_dict(
                payload["scaler_state"]
            )
        except Exception:
            pass

    restored_step = int(
        payload.get(
            "step",
            0
        )
    )

    restored_tokens = int(
        payload.get(
            "tokens_seen",
            0
        )
    )

    print(
        (
            "[V01] Reprise prête : "
            f"step {restored_step}, "
            f"{restored_tokens:,} tokens."
        ),
        flush=True
    )

    return {
        "step": restored_step,
        "tokens_seen": restored_tokens,
        "train_loss": payload.get(
            "train_loss"
        ),
        "val_loss": payload.get(
            "val_loss"
        ),
    }


@torch.no_grad()
def validation_loss(
    *,
    model,
    validation_data,
    device,
    amp_dtype,
    sequence_length,
    batches=8,
):
    if (
        validation_data is None
        or len(validation_data)
        <= sequence_length + 1
    ):
        return None

    model.eval()

    rng = np.random.default_rng(
        20260921
    )

    losses = []

    for _ in range(
        batches
    ):
        x_np, y_np = validation_data.batch(
            1,
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
            dtype=amp_dtype
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

    model.train()

    return sum(losses) / len(
        losses
    )


def format_eta(seconds):
    if seconds is None:
        return "--:--:--"

    seconds = max(
        0,
        int(seconds)
    )

    hours, remainder = divmod(
        seconds,
        3600
    )

    minutes, seconds = divmod(
        remainder,
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=tuple(
            PROFILES.keys()
        ),
        default="balanced"
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=100_000_000
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore et archive un checkpoint local orphelin/0-token "
            "avant de démarrer."
        )
    )

    args = parser.parse_args()

    metadata = load_cache_metadata()

    if not metadata:
        raise RuntimeError(
            "Cache tokens Amber 0.1 absent. "
            "Construis-le d'abord depuis l'application."
        )

    vocab_size = int(
        metadata["vocab_size"]
    )

    target_tokens = int(
        args.target_tokens
    )

    if target_tokens <= 0:
        raise ValueError(
            "target-tokens doit être > 0."
        )

    profile = PROFILES[
        args.mode
    ]

    device = find_amber_gpu()

    torch.cuda.set_device(
        device
    )

    print(
        f"[V01] GPU sélectionné : {torch.cuda.get_device_name(device)}",
        flush=True
    )

    config = AmberV01Config(
        vocab_size=vocab_size
    )

    print(
        "[V01] Initialisation du modèle 100 M paramètres...",
        flush=True
    )

    model_started = time.time()

    model = AmberV01Model(
        config
    ).to(
        device
    )

    print(
        (
            "[V01] Modèle initialisé en "
            f"{time.time() - model_started:.1f}s."
        ),
        flush=True
    )

    print(
        "[V01] Initialisation optimizer...",
        flush=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile[
            "learning_rate"
        ],
        betas=(
            0.9,
            0.95
        ),
        eps=1e-8,
        weight_decay=0.1,
    )

    print(
        "[V01] Optimizer prêt.",
        flush=True
    )

    amp_dtype, scaler = (
        amp_setup()
    )

    print(
        f"[V01] Mixed precision : {amp_dtype}.",
        flush=True
    )

    if args.fresh and CHECKPOINT.exists():
        archive_name = (
            "amber_v01_abandoned_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".pt"
        )

        archive_path = (
            CHECKPOINT_DIR
            / archive_name
        )

        try:
            CHECKPOINT.replace(
                archive_path
            )

            print(
                (
                    "[V01] Ancien checkpoint sans progression "
                    f"archivé : {archive_name}"
                ),
                flush=True
            )

        except Exception as exc:
            raise RuntimeError(
                "Impossible d'archiver l'ancien checkpoint : "
                + str(exc)
            )

        try:
            if STATUS_FILE.exists():
                STATUS_FILE.unlink()
        except Exception:
            pass

    state = load_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    step = state[
        "step"
    ]

    tokens_seen = state[
        "tokens_seen"
    ]

    last_train_loss = state[
        "train_loss"
    ]

    last_val_loss = state[
        "val_loss"
    ]

    if tokens_seen >= target_tokens:
        print(
            (
                "[V01] Objectif déjà atteint : "
                f"{tokens_seen:,} / "
                f"{target_tokens:,} tokens."
            ),
            flush=True
        )
        return

    train_data = TokenBin(
        TRAIN_BIN
    )

    validation_data = (
        TokenBin(
            VALIDATION_BIN
        )
        if VALIDATION_BIN.exists()
        else None
    )

    clear_file(
        STOP_FILE
    )

    clear_file(
        PAUSE_FILE
    )

    governor = TrainingGovernor(
        args.mode
    )

    rng = np.random.default_rng(
        int(
            time.time()
        )
        & 0xFFFFFFFF
    )

    params = model.parameter_count()

    sequence_length = profile[
        "sequence_length"
    ]

    micro_batch = profile[
        "micro_batch"
    ]

    accumulation = profile[
        "gradient_accumulation"
    ]

    effective_tokens = (
        sequence_length
        * micro_batch
        * accumulation
    )

    print(
        "=" * 76,
        flush=True
    )

    print(
        "AMBER 0.1.3 PRETRAINER",
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(device)}",
        flush=True
    )

    print(
        f"Parameters   : {params:,}",
        flush=True
    )

    print(
        f"Vocab        : {vocab_size:,}",
        flush=True
    )

    print(
        f"Context      : {config.context_length}",
        flush=True
    )

    print(
        f"Profile      : {args.mode.upper()}",
        flush=True
    )

    print(
        f"Sequence     : {sequence_length}",
        flush=True
    )

    print(
        f"Grad accum   : {accumulation}",
        flush=True
    )

    print(
        f"AMP          : {amp_dtype}",
        flush=True
    )

    print(
        f"Dataset train: {len(train_data):,} tokens",
        flush=True
    )

    print(
        f"Target       : {target_tokens:,} tokens",
        flush=True
    )

    print(
        f"Resume       : step {step}, "
        f"{tokens_seen:,} tokens",
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )

    started = time.time()
    session_tokens_start = tokens_seen

    while tokens_seen < target_tokens:

        if STOP_FILE.exists():
            clear_file(
                STOP_FILE
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                step=step,
                tokens_seen=tokens_seen,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                profile=args.mode,
            )

            print(
                "[V01] Arrêt propre terminé.",
                flush=True
            )
            return

        if PAUSE_FILE.exists():
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                step=step,
                tokens_seen=tokens_seen,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                profile=args.mode,
            )

            print(
                "[V01] En pause.",
                flush=True
            )

            while PAUSE_FILE.exists():
                if STOP_FILE.exists():
                    clear_file(
                        STOP_FILE
                    )

                    clear_file(
                        PAUSE_FILE
                    )

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        step=step,
                        tokens_seen=tokens_seen,
                        train_loss=last_train_loss,
                        val_loss=last_val_loss,
                        profile=args.mode,
                    )

                    print(
                        "[V01] Arrêt propre terminé.",
                        flush=True
                    )
                    return

                time.sleep(
                    0.5
                )

            print(
                "[V01] Reprise.",
                flush=True
            )

        game = governor.active_game()

        if game:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                step=step,
                tokens_seen=tokens_seen,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                profile=args.mode,
            )

            print(
                (
                    f"[V01 GOVERNOR] "
                    f"Jeu détecté : {game}"
                ),
                flush=True
            )

            while governor.active_game():
                if STOP_FILE.exists():
                    clear_file(
                        STOP_FILE
                    )

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        step=step,
                        tokens_seen=tokens_seen,
                        train_loss=last_train_loss,
                        val_loss=last_val_loss,
                        profile=args.mode,
                    )

                    print(
                        "[V01] Arrêt propre terminé.",
                        flush=True
                    )
                    return

                time.sleep(
                    2
                )

            print(
                "[V01 GOVERNOR] Reprise.",
                flush=True
            )

        model.train()

        lr = learning_rate(
            tokens_seen=min(
                target_tokens,
                tokens_seen + effective_tokens
            ),
            target_tokens=target_tokens,
            peak_lr=profile[
                "learning_rate"
            ],
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0
        actual_micro_steps = 0

        for _ in range(
            accumulation
        ):
            if (
                step == state["step"]
                and actual_micro_steps in {
                    0,
                    3,
                    6,
                    9
                }
            ):
                print(
                    (
                        "[V01] Premier step en cours : "
                        f"micro-batch {actual_micro_steps + 1}/"
                        f"{accumulation}"
                    ),
                    flush=True
                )

            if (
                tokens_seen
                + (
                    sequence_length
                    * micro_batch
                    * actual_micro_steps
                )
                >= target_tokens
                and actual_micro_steps > 0
            ):
                break

            x_np, y_np = train_data.batch(
                micro_batch,
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
                dtype=amp_dtype
            ):
                _, loss = model(
                    x,
                    y
                )

                scaled_loss = (
                    loss
                    / accumulation
                )

            scaler.scale(
                scaled_loss
            ).backward()

            accumulated_loss += float(
                loss.detach().item()
            )

            actual_micro_steps += 1

        if actual_micro_steps == 0:
            break

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        step += 1

        new_tokens = (
            actual_micro_steps
            * micro_batch
            * sequence_length
        )

        tokens_seen += new_tokens

        last_train_loss = (
            accumulated_loss
            / actual_micro_steps
        )

        if (
            step <= 5
            or step % 10 == 0
            or tokens_seen >= target_tokens
        ):
            if (
                step % 100 == 0
                or (
                    last_val_loss is None
                    and step >= 5
                )
            ):
                try:
                    last_val_loss = validation_loss(
                        model=model,
                        validation_data=validation_data,
                        device=device,
                        amp_dtype=amp_dtype,
                        sequence_length=min(
                            sequence_length,
                            512
                        ),
                        batches=4,
                    )
                except Exception:
                    last_val_loss = None

            elapsed = (
                time.time()
                - started
            )

            session_tokens = (
                tokens_seen
                - session_tokens_start
            )

            tok_per_second = (
                session_tokens
                / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = max(
                0,
                target_tokens
                - tokens_seen
            )

            eta = (
                remaining
                / tok_per_second
                if tok_per_second > 0
                else None
            )

            vram = (
                torch.cuda.memory_allocated(
                    device
                )
                / 1024**3
            )

            val_text = (
                f"{last_val_loss:.4f}"
                if last_val_loss is not None
                else "-"
            )

            print(
                (
                    f"[V01 step={step:06d}] "
                    f"loss={last_train_loss:.4f} | "
                    f"val={val_text} | "
                    f"tokens={tokens_seen:,}/"
                    f"{target_tokens:,} | "
                    f"speed={tok_per_second:,.0f} tok/s | "
                    f"lr={lr:.2e} | "
                    f"vram={vram:.2f} GB | "
                    f"eta={format_eta(eta)}"
                ),
                flush=True
            )

        if (
            step % 250 == 0
            or tokens_seen >= target_tokens
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                step=step,
                tokens_seen=tokens_seen,
                train_loss=last_train_loss,
                val_loss=last_val_loss,
                profile=args.mode,
            )

        governor.throttle()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        step=step,
        tokens_seen=tokens_seen,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        profile=args.mode,
    )

    print(
        "=" * 76,
        flush=True
    )

    print(
        "AMBER 0.1 PRETRAINING TARGET REACHED",
        flush=True
    )

    print(
        f"Tokens seen : {tokens_seen:,}",
        flush=True
    )

    print(
        f"Steps       : {step:,}",
        flush=True
    )

    print(
        f"Train loss  : {last_train_loss}",
        flush=True
    )

    print(
        f"Val loss    : {last_val_loss}",
        flush=True
    )

    print(
        "=" * 76,
        flush=True
    )


if __name__ == "__main__":
    main()
