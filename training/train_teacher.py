from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from teacher.curriculum import (
    TRAIN_FILE,
    REMEDIAL_FILE,
)
from tokenizer.amber_bpe import AmberBPETokenizer
from training.governor import TrainingGovernor


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"

BASE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "amber_v02_latest.pt"
)

CHECKPOINT = (
    CHECKPOINT_DIR
    / "amber_teacher_latest.pt"
)

TEMP_CHECKPOINT = (
    CHECKPOINT_DIR
    / "amber_teacher_latest.tmp"
)

STATUS_FILE = (
    CHECKPOINT_DIR
    / "amber_teacher_status.json"
)

STOP_FILE = (
    ROOT
    / "training"
    / ".teacher_stop_requested"
)

PAUSE_FILE = (
    ROOT
    / "training"
    / ".teacher_pause_requested"
)

PROFILES = {
    "eco": {
        "micro_batch": 4,
        "gradient_accumulation": 4,
    },
    "balanced": {
        "micro_batch": 8,
        "gradient_accumulation": 2,
    },
    "full": {
        "micro_batch": 16,
        "gradient_accumulation": 1,
    },
}


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


def clear_file(
    path: Path
):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def load_jsonl(
    path: Path
):
    if not path.exists():
        raise FileNotFoundError(
            (
                "Cours Amber Teacher introuvables. "
                "Clique d'abord sur « Construire les cours »."
            )
        )

    items = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            user = str(
                item.get(
                    "user",
                    ""
                )
            ).strip()

            assistant = str(
                item.get(
                    "assistant",
                    ""
                )
            ).strip()

            if user and assistant:
                items.append(
                    item
                )

    if not items:
        raise RuntimeError(
            "Aucun cours valide."
        )

    return items


def encode_example(
    tokenizer,
    item,
):
    prefix = (
        "Utilisateur : "
        + item["user"].strip()
        + "\nAssistant : "
    )

    prefix_ids = tokenizer.encode(
        prefix,
        add_bos=True,
        add_eos=False,
    )

    answer_ids = tokenizer.encode(
        item["assistant"].strip(),
        add_bos=False,
        add_eos=False,
    )

    if tokenizer.eos_id is None:
        raise RuntimeError(
            "Le tokenizer Amber n'a pas de token EOS."
        )

    ids = (
        prefix_ids
        + answer_ids
        + [
            int(
                tokenizer.eos_id
            )
        ]
    )

    # False = token dont la prédiction ne doit pas contribuer à la loss.
    trainable = (
        [
            False
        ]
        * len(
            prefix_ids
        )
        + [
            True
        ]
        * (
            len(
                answer_ids
            )
            + 1
        )
    )

    return (
        ids,
        trainable,
    )


def build_blocks(
    tokenizer,
    examples,
    *,
    sequence_length=512,
):
    pad_id = (
        int(
            tokenizer.pad_id
        )
        if tokenizer.pad_id is not None
        else 0
    )

    block_size = (
        sequence_length
        + 1
    )

    blocks = []
    ids_buffer = []
    mask_buffer = []

    for item in examples:
        ids, trainable = encode_example(
            tokenizer,
            item,
        )

        if len(
            ids
        ) > block_size:
            ids = ids[
                :block_size
            ]

            trainable = trainable[
                :block_size
            ]

        if (
            ids_buffer
            and len(
                ids_buffer
            )
            + len(
                ids
            )
            > block_size
        ):
            missing = (
                block_size
                - len(
                    ids_buffer
                )
            )

            ids_buffer.extend(
                [
                    pad_id
                ]
                * missing
            )

            mask_buffer.extend(
                [
                    False
                ]
                * missing
            )

            blocks.append(
                (
                    ids_buffer,
                    mask_buffer,
                )
            )

            ids_buffer = []
            mask_buffer = []

        ids_buffer.extend(
            ids
        )

        mask_buffer.extend(
            trainable
        )

    if ids_buffer:
        missing = (
            block_size
            - len(
                ids_buffer
            )
        )

        ids_buffer.extend(
            [
                pad_id
            ]
            * missing
        )

        mask_buffer.extend(
            [
                False
            ]
            * missing
        )

        blocks.append(
            (
                ids_buffer,
                mask_buffer,
            )
        )

    x = np.empty(
        (
            len(
                blocks
            ),
            sequence_length,
        ),
        dtype=np.int64,
    )

    y = np.empty_like(
        x
    )

    for row, (
        ids,
        mask
    ) in enumerate(
        blocks
    ):
        ids_array = np.asarray(
            ids,
            dtype=np.int64,
        )

        mask_array = np.asarray(
            mask,
            dtype=bool,
        )

        x[
            row
        ] = ids_array[
            :-1
        ]

        labels = ids_array[
            1:
        ].copy()

        target_mask = mask_array[
            1:
        ]

        labels[
            ~target_mask
        ] = -100

        y[
            row
        ] = labels

    return (
        x,
        y,
    )


def save_status(
    *,
    epoch,
    epochs,
    step,
    train_loss,
    examples,
    blocks,
    profile,
    epoch_complete=False,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": "0.1",
        "epoch": int(
            epoch
        ),
        "epochs": int(
            epochs
        ),
        "completed_epochs": int(
            epoch
            if epoch_complete
            else max(
                0,
                epoch - 1
            )
        ),
        "step": int(
            step
        ),
        "train_loss": (
            float(
                train_loss
            )
            if train_loss is not None
            else None
        ),
        "examples": int(
            examples
        ),
        "blocks": int(
            blocks
        ),
        "profile": profile,
    }

    STATUS_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_checkpoint(
    *,
    model,
    optimizer,
    config,
    epoch,
    epochs,
    step,
    train_loss,
    examples,
    blocks,
    profile,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "amber_version": "teacher-0.1",
        "base": "amber_v02_latest.pt",
        "config": config.to_dict(),
        "epoch": int(
            epoch
        ),
        "epochs": int(
            epochs
        ),
        "step": int(
            step
        ),
        "train_loss": (
            float(
                train_loss
            )
            if train_loss is not None
            else None
        ),
        "examples": int(
            examples
        ),
        "blocks": int(
            blocks
        ),
        "profile": profile,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(
        payload,
        TEMP_CHECKPOINT,
    )

    TEMP_CHECKPOINT.replace(
        CHECKPOINT
    )

    save_status(
        epoch=epoch,
        epochs=epochs,
        step=step,
        train_loss=train_loss,
        examples=examples,
        blocks=blocks,
        profile=profile,
    )

    print(
        (
            "[TEACHER CHECKPOINT] "
            f"epoch={epoch}/{session_last_epoch} | "
            f"step={step:,}"
        ),
        flush=True,
    )


def load_training_model(
    *,
    device,
):
    source = (
        CHECKPOINT
        if CHECKPOINT.exists()
        else BASE_CHECKPOINT
    )

    if not source.exists():
        raise FileNotFoundError(
            (
                "Amber Teacher attend le checkpoint Amber 0.2. "
                "Termine d'abord la continuation propre 0.2 avant le SFT."
            )
        )

    is_resume = (
        source == CHECKPOINT
    )

    print(
        (
            "[TEACHER] "
            + (
                "Reprise du checkpoint Teacher..."
                if is_resume
                else "Chargement des poids Amber 0.2..."
            )
        ),
        flush=True,
    )

    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(
            source,
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

    resume = {
        "is_resume": is_resume,
        "completed_epochs": int(
            payload.get(
                "completed_epochs",
                0
            )
        ),
        "step": int(
            payload.get(
                "step",
                0
            )
        ),
        "optimizer_state": (
            payload.get(
                "optimizer_state"
            )
            if is_resume
            else None
        ),
    }

    del payload

    return (
        model,
        config,
        resume,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=tuple(
            PROFILES.keys()
        ),
        default="full",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.0e-5,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    epochs = max(
        1,
        int(
            args.epochs
        ),
    )

    learning_rate = max(
        1e-7,
        float(
            args.learning_rate
        ),
    )

    sequence_length = min(
        max(
            128,
            int(
                args.sequence_length
            ),
        ),
        1024,
    )

    profile = dict(
        PROFILES[
            args.mode
        ]
    )

    clear_file(
        STOP_FILE
    )

    clear_file(
        PAUSE_FILE
    )

    tokenizer = AmberBPETokenizer()
    examples = load_jsonl(
        TRAIN_FILE
    )

    remedial_count = 0

    if REMEDIAL_FILE.exists():
        remedial = load_jsonl(
            REMEDIAL_FILE
        )

        remedial_count = len(
            remedial
        )

        examples.extend(
            remedial
        )

    print(
        (
            "[TEACHER] Cours chargés : "
            f"{len(examples):,} exemples "
            f"(dont {remedial_count:,} rattrapage)."
        ),
        flush=True,
    )

    x_all, y_all = build_blocks(
        tokenizer,
        examples,
        sequence_length=sequence_length,
    )

    print(
        (
            "[TEACHER] Dataset SFT packé : "
            f"{len(x_all):,} blocs × "
            f"{sequence_length} tokens."
        ),
        flush=True,
    )

    device = find_amber_gpu()

    torch.cuda.set_device(
        device
    )

    model, config, resume = load_training_model(
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(
            0.9,
            0.95,
        ),
        eps=1e-8,
        weight_decay=0.05,
    )

    if resume[
        "optimizer_state"
    ] is not None:
        optimizer.load_state_dict(
            resume[
                "optimizer_state"
            ]
        )

        for group in optimizer.param_groups:
            group[
                "lr"
            ] = learning_rate

        print(
            (
                "[TEACHER] Optimizer Teacher restauré | "
                f"step={resume['step']:,} | "
                f"epochs terminées={resume['completed_epochs']}."
            ),
            flush=True,
        )

    dtype = amp_dtype()

    micro_batch = int(
        profile[
            "micro_batch"
        ]
    )

    accumulation = int(
        profile[
            "gradient_accumulation"
        ]
    )

    governor = TrainingGovernor(
        args.mode
    )

    rng = np.random.default_rng(
        20260921
    )

    step = int(
        resume[
            "step"
        ]
    )

    completed_epochs = int(
        resume[
            "completed_epochs"
        ]
    )

    session_first_epoch = (
        completed_epochs
        + 1
    )

    session_last_epoch = (
        completed_epochs
        + epochs
    )

    last_loss = None
    started = time.time()

    print(
        "=" * 76,
        flush=True,
    )

    print(
        "AMBER TEACHER 0.1 - SUPERVISED FINE-TUNING",
        flush=True,
    )

    print(
        (
            f"GPU          : "
            f"{torch.cuda.get_device_name(device)}"
        ),
        flush=True,
    )

    print(
        f"Examples     : {len(examples):,}",
        flush=True,
    )

    print(
        f"Blocks       : {len(x_all):,}",
        flush=True,
    )

    print(
        f"Epochs       : {epochs}",
        flush=True,
    )

    print(
        f"LR           : {learning_rate:.2e}",
        flush=True,
    )

    print(
        f"Micro batch  : {micro_batch}",
        flush=True,
    )

    print(
        f"Grad accum   : {accumulation}",
        flush=True,
    )

    print(
        "=" * 76,
        flush=True,
    )

    for epoch in range(
        session_first_epoch,
        session_last_epoch + 1,
    ):
        order = rng.permutation(
            len(
                x_all
            )
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated = 0
        epoch_losses = []

        for batch_start in range(
            0,
            len(
                order
            ),
            micro_batch,
        ):
            if STOP_FILE.exists():
                clear_file(
                    STOP_FILE
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    epochs=session_last_epoch,
                    step=step,
                    train_loss=last_loss,
                    examples=len(
                        examples
                    ),
                    blocks=len(
                        x_all
                    ),
                    profile=args.mode,
                )

                print(
                    "[TEACHER] Arrêt propre terminé.",
                    flush=True,
                )

                return

            while PAUSE_FILE.exists():
                print(
                    "[TEACHER] En pause.",
                    flush=True,
                )

                time.sleep(
                    0.5
                )

                if STOP_FILE.exists():
                    clear_file(
                        PAUSE_FILE
                    )
                    break

            game = governor.active_game()

            if game:
                print(
                    (
                        "[TEACHER GOVERNOR] "
                        f"Jeu détecté : {game}. Pause."
                    ),
                    flush=True,
                )

                while governor.active_game():
                    time.sleep(
                        2
                    )

                print(
                    "[TEACHER GOVERNOR] Reprise.",
                    flush=True,
                )

            indices = order[
                batch_start:
                batch_start
                + micro_batch
            ]

            x = torch.from_numpy(
                x_all[
                    indices
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            y = torch.from_numpy(
                y_all[
                    indices
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
            ):
                _, loss = model(
                    x,
                    y,
                )

                scaled_loss = (
                    loss
                    / accumulation
                )

            scaled_loss.backward()

            accumulated += 1
            epoch_losses.append(
                float(
                    loss.detach().item()
                )
            )

            if (
                accumulated
                >= accumulation
                or batch_start
                + micro_batch
                >= len(
                    order
                )
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulated = 0
                step += 1

                last_loss = (
                    sum(
                        epoch_losses[
                            -max(
                                1,
                                accumulation
                            ):
                        ]
                    )
                    / min(
                        len(
                            epoch_losses
                        ),
                        max(
                            1,
                            accumulation
                        ),
                    )
                )

                if (
                    step <= 5
                    or step % 10 == 0
                ):
                    elapsed = (
                        time.time()
                        - started
                    )

                    done_blocks = (
                        (
                            epoch
                            - session_first_epoch
                        )
                        * len(
                            x_all
                        )
                        + min(
                            batch_start
                            + micro_batch,
                            len(
                                x_all
                            ),
                        )
                    )

                    total_blocks = (
                        (
                            session_last_epoch
                            - session_first_epoch
                            + 1
                        )
                        * len(
                            x_all
                        )
                    )

                    progress = (
                        done_blocks
                        / total_blocks
                    )

                    eta = (
                        elapsed
                        * (
                            1.0
                            - progress
                        )
                        / progress
                        if progress > 0
                        else 0.0
                    )

                    peak = (
                        torch.cuda.max_memory_allocated(
                            device
                        )
                        / 1024**3
                    )

                    reserved = (
                        torch.cuda.memory_reserved(
                            device
                        )
                        / 1024**3
                    )

                    print(
                        (
                            f"[TEACHER step={step:05d}] "
                            f"epoch={epoch}/{session_last_epoch} | "
                            f"loss={last_loss:.4f} | "
                            f"progress={progress * 100:.2f}% | "
                            f"peak={peak:.2f} GB | "
                            f"reserved={reserved:.2f} GB | "
                            f"eta={eta:.0f}s"
                        ),
                        flush=True,
                    )

            governor.throttle()

        epoch_loss = (
            sum(
                epoch_losses
            )
            / len(
                epoch_losses
            )
        )

        last_loss = epoch_loss

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            epochs=session_last_epoch,
            step=step,
            train_loss=epoch_loss,
            examples=len(
                examples
            ),
            blocks=len(
                x_all
            ),
            profile=args.mode,
            epoch_complete=True,
        )

        print(
            (
                f"[TEACHER EPOCH COMPLETE] "
                f"{epoch}/{session_last_epoch} | "
                f"loss={epoch_loss:.4f}"
            ),
            flush=True,
        )

    print(
        "[TEACHER TRAINING COMPLETE]",
        flush=True,
    )


if __name__ == "__main__":
    main()
