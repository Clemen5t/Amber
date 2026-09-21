import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from amber.model import AmberConfig, AmberModel
from tokenizer import AmberByteTokenizer
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

DATA_FILE = (
    ROOT
    / "data"
    / "train.txt"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "amber_seed_latest.pt"
)

STOP_FILE = (
    ROOT
    / "training"
    / ".stop_requested"
)

PAUSE_FILE = (
    ROOT
    / "training"
    / ".pause_requested"
)


MODES = {
    "eco": {
        "batch_size": 4,
        "seq_len": 128,
        "save_every": 50,
    },

    "balanced": {
        "batch_size": 8,
        "seq_len": 128,
        "save_every": 50,
    },

    "full": {
        "batch_size": 16,
        "seq_len": 192,
        "save_every": 50,
    },
}


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


def build_batch(
    tokens,
    batch_size,
    sequence_length,
    device
):
    max_start = (
        tokens.numel()
        - sequence_length
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    offsets = torch.arange(
        sequence_length
    )

    indices = (
        starts[:, None]
        + offsets[None, :]
    )

    x = tokens[indices]
    y = tokens[indices + 1]

    return (
        x.to(device),
        y.to(device)
    )


def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    config,
    mode
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "amber_version": "0.0.3",
        "step": int(step),
        "loss": float(loss),
        "mode": mode,
        "config": asdict(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    temp = (
        CHECKPOINT_DIR
        / "amber_seed_latest.tmp"
    )

    torch.save(
        checkpoint,
        temp
    )

    temp.replace(
        LATEST_CHECKPOINT
    )

    print(
        f"[Checkpoint] step {step} saved.",
        flush=True
    )


def load_checkpoint(
    model,
    optimizer,
    device
):
    if not LATEST_CHECKPOINT.exists():
        return 0, None

    print(
        "[Checkpoint] Loading...",
        flush=True
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    step = int(
        checkpoint.get(
            "step",
            0
        )
    )

    loss = checkpoint.get(
        "loss",
        None
    )

    print(
        f"[Checkpoint] resume step {step} | "
        f"previous loss = {loss}",
        flush=True
    )

    return step, loss


def clear_file(path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "eco",
            "balanced",
            "full"
        ],
        default="balanced"
    )

    parser.add_argument(
        "--target-step",
        type=int,
        default=500
    )

    args = parser.parse_args()

    settings = MODES[
        args.mode
    ]

    print(
        "=" * 64,
        flush=True
    )

    print(
        "AMBER TRAINER 0.0.3",
        flush=True
    )

    print(
        "=" * 64,
        flush=True
    )

    device = find_amber_gpu()

    print(
        "GPU :",
        torch.cuda.get_device_name(
            device
        ),
        flush=True
    )

    print(
        "Mode :",
        args.mode.upper(),
        flush=True
    )

    tokenizer = AmberByteTokenizer()

    text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    encoded = tokenizer.encode(
        text
    )

    encoded = encoded * 32

    tokens = torch.tensor(
        encoded,
        dtype=torch.long
    )

    print(
        f"Corpus : {len(text):,} characters",
        flush=True
    )

    print(
        f"Working tokens : {tokens.numel():,}",
        flush=True
    )

    config = AmberConfig()

    model = AmberModel(
        config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(
            0.9,
            0.95
        ),
        weight_decay=0.1
    )

    start_step, previous_loss = load_checkpoint(
        model,
        optimizer,
        device
    )

    target_step = int(
        args.target_step
    )

    if target_step <= start_step:
        print(
            f"Target {target_step} already reached. "
            f"Current step = {start_step}.",
            flush=True
        )

        return

    clear_file(
        STOP_FILE
    )

    clear_file(
        PAUSE_FILE
    )

    governor = TrainingGovernor(
        args.mode
    )

    print(
        f"Parameters : "
        f"{model.parameter_count():,}",
        flush=True
    )

    print(
        f"Current step : {start_step}",
        flush=True
    )

    print(
        f"Target step  : {target_step}",
        flush=True
    )

    print(
        f"Training from {start_step + 1} "
        f"to {target_step}",
        flush=True
    )

    print(
        "",
        flush=True
    )

    first_session_loss = None
    last_loss = previous_loss

    started_at = time.time()

    for step in range(
        start_step + 1,
        target_step + 1
    ):

        # ----------------------------------------------------
        # STOP AVANT ETAPE
        # ----------------------------------------------------

        if STOP_FILE.exists():
            clear_file(
                STOP_FILE
            )

            save_checkpoint(
                model,
                optimizer,
                step - 1,
                last_loss or 0.0,
                config,
                args.mode
            )

            print(
                "[Amber] Clean stop completed.",
                flush=True
            )

            return

        # ----------------------------------------------------
        # PAUSE UTILISATEUR
        # ----------------------------------------------------

        if PAUSE_FILE.exists():

            save_checkpoint(
                model,
                optimizer,
                step - 1,
                last_loss or 0.0,
                config,
                args.mode
            )

            print(
                "[Amber] Training paused.",
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
                        model,
                        optimizer,
                        step - 1,
                        last_loss or 0.0,
                        config,
                        args.mode
                    )

                    print(
                        "[Amber] Clean stop completed.",
                        flush=True
                    )

                    return

                time.sleep(
                    0.5
                )

            print(
                "[Amber] Training resumed.",
                flush=True
            )

        # ----------------------------------------------------
        # JEU DETECTE
        # ----------------------------------------------------

        game = governor.active_game()

        if game:

            save_checkpoint(
                model,
                optimizer,
                step - 1,
                last_loss or 0.0,
                config,
                args.mode
            )

            print(
                f"[Governor] Game detected: {game}",
                flush=True
            )

            print(
                "[Governor] Training paused.",
                flush=True
            )

            while governor.active_game():

                if STOP_FILE.exists():
                    clear_file(
                        STOP_FILE
                    )

                    save_checkpoint(
                        model,
                        optimizer,
                        step - 1,
                        last_loss or 0.0,
                        config,
                        args.mode
                    )

                    print(
                        "[Amber] Clean stop completed.",
                        flush=True
                    )

                    return

                time.sleep(
                    2
                )

            print(
                "[Governor] Game closed. Training resumed.",
                flush=True
            )

        # ----------------------------------------------------
        # TRAIN STEP
        # ----------------------------------------------------

        model.train()

        x, y = build_batch(
            tokens,
            settings["batch_size"],
            settings["seq_len"],
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            x,
            y
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        current_loss = float(
            loss.detach().item()
        )

        last_loss = current_loss

        if first_session_loss is None:
            first_session_loss = current_loss

        if (
            step == start_step + 1
            or step % 10 == 0
            or step == target_step
        ):
            elapsed = (
                time.time()
                - started_at
            )

            done = (
                step
                - start_step
            )

            speed = (
                done / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"[{step:06d}] "
                f"loss={current_loss:.4f} | "
                f"speed={speed:.2f} step/s",
                flush=True
            )

        if (
            step
            % settings["save_every"]
            == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                step,
                current_loss,
                config,
                args.mode
            )

        # ----------------------------------------------------
        # STOP APRES ETAPE
        # ----------------------------------------------------

        if STOP_FILE.exists():

            clear_file(
                STOP_FILE
            )

            save_checkpoint(
                model,
                optimizer,
                step,
                current_loss,
                config,
                args.mode
            )

            print(
                "[Amber] Clean stop completed.",
                flush=True
            )

            return

        governor.throttle()

    save_checkpoint(
        model,
        optimizer,
        target_step,
        last_loss,
        config,
        args.mode
    )

    elapsed = (
        time.time()
        - started_at
    )

    print(
        "",
        flush=True
    )

    print(
        "=" * 64,
        flush=True
    )

    print(
        "AMBER TRAINING COMPLETE",
        flush=True
    )

    print(
        f"Final step : {target_step}",
        flush=True
    )

    print(
        f"Final loss : {last_loss:.4f}",
        flush=True
    )

    print(
        f"Duration   : {elapsed:.1f} s",
        flush=True
    )

    print(
        "=" * 64,
        flush=True
    )


if __name__ == "__main__":
    main()
