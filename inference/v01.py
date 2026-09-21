from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
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


def load_v01():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Checkpoint Amber 0.1 introuvable."
        )

    device = find_amber_gpu()

    print(
        "[V01 INFERENCE] Chargement du checkpoint...",
        flush=True
    )

    started = time.perf_counter()

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

    tokenizer = AmberBPETokenizer()

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
        "train_loss": payload.get(
            "train_loss"
        ),
        "val_loss": payload.get(
            "val_loss"
        ),
        "amber_version": payload.get(
            "amber_version",
            "0.1"
        ),
    }

    del payload

    print(
        (
            "[V01 INFERENCE] Prêt en "
            f"{time.perf_counter() - started:.2f}s | "
            f"{metadata['tokens_seen']:,} tokens entraînés."
        ),
        flush=True
    )

    return (
        model,
        tokenizer,
        device,
        metadata,
    )


def prepare_prompt(
    prompt,
    mode,
):
    prompt = prompt.strip()

    if mode == "qa":
        return (
            "Question : "
            + prompt
            + "\nRéponse :"
        )

    return prompt


@torch.no_grad()
def generate(
    *,
    model,
    tokenizer,
    device,
    prompt,
    mode="continuation",
    max_new_tokens=120,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.05,
):
    rendered_prompt = prepare_prompt(
        prompt,
        mode
    )

    prompt_ids = tokenizer.encode(
        rendered_prompt,
        add_bos=True,
        add_eos=False,
    )

    if not prompt_ids:
        raise ValueError(
            "Prompt vide."
        )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    eos_id = tokenizer.eos_id

    generated = []

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    started = time.perf_counter()

    for _ in range(
        int(max_new_tokens)
    ):
        context = input_ids[
            :,
            -model.config.context_length:
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype
        ):
            logits, _ = model(
                context
            )

        logits = logits[
            :,
            -1,
            :
        ].float()

        if (
            repetition_penalty
            and repetition_penalty != 1.0
        ):
            unique_tokens = set(
                int(token)
                for token in input_ids[
                    0,
                    -256:
                ].tolist()
            )

            for token_id in unique_tokens:
                value = logits[
                    0,
                    token_id
                ]

                logits[
                    0,
                    token_id
                ] = (
                    value
                    / repetition_penalty
                    if value > 0
                    else value
                    * repetition_penalty
                )

        if temperature <= 0:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        else:
            logits = (
                logits
                / float(
                    temperature
                )
            )

            if top_k and top_k > 0:
                k = min(
                    int(top_k),
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                cutoff = values[
                    :,
                    [-1]
                ]

                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            probs = F.softmax(
                logits,
                dim=-1
            )

            if (
                top_p
                and 0.0 < top_p < 1.0
            ):
                sorted_probs, sorted_indices = (
                    torch.sort(
                        probs,
                        descending=True,
                        dim=-1
                    )
                )

                cumulative = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )

                remove = cumulative > float(
                    top_p
                )

                remove[
                    :,
                    1:
                ] = remove[
                    :,
                    :-1
                ].clone()

                remove[
                    :,
                    0
                ] = False

                sorted_probs = (
                    sorted_probs.masked_fill(
                        remove,
                        0.0
                    )
                )

                sorted_probs = (
                    sorted_probs
                    / sorted_probs.sum(
                        dim=-1,
                        keepdim=True
                    )
                )

                sampled = torch.multinomial(
                    sorted_probs,
                    num_samples=1
                )

                next_token = sorted_indices.gather(
                    -1,
                    sampled
                )

            else:
                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

        token_id = int(
            next_token.item()
        )

        if (
            eos_id is not None
            and token_id == int(
                eos_id
            )
        ):
            break

        generated.append(
            token_id
        )

        input_ids = torch.cat(
            (
                input_ids,
                next_token
            ),
            dim=1
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    text = tokenizer.decode(
        generated
    ).strip()

    return {
        "text": text,
        "rendered_prompt": rendered_prompt,
        "prompt_tokens": len(
            prompt_ids
        ),
        "generated_tokens": len(
            generated
        ),
        "tokens_per_second": (
            len(generated)
            / elapsed
            if elapsed > 0
            else 0.0
        ),
        "seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=(
            "continuation",
            "qa"
        ),
        default="continuation"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05
    )

    args = parser.parse_args()

    model, tokenizer, device, metadata = (
        load_v01()
    )

    result = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        mode=args.mode,
        max_new_tokens=max(
            1,
            args.max_new_tokens
        ),
        temperature=max(
            0.0,
            args.temperature
        ),
        top_k=max(
            0,
            args.top_k
        ),
        top_p=min(
            max(
                args.top_p,
                0.0
            ),
            1.0
        ),
        repetition_penalty=max(
            1.0,
            args.repetition_penalty
        ),
    )

    result.update(
        metadata
    )

    print(
        "[V01 GEN JSON] "
        + json.dumps(
            result,
            ensure_ascii=False
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
