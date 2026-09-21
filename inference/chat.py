from pathlib import Path

import torch
import torch.nn.functional as F

from amber.model import AmberConfig, AmberModel
from tokenizer import AmberByteTokenizer


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "amber_seed_latest.pt"
)


def find_amber_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Aucun GPU ROCm disponible."
        )

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)

        if "7900 XT" in name.upper():
            return torch.device(f"cuda:{i}")

    raise RuntimeError(
        "AMD Radeon RX 7900 XT introuvable."
    )


def load_amber():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Aucun checkpoint Amber disponible."
        )

    device = find_amber_gpu()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    config_data = checkpoint.get(
        "config",
        {}
    )

    config = AmberConfig(
        **config_data
    )

    model = AmberModel(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    tokenizer = AmberByteTokenizer()

    return (
        model,
        tokenizer,
        device,
        checkpoint
    )


@torch.no_grad()
def generate_reply(
    user_text,
    max_new_tokens=180,
    temperature=0.0,
    top_k=None
):
    model, tokenizer, device, checkpoint = load_amber()

    prompt = (
        f"Question : {user_text.strip()}\n"
        f"Réponse :"
    )

    prompt_tokens = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device
    )

    generated = []

    for _ in range(max_new_tokens):

        context = input_ids[
            :,
            -model.config.context_length:
        ]

        logits, _ = model(
            context
        )

        logits = logits[:, -1, :]

        # Amber Seed n'a été entraînée que
        # sur les octets UTF-8 0..255.
        logits[:, 256:] = float("-inf")

        if temperature <= 0:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        else:
            logits = logits / temperature

            if top_k is not None:
                k = min(
                    top_k,
                    256
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                cutoff = values[:, [-1]]

                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

        token_id = int(
            next_token.item()
        )

        generated.append(
            token_id
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )

        text = tokenizer.decode(
            generated
        )

        # On coupe quand Amber semble démarrer
        # une nouvelle question.
        if "\nQuestion :" in text:
            text = text.split(
                "\nQuestion :",
                1
            )[0]

            break

        if "\n\n" in text:
            text = text.split(
                "\n\n",
                1
            )[0]

            break

    answer = tokenizer.decode(
        generated
    )

    if "\nQuestion :" in answer:
        answer = answer.split(
            "\nQuestion :",
            1
        )[0]

    if "\n\n" in answer:
        answer = answer.split(
            "\n\n",
            1
        )[0]

    answer = answer.strip()

    step = checkpoint.get(
        "step",
        0
    )

    loss = checkpoint.get(
        "loss",
        None
    )

    del model

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "text": answer,
        "step": step,
        "loss": loss
    }
