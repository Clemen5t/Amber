from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch

from amber.model_v01 import (
    AmberV01Config,
    AmberV01Model,
)
from teacher.curriculum import (
    EXAM_FILE,
    REMEDIAL_FILE,
)
from tokenizer.amber_bpe import AmberBPETokenizer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "amber_teacher_latest.pt"


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


def normalize(
    text: str
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        text.lower(),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(
            char
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def load_jsonl(
    path: Path
):
    if not path.exists():
        raise FileNotFoundError(
            (
                "Examen Amber Teacher introuvable. "
                "Construis d'abord les cours."
            )
        )

    items = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                items.append(
                    json.loads(
                        line
                    )
                )

    return items


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            (
                "Checkpoint Amber Teacher introuvable. "
                "Entraîne d'abord Amber avec les cours."
            )
        )

    device = find_amber_gpu()

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

    del payload

    return (
        model,
        device,
    )


@torch.no_grad()
def answer(
    *,
    model,
    tokenizer,
    device,
    question,
    max_new_tokens=64,
):
    prompt = (
        "Utilisateur : "
        + question.strip()
        + "\nAssistant : "
    )

    ids = tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )

    input_ids = torch.tensor(
        [
            ids
        ],
        dtype=torch.long,
        device=device,
    )

    generated = []

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for _ in range(
        max_new_tokens
    ):
        context = input_ids[
            :,
            -model.config.context_length:
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
        ):
            logits, _ = model(
                context
            )

        next_token = torch.argmax(
            logits[
                :,
                -1,
                :
            ],
            dim=-1,
            keepdim=True,
        )

        token_id = int(
            next_token.item()
        )

        if (
            tokenizer.eos_id is not None
            and token_id
            == int(
                tokenizer.eos_id
            )
        ):
            break

        generated.append(
            token_id
        )

        input_ids = torch.cat(
            (
                input_ids,
                next_token,
            ),
            dim=1,
        )

    return tokenizer.decode(
        generated
    ).strip()


def score_item(
    item,
    response,
):
    checks = item.get(
        "checks",
        {}
    )

    normalized = normalize(
        response
    )

    all_terms = [
        normalize(
            str(
                term
            )
        )
        for term in checks.get(
            "all",
            []
        )
    ]

    any_terms = [
        normalize(
            str(
                term
            )
        )
        for term in checks.get(
            "any",
            []
        )
    ]

    if all_terms and not all(
        term in normalized
        for term in all_terms
    ):
        return False

    if any_terms and not any(
        term in normalized
        for term in any_terms
    ):
        return False

    if not all_terms and not any_terms:
        expected = normalize(
            item.get(
                "assistant",
                ""
            )
        )

        return (
            expected in normalized
            or normalized in expected
        )

    return True


def write_remedial(
    failures
):
    REMEDIAL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with REMEDIAL_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for item, response in failures:
            expected = item[
                "assistant"
            ]

            prompts = [
                item[
                    "user"
                ],
                (
                    "Réponds correctement et clairement : "
                    + item[
                        "user"
                    ]
                ),
                (
                    "Exercice de rattrapage. "
                    + item[
                        "user"
                    ]
                ),
            ]

            for prompt in prompts:
                lesson = {
                    "user": prompt,
                    "assistant": expected,
                    "category": (
                        "rattrapage/"
                        + item.get(
                            "category",
                            "général",
                        )
                    ),
                    "difficulty": int(
                        item.get(
                            "difficulty",
                            1,
                        )
                    ),
                    "source": (
                        "Amber Teacher · correction automatique d'examen"
                    ),
                    "quality": "validated",
                    "failed_response": response,
                }

                handle.write(
                    json.dumps(
                        lesson,
                        ensure_ascii=False,
                    )
                )

                handle.write(
                    "\n"
                )

                count += 1

    return count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    exam = load_jsonl(
        EXAM_FILE
    )

    tokenizer = AmberBPETokenizer()

    model, device = load_model()

    passed = 0
    failures = []
    results = []

    for index, item in enumerate(
        exam,
        start=1,
    ):
        response = answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=item[
                "user"
            ],
            max_new_tokens=max(
                8,
                args.max_new_tokens
            ),
        )

        ok = score_item(
            item,
            response,
        )

        if ok:
            passed += 1
        else:
            failures.append(
                (
                    item,
                    response,
                )
            )

        results.append(
            {
                "index": index,
                "category": item.get(
                    "category"
                ),
                "question": item[
                    "user"
                ],
                "expected": item[
                    "assistant"
                ],
                "response": response,
                "passed": ok,
            }
        )

        print(
            (
                f"[TEACHER EXAM] "
                f"{index}/{len(exam)} | "
                f"{'OK' if ok else 'FAIL'} | "
                f"{item.get('category', '-')}"
            ),
            flush=True,
        )

    score = (
        passed
        / len(
            exam
        )
        * 100.0
        if exam
        else 0.0
    )

    remedial_count = write_remedial(
        failures
    )

    summary = {
        "total": len(
            exam
        ),
        "passed": passed,
        "failed": len(
            failures
        ),
        "score_percent": score,
        "remedial_examples": remedial_count,
        "results": results,
    }

    print(
        (
            "[TEACHER EXAM JSON] "
            + json.dumps(
                summary,
                ensure_ascii=False,
            )
        ),
        flush=True,
    )

    print(
        (
            "[TEACHER EXAM COMPLETE] "
            f"score={score:.1f}% | "
            f"passed={passed}/{len(exam)} | "
            f"remedial={remedial_count}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
