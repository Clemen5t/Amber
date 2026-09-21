from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


@dataclass
class AmberV01Config:
    # Amber 0.1 — premier vrai modèle de pré-entraînement.
    vocab_size: int = 32000
    context_length: int = 1024

    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    model_name: str = "Amber 0.1"
    version: str = "0.1.0"

    def to_dict(self):
        return asdict(self)


class AmberV01RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        return (xf * self.weight).to(dtype)


class AmberRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position: int,
        theta: float,
    ):
        super().__init__()

        if head_dim % 2:
            raise ValueError("head_dim doit être pair pour RoPE.")

        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2).float()
                / head_dim
            )
        )

        positions = torch.arange(
            max_position
        ).float()

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos_cached",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            freqs.sin(),
            persistent=False
        )

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack(
            (-x2, x1),
            dim=-1
        ).flatten(-2)

    def forward(self, q, k):
        seq_len = q.size(-2)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1
        ).to(
            device=q.device,
            dtype=q.dtype
        )[None, None, :, :]

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1
        ).to(
            device=q.device,
            dtype=q.dtype
        )[None, None, :, :]

        q = (
            q * cos
            + self._rotate_half(q) * sin
        )

        k = (
            k * cos
            + self._rotate_half(k) * sin
        )

        return q, k


class AmberV01Attention(nn.Module):
    def __init__(self, config: AmberV01Config):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError(
                "d_model doit être divisible par n_heads."
            )

        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(
                "n_heads doit être divisible par n_kv_heads."
            )

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = (
            config.d_model
            // config.n_heads
        )

        self.kv_repeat = (
            config.n_heads
            // config.n_kv_heads
        )

        self.q_proj = nn.Linear(
            config.d_model,
            config.n_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.rope = AmberRotaryEmbedding(
            self.head_dim,
            config.context_length,
            config.rope_theta
        )

        self.dropout = config.dropout

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(
            q,
            k
        )

        if self.kv_repeat != 1:
            k = k.repeat_interleave(
                self.kv_repeat,
                dim=1
            )

            v = v.repeat_interleave(
                self.kv_repeat,
                dim=1
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        y = y.transpose(
            1,
            2
        ).contiguous().view(
            batch,
            seq_len,
            -1
        )

        return self.out_proj(y)


class AmberV01SwiGLU(nn.Module):
    def __init__(self, config: AmberV01Config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )

        self.up_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False
        )

        self.down_proj = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(
                self.gate_proj(x)
            )
            * self.up_proj(x)
        )


class AmberV01Block(nn.Module):
    def __init__(self, config: AmberV01Config):
        super().__init__()

        self.attention_norm = AmberV01RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.attention = AmberV01Attention(
            config
        )

        self.mlp_norm = AmberV01RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.mlp = AmberV01SwiGLU(
            config
        )

    def forward(self, x):
        x = x + self.attention(
            self.attention_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


class AmberV01Model(nn.Module):
    def __init__(self, config: AmberV01Config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                AmberV01Block(config)
                for _ in range(
                    config.n_layers
                )
            ]
        )

        self.final_norm = AmberV01RMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying : embedding et projection de sortie partagent les poids.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(
            self._init_weights
        )

        # Initialisation résiduelle plus stable pour un réseau plus profond.
        residual_std = (
            0.02
            / math.sqrt(
                2 * config.n_layers
            )
        )

        for block in self.blocks:
            nn.init.normal_(
                block.attention.out_proj.weight,
                mean=0.0,
                std=residual_std
            )

            nn.init.normal_(
                block.mlp.down_proj.weight,
                mean=0.0,
                std=residual_std
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(
            module,
            nn.Linear
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(
            module,
            nn.Embedding
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        targets=None
    ):
        _, sequence_length = (
            input_ids.shape
        )

        if (
            sequence_length
            > self.config.context_length
        ):
            raise ValueError(
                f"Séquence trop longue : "
                f"{sequence_length}. Maximum : "
                f"{self.config.context_length}."
            )

        x = self.token_embedding(
            input_ids
        )

        for block in self.blocks:
            if (
                self.training
                and self.config.gradient_checkpointing
            ):
                x = activation_checkpoint(
                    block,
                    x,
                    use_reentrant=False
                )

            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        eos_id=None,
    ):
        self.eval()

        for _ in range(
            max_new_tokens
        ):
            context = input_ids[
                :,
                -self.config.context_length:
            ]

            logits, _ = self(
                context
            )

            logits = logits[
                :,
                -1,
                :
            ]

            if temperature <= 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            else:
                logits = (
                    logits
                    / temperature
                )

                if top_k:
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

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            input_ids = torch.cat(
                (
                    input_ids,
                    next_token
                ),
                dim=1
            )

            if (
                eos_id is not None
                and int(
                    next_token.item()
                ) == int(eos_id)
            ):
                break

        return input_ids

    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


def default_v01_parameter_count(
    vocab_size=32000
):
    config = AmberV01Config(
        vocab_size=vocab_size
    )

    model = AmberV01Model(
        config
    )

    return model.parameter_count()
