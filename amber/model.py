from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AmberConfig:
    # Amber Seed 0.0.1
    vocab_size: int = 4096
    context_length: int = 256

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 768

    dropout: float = 0.0
    rms_norm_eps: float = 1e-5

    # Identité du modèle
    model_name: str = "Amber Seed"
    version: str = "0.0.1"


class AmberRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calcul de la normalisation en float32 pour la stabilité.
        input_dtype = x.dtype
        x_float = x.float()

        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)

        return (self.weight * x_norm).to(input_dtype)


class AmberCausalSelfAttention(nn.Module):
    def __init__(self, config: AmberConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError(
                "d_model doit être divisible par n_heads."
            )

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.dropout = config.dropout

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        # PyTorch SDPA applique directement le masque causal.
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            batch_size,
            sequence_length,
            channels
        )

        return self.out_proj(attention)


class AmberMLP(nn.Module):
    def __init__(self, config: AmberConfig):
        super().__init__()

        # SwiGLU
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
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class AmberBlock(nn.Module):
    def __init__(self, config: AmberConfig):
        super().__init__()

        self.attention_norm = AmberRMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.attention = AmberCausalSelfAttention(config)

        self.mlp_norm = AmberRMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.mlp = AmberMLP(config)

    def forward(self, x):
        x = x + self.attention(
            self.attention_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


class AmberModel(nn.Module):
    def __init__(self, config: AmberConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                AmberBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = AmberRMSNorm(
            config.d_model,
            config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Partage des poids embedding / sortie.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids, targets=None):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Séquence trop longue : {sequence_length}. "
                f"Maximum Amber : "
                f"{self.config.context_length}."
            )

        positions = torch.arange(
            0,
            sequence_length,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
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
        max_new_tokens=50,
        temperature=1.0,
        top_k=50
    ):
        self.eval()

        for _ in range(max_new_tokens):

            context = input_ids[
                :,
                -self.config.context_length:
            ]

            logits, _ = self(context)

            logits = logits[:, -1, :]

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
                        logits.size(-1)
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

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        return input_ids

    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


if __name__ == "__main__":
    config = AmberConfig()
    model = AmberModel(config)

    print("=" * 55)
    print("AMBER MODEL")
    print("=" * 55)

    print(
        f"Nom       : {config.model_name}"
    )

    print(
        f"Version   : {config.version}"
    )

    print(
        f"Paramètres: "
        f"{model.parameter_count():,}"
    )

    print(
        f"Layers    : {config.n_layers}"
    )

    print(
        f"Dimension : {config.d_model}"
    )

    print(
        f"Heads     : {config.n_heads}"
    )

    print(
        f"Contexte  : {config.context_length}"
    )

    print("=" * 55)
