import torch

from amber.model import AmberConfig, AmberModel


print("=" * 60)
print("AMBER 0.0.1 - FIRST BRAIN TEST")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "Aucun GPU ROCm disponible."
    )

# La RX 7900 XT était cuda:1 lors de nos tests.
device = torch.device("cuda:1")

print(
    "GPU :",
    torch.cuda.get_device_name(device)
)

config = AmberConfig()

model = AmberModel(config).to(device)

parameters = model.parameter_count()

print(
    f"Paramètres Amber : {parameters:,}"
)

# Faux tokens uniquement pour tester le réseau.
batch_size = 4
sequence_length = 128

input_ids = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length),
    device=device
)

targets = torch.randint(
    0,
    config.vocab_size,
    (batch_size, sequence_length),
    device=device
)

print("Forward pass...")

logits, loss = model(
    input_ids,
    targets
)

print(
    "Logits :",
    tuple(logits.shape)
)

print(
    "Loss initiale :",
    loss.item()
)

print("Backward pass...")

loss.backward()

torch.cuda.synchronize(device)

allocated = (
    torch.cuda.memory_allocated(device)
    / 1024**3
)

print()
print("AMBER FORWARD     : OK")
print("AMBER LOSS        : OK")
print("AMBER BACKPROP    : OK")
print(
    f"VRAM utilisée     : "
    f"{allocated:.2f} Go"
)

print()
print(
    "Amber vient d'effectuer "
    "sa première rétropropagation."
)

print("=" * 60)
