from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

import torch

# Define model dimensions
model_dimensions = {
    "130M": {"num_layers": 24, "dim": 768},
    "370M": {"num_layers": 48, "dim": 1024},
    "790M": {"num_layers": 48, "dim": 1536},
    "1.4B": {"num_layers": 48, "dim": 2048},
    "2.8B": {"num_layers": 64, "dim": 2560}
}

# Activation tracking
activations = {}

def get_activation(name):
    def hook(model, input, output):
        try:
            if output.shape[-1] == model_dimensions[MODEL_SIZE]["dim"]:
                activations[name] = output.detach()
                print(f"Layer {name} has output of shape {output.shape}")
        except AttributeError as _:
            pass
    return hook

MODEL_SIZE = "130M"

tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{MODEL_SIZE}-hf")
model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{MODEL_SIZE}-hf")

hooks = []
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Module) and len(list(layer.parameters())) > 0:
        hook_handle = layer.register_forward_hook(get_activation(name))
        hooks.append(hook_handle)

# Generate text
input_ids = tokenizer("Tell me a joke", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.batch_decode(out))

# Clean up the hooks
for hook in hooks:
    hook.remove()
