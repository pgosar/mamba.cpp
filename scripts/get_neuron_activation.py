import argparse
import torch
from transformers import MambaForCausalLM, AutoTokenizer

# Define model dimensions
model_dimensions = {
    "130M": {"num_layers": 24, "dim": 768},
    "370M": {"num_layers": 48, "dim": 1024},
    "790M": {"num_layers": 48, "dim": 1536},
    "1.4B": {"num_layers": 48, "dim": 2048},
    "2.8B": {"num_layers": 64, "dim": 2560},
}

# Activation tracking
activations = {}


def get_activation(name, model_size):
    def hook(model, input, output):
        try:
            if output.shape[-1] == model_dimensions[model_size]["dim"]:
                activations[name] = output.detach()
                print(f"Layer {name} has output of shape {output.shape}")
        except AttributeError as _:
            pass

    return hook


def main(model_size):
    tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{model_size}-hf")
    model = MambaForCausalLM.from_pretrained(f"state-spaces/mamba-{model_size}-hf")

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Module) and len(list(layer.parameters())) > 0:
            hook_handle = layer.register_forward_hook(get_activation(name, model_size))
            hooks.append(hook_handle)

    # Generate text
    input_ids = tokenizer("Tell me a joke", return_tensors="pt")["input_ids"]
    out = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.batch_decode(out))
    print(activations)
    # Clean up the hooks
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified size.")
    parser.add_argument(
        "-m",
        "--model-size",
        type=str,
        choices=model_dimensions.keys(),
        help="Size of the model",
        required=True,
    )
    args = parser.parse_args()
    main(args.model_size)
