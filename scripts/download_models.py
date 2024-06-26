# pyright: basic

import struct
from argparse import ArgumentParser, Namespace, Action
from typing import BinaryIO, Tuple
from os import makedirs
import torch
from torch import Tensor
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer


def quantize_tensor(tensor: Tensor, num_bits: int) -> Tensor:
    x_range: float = float(torch.max(tensor) - torch.min(tensor))
    x_range = 1 if x_range == 0 else x_range
    num: int = 2 ** (num_bits - 1)
    scale: float = num / x_range
    zeropoint: Tensor = (-scale * torch.min(tensor) - num).round()
    x_quant: Tensor = torch.clip(
        (tensor * scale + zeropoint).round(),
        -num,
        num - 1,
    )
    return x_quant


def preprocess(
    tensor: torch.Tensor, activations: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    sparsity_threshold: float = 1e-3
    activation_threshold = 0.5
    # introduce sparsity
    tensor = torch.where(
        tensor.abs() < sparsity_threshold, torch.zeros_like(tensor), tensor
    )
    is_activated: torch.Tensor = (activations.abs() > activation_threshold).to(
        torch.bool
    )
    return tensor, is_activated


def serialize_fp32(
    file: BinaryIO, tensor: torch.Tensor, is_activated: torch.Tensor, num_bits: int
) -> None:
    tensor = tensor.detach().cpu().view(-1)
    # tensor, is_activated = preprocess(tensor, is_activated)

    quantized_tensor: Tensor = quantize_tensor(tensor, num_bits)

    # Determine the appropriate data type based on num_bits
    if num_bits <= 8:
        dtype = torch.int8
        fmt = "b"
    elif num_bits <= 16:
        dtype = torch.int16
        fmt = "h"
    else:
        dtype = torch.int32
        fmt = "i"

    # Convert quantized tensor to the appropriate dtype
    quantized_tensor = quantized_tensor.to(dtype)
    b: bytes = struct.pack(
        f"{len(quantized_tensor)}{fmt}", *quantized_tensor.numpy()
    )  # *isactivated.numpy())
    _ = file.write(b)


def model_export(
    model: MambaForCausalLM, config: MambaConfig, path: str, num_bits: int
) -> None:
    with open(path, "wb") as f:
        model_dict: dict[str, Tensor] = model.state_dict()
        d_inner: int = model_dict["backbone.layers.0.mixer.D"].shape[0]
        dt_rank: int = model_dict["backbone.layers.0.mixer.dt_proj.weight"].shape[1]
        d_state: int = model_dict["backbone.layers.0.mixer.A_log"].shape[1]
        d_conv: int = model_dict["backbone.layers.0.mixer.conv1d.weight"].shape[2]
        header: bytes = struct.pack(
            "iiiiiii",
            config.n_layer,
            config.vocab_size,
            config.hidden_size,
            d_inner,
            dt_rank,
            d_state,
            d_conv,
            # num_bits,
        )
        _ = f.write(header)

        # always guarantee 256 byte header size
        pad: int = 256 - f.tell()
        _ = f.write(b"\0" * pad)

        for n in range(config.n_layer):
            model_dict[f"backbone.layers.{n}.mixer.A"] = -torch.exp(
                model_dict.pop(f"backbone.layers.{n}.mixer.A_log")
            )

        # input from neural network, add to activations dictionary for each layer
        activations = {
            "backbone.embeddings.weight": torch.empty(0),
            "backbone.layers.%d.mixer.in_proj.weight": torch.empty(0),
            "backbone.layers.%d.mixer.conv1d.weight": torch.empty(0),
            "backbone.layers.%d.mixer.conv1d.bias": torch.empty(0),
            "backbone.layers.%d.mixer.x_proj.weight": torch.empty(0),
            "backbone.layers.%d.mixer.dt_proj.weight": torch.empty(0),
            "backbone.layers.%d.mixer.dt_proj.bias": torch.empty(0),
            "backbone.layers.%d.mixer.A": torch.empty(0),
            "backbone.layers.%d.mixer.D": torch.empty(0),
            "backbone.layers.%d.mixer.out_proj.weight": torch.empty(0),
            "backbone.layers.%d.norm.weight": torch.empty(0),
            "backbone.norm_f.weight": torch.empty(0),
        }
        serialize_fp32(
            f,
            model_dict["backbone.embeddings.weight"],
            activations["backbone.embeddings.weight"],
            num_bits,
        )

        layer_weights: list[str] = [
            "backbone.layers.%d.mixer.in_proj.weight",
            "backbone.layers.%d.mixer.conv1d.weight",
            "backbone.layers.%d.mixer.conv1d.bias",
            "backbone.layers.%d.mixer.x_proj.weight",
            "backbone.layers.%d.mixer.dt_proj.weight",
            "backbone.layers.%d.mixer.dt_proj.bias",
            "backbone.layers.%d.mixer.A",
            "backbone.layers.%d.mixer.D",
            "backbone.layers.%d.mixer.out_proj.weight",
            "backbone.layers.%d.norm.weight",
        ]

        for layer in layer_weights:
            print(f"writing {layer}")
            for n in range(config.n_layer):
                serialize_fp32(f, model_dict[layer % n], activations[layer], num_bits)

        serialize_fp32(
            f,
            model_dict["backbone.norm_f.weight"],
            activations["backbone.norm_f.weight"],
            num_bits,
        )
    print("model written to", path)


def tokenizer_export(model: str, path: str) -> None:
    print("exporting tokenizer...")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        "state-spaces/mamba-" + model + "-hf"
    )
    tokens: list[bytes] = []
    for i in range(50277):
        t: str = tokenizer.decode([i])
        b: bytes = t.encode("utf-8")
        tokens.append(b)

    max_token_length: int = max(len(t) for t in tokens)
    with open(path, "wb") as f:
        header: bytes = struct.pack("ii", len(tokens), max_token_length)
        _ = f.write(header)
        for token_bytes in tokens:
            _ = f.write(struct.pack("i", len(token_bytes)))
            _ = f.write(token_bytes)


def main() -> None:
    makedirs("models", exist_ok=True)
    parser: ArgumentParser = ArgumentParser()
    _: Action = parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["130m", "370m", "790m", "1.4b", "2.8b"],
        help="model name (allowed options: 130m, 370m, 790m, 1.4b, 2.8b)",
        default="130m",
    )
    _: Action = parser.add_argument(
        "-md",
        "--model_dir",
        type=str,
        help="output bin file",
        default="models/model.bin",
    )
    _: Action = parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        choices=["130m", "370m", "790m", "1.4b", "2.8b"],
        help="tokenizer name (allowed options: 130m, 370m, 790m, 1.4b, 2.8b)",
        default="130m",
    )
    _: Action = parser.add_argument(
        "-td",
        "--tokenizer_dir",
        type=str,
        help="output tokenizer bin file",
        default="models/tokenizer.bin",
    )
    _: Action = parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of bits for quantization, use 8 for now until I figure out a better way to generalize it",
    )
    args: Namespace = parser.parse_args()

    model: MambaForCausalLM = MambaForCausalLM.from_pretrained(
        "state-spaces/mamba-" + args.model + "-hf"
    )

    config: MambaConfig = MambaConfig.from_pretrained(
        "state-spaces/mamba-" + args.model + "-hf"
    )
    model_export(model, config, args.model_dir, args.bits)
    tokenizer_export(args.tokenizer, args.tokenizer_dir)


if __name__ == "__main__":
    main()
