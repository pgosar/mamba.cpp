import os
import struct
import argparse
import json
import numpy as np
import torch


def quantize_tensor(tensor, num_bits):
    x_range = torch.max(tensor) - torch.min(tensor)
    x_range = 1 if x_range == 0 else x_range
    num = 2 ** (num_bits - 1)
    scale = num / x_range
    zeropoint = (-scale * torch.min(tensor) - num).round()
    x_quant = torch.clip(
        (tensor * scale + zeropoint).round(),
        -num,
        num - 1,
    )
    return x_quant.to(tensor.dtype)


def serialize_fp32(file, tensor, num_bits):
    if num_bits < 32:
        tensor = quantize_tensor(tensor, num_bits)
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def write_weights(file, model, key, num_bits):
    print(f"writing {key} {list(model[key].shape)[::-1]}")
    serialize_fp32(file, model[key], num_bits)


def write_layer_weights(file, model, layer, n_layers, num_bits):
    print(f"writing {layer % n_layers} {list(model[layer % 0].shape)[::-1]}")
    for n in range(n_layers):
        serialize_fp32(file, model[layer % n], num_bits)


def model_export(model, config, filepath, num_bits):
    version = 1
    with open(filepath, "wb") as out_file:
        out_file.write(struct.pack("I", 0x4D616D62))  # Magic "Mamb"
        out_file.write(struct.pack("i", version))

        d_inner = model["layers.0.mixer.D"].shape[0]
        dt_rank = model["layers.0.mixer.dt_proj.weight"].shape[1]
        d_state = model["layers.0.mixer.A_log"].shape[1]
        d_conv = model["layers.0.mixer.conv1d.weight"].shape[2]
        shared_classifier = torch.equal(
            model["embedding.weight"], model["lm_head.weight"]
        )

        print(
            f"writing header\n  layers: {config.n_layers}\n  vocab_size: {config.vocab_size}\n  d_model: {config.d_model}\n  d_inner: {d_inner}\n  dt_rank: {dt_rank}\n  d_state: {d_state}\n  d_conv: {d_conv}\n  shared classifier: {shared_classifier}\n  quantization bits: {num_bits}"
        )

        header = struct.pack(
            "iiiiiiiii",
            config.n_layers,
            config.vocab_size,
            config.d_model,
            d_inner,
            dt_rank,
            d_state,
            d_conv,
            int(shared_classifier),
            num_bits,
        )
        out_file.write(header)

        pad = 256 - out_file.tell()
        assert pad >= 0
        out_file.write(b"\0" * pad)

        for n in range(config.n_layers):
            model[f"layers.{n}.mixer.A"] = -torch.exp(
                model.pop(f"layers.{n}.mixer.A_log")
            )

        write_weights(out_file, model, "embedding.weight", num_bits)

        layer_weights = [
            "layers.%d.mixer.in_proj.weight",
            "layers.%d.mixer.conv1d.weight",
            "layers.%d.mixer.conv1d.bias",
            "layers.%d.mixer.x_proj.weight",
            "layers.%d.mixer.dt_proj.weight",
            "layers.%d.mixer.dt_proj.bias",
            "layers.%d.mixer.A",
            "layers.%d.mixer.D",
            "layers.%d.mixer.out_proj.weight",
            "layers.%d.norm.weight",
        ]

        for layer in layer_weights:
            write_layer_weights(out_file, model, layer, config.n_layers, num_bits)

        write_weights(out_file, model, "norm_f.weight", num_bits)

        if not shared_classifier:
            write_weights(out_file, model, "lm_head.weight", num_bits)

    print(f"done. saved to {filepath}")


def load_model(path):
    print(f"loading model from {path}")

    if os.path.isdir(path):
        filepath = os.path.join(path, "pytorch_model.bin")
    else:
        filepath = path
    model = torch.load(filepath, map_location="cpu")

    unwanted_prefix = "backbone."
    for k, v in list(model.items()):
        if k.startswith(unwanted_prefix):
            model[k[len(unwanted_prefix) :]] = model.pop(k)

    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
    else:
        config_path = os.path.join(os.path.dirname(path), "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["n_layers"] = config.pop("n_layer")
    config = argparse.Namespace(**config)

    return model, config


def get_model_from_huggingface(model_name: str):
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    config_path = cached_file(
        model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
    )
    model_path = cached_file(
        model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        help="model name or folder where the model files are located",
        default="state-spaces/mamba-130m",
    )
    parser.add_argument(
        "destination",
        type=str,
        help="full path to the output file",
        default="model.bin",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of bits for quantization",
    )
    args = parser.parse_args()

    if args.source.startswith("state-spaces/mamba-"):
        model_path = get_model_from_huggingface(args.source)
    else:
        model_path = args.source

    model, config = load_model(model_path)

    if model is None:
        parser.error("Can't load input model!")

    model_export(model, config, args.destination, args.bits)
