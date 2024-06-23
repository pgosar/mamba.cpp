import struct
import argparse
import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer


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


def model_export(model, config, path, num_bits):
    with open(path, "wb") as f:
        model = model.state_dict()
        d_inner = model["backbone.layers.0.mixer.D"].shape[0]
        dt_rank = model["backbone.layers.0.mixer.dt_proj.weight"].shape[1]
        d_state = model["backbone.layers.0.mixer.A_log"].shape[1]
        d_conv = model["backbone.layers.0.mixer.conv1d.weight"].shape[2]
        header = struct.pack(
            "iiiiiiii",
            config.n_layer,
            config.vocab_size,
            config.d_model,
            d_inner,
            dt_rank,
            d_state,
            d_conv,
            num_bits,
        )
        f.write(header)

        # always guarantee 256 byte header size
        pad = 256 - f.tell()
        f.write(b"\0" * pad)

        for n in range(config.n_layer):
            model[f"backbone.layers.{n}.mixer.A"] = -torch.exp(
                model.pop(f"backbone.layers.{n}.mixer.A_log")
            )

        serialize_fp32(f, model["backbone.embeddings.weight"], num_bits)

        layer_weights = [
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
                serialize_fp32(f, model[layer % n], num_bits)

        serialize_fp32(f, model["backbone.norm_f.weight"], num_bits)


def tokenizer_export(model):
    print("exporting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    # get all the tokens. For some reason tokenizer.vocab_size is returning 50254 (wrong)
    # vs 50277 (expected) :(
    # https://huggingface.co/state-spaces/mamba-2.8b-hf/blob/main/tokenizer.json
    tokens = []
    for i in range(50277):
        t = tokenizer.decode([i])
        b = t.encode("utf-8")
        tokens.append(b)

    max_token_length = max(len(t) for t in tokens)
    tokenizer_bin = "../models/tokenizer.bin"
    with open(tokenizer_bin, "wb") as f:
        header = struct.pack("ii", len(tokens), max_token_length)
        f.write(header)
        for bytes in tokens:
            f.write(struct.pack("i", len(bytes)))
            f.write(bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        choices=["130m", "370m", "790m", "1.4b", "2.8b"],
        help="model name (allowed options: 130m, 370m, 790m, 1.4b, 2.8b)",
        default="130m",
    )
    parser.add_argument(
        "destination",
        type=str,
        help="output bin file",
        default="models/model.bin",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of bits for quantization",
    )
    args = parser.parse_args()

    model = MambaForCausalLM.from_pretrained(
        "state-spaces/mamba-" + args.source + "-hf"
    )
    config = MambaConfig.from_pretrained("state-spaces/mamba-" + args.source + "-hf")
    model_export(model, config, args.destination, args.bits)
    tokenizer_export(args.source)


if __name__ == "__main__":
    main()
