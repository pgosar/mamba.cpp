To run:

1. python3 scripts/tokenizer.py
2. python3 scripts/export.py state-spaces/mamba-130m models/model.bin
3. make fast
4. ./build/mamba models/model.bin -n 20 -i "Customer Support should" -t 0.0

Command line arguments will be used to control inference, for example, quantization level,
debugging verbosity, input prompt.

## TODO
Model configuration will be done through model_config.yaml, for example, temperature (text
diversity), generated text amount, batch size. There may be multiple selectable configurations,
these are selected through the command line arguments.

## TODO
- [ ] Initial C++ Implementation

- [ ] Quantization

- [ ] 1-bit weight experimentation
  
- [ ] Speculative Decoding
    - [ ] draft model fine tuning for jamba

- [ ] Flash mem
    - [x] neuron activation data
    - [ ] hot and cold neurons prediction
          
- [ ] Matrix mult optimization and overall optimization


## Helpful references:

### Models

[Jamba](https://huggingface.co/ai21labs/Jamba-v0.1)

[Mamba Variants](https://huggingface.co/state-spaces)

### Model Configuration

https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910

### Implementations:

Implementation of some optimization techniques

https://github.com/MDK8888/GPTFast/tree/master

Mamba LLM

https://github.com/redotvideo/mamba-chat


### Using ReLu instead of SiLu (mamba's default):

https://arxiv.org/abs/2310.04564

### Flash memory:

https://arxiv.org/abs/2312.11514

### Speculative Streaming:
https://arxiv.org/abs/2402.11131

### Speculative Decoding:

https://arxiv.org/abs/2211.17192

### 1 bit model variant:

https://arxiv.org/abs/2402.17764

### Quantization:

https://github.com/state-spaces/mamba/issues/133 (only quantize nn.linear)

https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/quantization

https://leimao.github.io/article/Neural-Networks-Quantization/

### Fast matrix mult:

https://coffeebeforearch.github.io/2020/06/23/mmul.html

https://justine.lol/matmul/

# mamba.cpp

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

Inference of Mamba models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c)

This implements only the recurrent mode of Mamba SSM

You can compare it with the [related pytorch implementation](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

No support for batches. The code is minimal for learning purposes.

Even so, it is faster than pytorch on CPU!!!

## Models

You can use these models stored on [HuggingFace](https://huggingface.co/state-spaces):

* `state-spaces/mamba-130m`
* `state-spaces/mamba-370m`
* `state-spaces/mamba-790m`
* `state-spaces/mamba-1.4b`
* `state-spaces/mamba-2.8b`
* `state-spaces/mamba-2.8b-slimpj`

You can specify the model name as an argument to the `export.py` script

Note that the export script will download the model (if it's not already downloaded) to the hugingface cache directory.

Optionally you can also specify the path to the model file, if you downloaded it manually. Example:

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
```

## Internal State

As it is a recurrent model, it is possible to save the internal state and then return to that state later

To get a copy of the internal state:

```c
  int state_size;
  char* state = get_internal_state(mamba, &state_size);
```

To set the internal state:

```c
  set_internal_state(mamba, state, state_size);
```
