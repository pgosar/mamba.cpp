To run:

1. Create build directory
2. vcpkg install
3. python3 download_models.py
4. ./build.sh

Command line arguments will be used to control inference, for example, quantization level,
debugging verbosity, input prompt.

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
