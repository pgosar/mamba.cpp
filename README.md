To run:

1. python3 scripts/download_models.py -m 370m --bits 32 -md models/370m_32bit.bin
2. make fast
3. ./build/mamba models/model.bin -n 20 -i "Customer Support should" -t 0.0

Command line arguments will be used to control inference, for example, quantization level,
debugging verbosity, input prompt.

You can use the download models shell script to download the useful configurations for testing, including tokenizers.

## TODO
Model configuration will be done through model_config.yaml, for example, temperature (text
diversity), generated text amount, batch size. There may be multiple selectable configurations,
these are selected through the command line arguments.

## TODO
- [x] Initial C++ Implementation
    
- [x] C++ Memory optimization

- [x] Quantization
  
- [ ] Speculative Decoding

- [ ] Flash mem
    - [x] neuron activation data
    - [x] hot and cold neurons prediction
    - [ ] Actually load in partial model
          
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
