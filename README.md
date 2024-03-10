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

Helpful references:

https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910

https://arxiv.org/abs/2310.04564

https://arxiv.org/abs/2312.11514

https://arxiv.org/abs/2402.11131

https://arxiv.org/abs/2305.13245

https://pytorch.org/blog/quantization-in-practice/

https://leimao.github.io/article/Neural-Networks-Quantization/

https://coffeebeforearch.github.io/2020/06/23/mmul.html

