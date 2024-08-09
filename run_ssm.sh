cd build && ninja && cd .. && ./build/fast_mamba_inference models/model.bin -n 200 -i "1+1=" -t 0.0 -r 0.95
