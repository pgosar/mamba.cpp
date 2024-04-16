#ifndef NORM_HPP
#define NORM_HPP

#include <torch/torch.h>
#include <ATen/Tensor.h>

using namespace torch::autograd;

class LayerNormFn : public Function<LinearFunction> {

}

#endif