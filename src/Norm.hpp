#ifndef Norm_hpp
#define Norm_hpp

#include <torch/torch.h>
#include <ATen/Tensor.h>

using namespace torch::autograd;

class LayerNormFn : public Function<LayerNormFn> {
public:
  static torch::Tensor forward(AutogradContext&, torch::Tensor&, 
  torch::Tensor&, torch::Tensor&, torch::Tensor&, float, bool, bool, bool);

  static torch::Tensor backward(AutogradContext*, tensor_list grad_outputs);
};

#endif