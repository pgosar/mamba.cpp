#include "Norm.hpp"

#include <algorithm>

static size_t _log(size_t x) {
  size_t result = 0;
  while(x>>=1)  {
    result++;
  }
  return result;
}

torch::Tensor LayerNormFn::forward(
  AutogradContext& ctx, 
  torch::Tensor& input, torch::Tensor& weight, 
  torch::Tensor& bias, torch::Tensor& residual, 
  float eps, bool prenorm, bool residual_in_fp32, bool is_rms_norm=false) {

  auto input_shape_og = input.sizes();
  input = input.reshape({-1, input.sizes()[0]});

  if(input.stride(-1) != 1) {
    input = input.contiguous();
  }

  caffe2::TypeMeta residual_dtype;
  if(residual.defined()) {
    //assert
    residual = residual.reshape({-1, residual.sizes()[-1]});
    if(residual.stride(-1) != 1) {
      residual = residual.contiguous();
    }
    
    residual_dtype = residual.dtype();
  } else {
    residual_dtype = caffe2::TypeMeta::fromScalarType(torch::kFloat32);
  }

  weight = weight.contiguous();
  if(bias.defined()) {
    bias = bias.contiguous();
  }

  //TODO this is proly wrong
  int M = input.sizes()[0];
  int N = input.sizes()[1];

  torch::Tensor output = torch::empty_like(input);
  torch::Tensor residual_out;

  if(residual.defined() || residual_dtype != input.dtype()) {
    residual_out = torch::empty({M, N});
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

  //Translation of _layer_norm_fwd

  torch::Tensor mean;
  if(is_rms_norm) {
    mean = torch::empty({M,}, options);
  }

  torch::Tensor rstd = torch::empty({M,}, options);

  size_t MAX_FUSED_SIZE = 65536 / input.element_size();
  size_t BLOCK_N = std::min(MAX_FUSED_SIZE, _log(N));

  //This stuff used to be CUDA
  for(int row = 0; row < M; row++) {
    int X = row * input.stride(0);
    int Y = row * output.stride(0);

    
  }

  // ctx.save_for_backward(residual_out, weight, bias, mean, rstd);
  
  // if(prenorm) {
  //   return y;
  // }

  //return {y, residual_out.reshape(input_shape_og)};
}