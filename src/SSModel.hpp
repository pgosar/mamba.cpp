#ifndef SSModel_hpp
#define SSModel_hpp

#include <string>
#include <vector>
#include <torch/torch.h>
#include <ATen/Tensor.h>

#include "util.hpp"

/**
 * Needed for inference:
 *  from_pretrained (adapted)
 *  eval
 *  generate
*/

struct DecodingCGCache {
  int max_batch_size;
  int max_seqlen;
  torch::DeviceType device;
  torch::Dtype dtype;
  InferenceParams params;
  //TODO run, callables

  DecodingCGCache() :
    max_batch_size(0),
    max_seqlen(0),
    device(torch::kCPU),
    dtype(torch::kFloat32),
    params(0, 0)
  {}
};

class SSModel : torch::nn::Module {
private:
  //config data
  //config
  const torch::DeviceType device = torch::kCPU; 
  DecodingCGCache _decoding_cache;

public: 
  SSModel();
  static SSModel from_pretrained(const std::string&);

  void generate(torch::Tensor&, Config);

  void allocate_inference_cache(int, int, torch::Dtype);
  void update_graph_cache(int, int, int, torch::Dtype);

  // torch::Tensor forward(Graph tree);
};

void modify_logits_for_min_p_filtering(torch::Tensor&, float);
void modify_logits_for_top_p_filtering(torch::Tensor&, float);
void modify_logits_for_top_k_filtering(torch::Tensor&, int);

torch::Tensor& modify_logit_for_repetition_penalty(torch::Tensor&, torch::Tensor&, float);

#endif