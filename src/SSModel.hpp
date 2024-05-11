#ifndef SSModel_hpp
#define SSModel_hpp

#include <string>
#include <vector>
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include "Norm.hpp"
#include "util.hpp"

/**
 * Needed for inference:
 *  from_pretrained (adapted)
 *  eval
 *  generate
*/

//BIG TODO ints->longs probably needed for larger models

#ifdef CUDA_SSM
typedef torch::Tensor (*callable)(torch::Tensor&, torch::Tensor&, int);

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

  torch::Tensor run(torch::Tensor& input_ids, torch::Tensor& position_ids, int seqlen) {
    int batch_size = input_ids.sizes()[0];
    int decoding_seqlen = input_ids.sizes()[0];

    return callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen);
  }
};
#endif

class MixerModel : torch::nn::Module {
private:
  bool _residual_in_fp32;
  bool _fused_add_norm;
  //torch::nn::Embedding _embedding;
  //torch::nn::ModuleList _layers;

public:
  MixerModel(int, int, int, Config&, float, bool, bool, bool, 
  torch::Device, torch::Dtype);
};

struct Block : torch::nn::Module {
private:
  bool _residual_in_fp32;
  bool _fused_add_norm;
  LayerNormFn _layer_norm;
  //MixerNormFn _mixer_norm;
  //torch::nn::Embedding _embedding;
  //torch::nn::ModuleList _layers;

public:
  Block();
  //std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward();
};

//TODO we'll need to credit if we keep this in when going public
struct SSMConfig {
  int n_layers;   // number of layers
  int vocab_size; // vocabulary size
  int dim;        // embedding dimension
  int d_inner;
  int dt_rank;
  int d_state;
  int d_conv;
  int shared_classifier;
  int rounded_vocab_size;
};

class SSModel : torch::nn::Module {
private:
  //config data
  //config
  const torch::DeviceType device = torch::kCPU;
  SSMConfig cfg;

  float* data;  //so we may unmap in destructor
  size_t filesize;

#ifdef CUDA_SSM
  DecodingCGCache _decoding_cache;
#endif

  torch::Tensor get_logits(torch::Tensor&, InferenceParams&, long);
  torch::Tensor sample_tokens(torch::Tensor&, Config&);
  bool should_stop(torch::Tensor, InferenceParams&, int);
    
  torch::Tensor forward_layer(torch::Tensor&, torch::Tensor&, InferenceParams&, int);

public: 
  SSModel();
  ~SSModel();
  static SSModel from_pretrained(const std::string&);

  void generate(torch::Tensor&, Config);
  torch::Tensor forward(torch::Tensor&, torch::Tensor&, InferenceParams&, int);
  torch::Tensor sample(torch::Tensor&, int64_t, float, float, float);

#ifdef CUDA_SSM
  void allocate_inference_cache(int, int, torch::Dtype);
  void update_graph_cache(int, int, int, torch::Dtype);

  torch::Tensor forward(Graph tree);
#endif
};

void modify_logits_for_min_p_filtering(torch::Tensor&, float);
void modify_logits_for_top_p_filtering(torch::Tensor&, float);
void modify_logits_for_top_k_filtering(torch::Tensor&, int);

torch::Tensor& modify_logit_for_repetition_penalty(torch::Tensor, torch::Tensor&, float);

#endif