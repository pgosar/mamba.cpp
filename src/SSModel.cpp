#include "SSModel.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <algorithm>

SSModel::SSModel() {}

/**
 * Follow along modeling_utils.py:4029, reading in model step-by-step
 * First read safetensors metadata and header, then go from there
 * Verify correctness as we go
 * Lets get this rock solid
 * 
 * Once this is chill, migrate to quantized format
 * And load that
 * We can generate quantized model from python, that's fine and dandy
*/

SSModel SSModel::from_pretrained(const std::string& filepath) {
  FILE* fp = fopen(filepath.c_str(),"rb");

  if(fp) {
    size_t N;
    fread(&N, sizeof(size_t), 1, fp);

    size_t size = sizeof(char) * (N+1);
    char* header = (char*) malloc(size);
    fread(header, sizeof(char), size, fp);
    header[N] = 0;

    printf("%ld\n", N);

    free(header);
  } else {
    //File not found
    printf("Model file not found\n");
  }

  fclose(fp);

  return SSModel();
}

void SSModel::generate(torch::Tensor& input_ids, Config cfg) {
  //Decode logic
  //Top-k -> top-p
  unsigned long batch_size = input_ids.sizes()[0];
  unsigned long seq_len_og = input_ids.sizes()[1];

  //Assume CG

  //TODO
  //decoding cache
  //inference params

  //External functions:
  //get logits
  //sample tokens
  //should stop

  //will want timing, but not CUDA
  
  //main loop
  //but NOT CUDA
}

void SSModel::allocate_inference_cache(
  int batch_size, 
  int max_seq_len, 
  torch::Dtype dtype) {
  
}

void SSModel::update_graph_cache(
  int batch_size, int seqlen_og, int max_seqlen, torch::Dtype dtype) { 
    //TODO decoding_seqlens, warmups
  
  //TODO infer device and dtype from model params
  if (_decoding_cache.device != torch::kCPU ||
    _decoding_cache.dtype != dtype || 
    batch_size > _decoding_cache.max_batch_size || 
    max_seqlen > _decoding_cache.max_seqlen) {
      _decoding_cache.device = torch::kCPU;
      _decoding_cache.dtype = dtype;
      _decoding_cache.max_batch_size = batch_size;
      _decoding_cache.max_seqlen = max_seqlen;
      // _decoding_cache.params = 
    }
}

void modify_logits_for_min_p_filtering(torch::Tensor& logits, float min_p) {
  if(min_p <= 0.0 || min_p >= 1.0) return;

  auto indices_to_remove = logits < min_p;
  logits.masked_fill_(indices_to_remove, -std::numeric_limits<float>::infinity());
}

void modify_logits_for_top_p_filtering(torch::Tensor& logits, float top_p) {
  if(top_p <= 0.0 || top_p >= 1.0) return;

  auto [sorted_logits, sorted_indices] = logits.sort();
  auto cumulative_probs = sorted_logits.softmax(-1).cumsum(-1);

  auto sorted_indices_to_remove = cumulative_probs <= (1 - top_p);

  auto indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove);
  logits.masked_fill_(indices_to_remove, -std::numeric_limits<float>::infinity());
}

void modify_logits_for_top_k_filtering(torch::Tensor& logits, int top_k) {
    auto topk_result = logits.topk(top_k);
    auto topk_values = std::get<0>(topk_result);
    auto topk_threshold = topk_values[-1].unsqueeze(1);
    auto indices_to_remove = logits < topk_threshold;

    logits.masked_fill_(indices_to_remove, -std::numeric_limits<float>::infinity());
}

torch::Tensor& modify_logit_for_repetition_penalty(
  torch::Tensor& logits, torch::Tensor& prev_output_tokens, float repetition_penalty) {
    if (repetition_penalty == 1.0) return logits;
    torch::Tensor score = torch::gather(logits, 1, prev_output_tokens);
    // if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch::where(score < 0, score * repetition_penalty, score / repetition_penalty);
    logits.scatter_(1, prev_output_tokens, score);
    return logits;
}

torch::Tensor sample(
  torch::Tensor& logits, int64_t top_k=1, float top_p=0.0, float min_p=0.0, float temperature=1.0) {
  /* Sample from top-k logits.
  Arguments:
      logits: Tensor of shape (batch_size, vocab_size) */
  if(top_k == 1) { // Short-circuit for greedy decoding
    return logits.argmax(-1);
  } else {
    if(top_p > 0.0) {
      assert(top_p <= 1.0);
    }
    if(top_k > 0) {
        top_k = std::min(top_k, logits.size(-1));  // Safety check

        auto [logits_top, indices] = torch::topk(logits, top_k, -1);
        if(temperature != 1.0) logits_top /= temperature;

        modify_logits_for_top_p_filtering(logits_top, top_p);

        
        return indices.index({
            torch::arange(indices.size(0), indices.device()),
            torch::multinomial(torch::softmax(logits_top, -1), 1).squeeze(-1)
        });
    } else {
      auto logits_top = logits.clone();
      if(min_p > 0.0) {
        auto max_prob = logits_top.index({"...",0}).item<float>();
        auto min_prob = max_prob * min_p;
        modify_logits_for_min_p_filtering(logits_top, min_p);
        if  (temperature != 1.0){
          logits_top /= temperature;
        }
        return torch::multinomial(torch::softmax(logits_top, -1), 1).squeeze(-1);
      }
      if (temperature != 1.0) {
        logits_top /= temperature;
      }
      modify_logits_for_top_p_filtering(logits_top, top_p);
      return torch::multinomial(torch::softmax(logits_top, -1), 1).squeeze(-1);
    }
  }
}