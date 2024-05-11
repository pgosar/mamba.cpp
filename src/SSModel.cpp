#include "SSModel.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <vector>
#include <algorithm>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

SSModel::SSModel() {}

SSModel::~SSModel() {
  if(data != MAP_FAILED)
    munmap(data, filesize);
}

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

SSModel from_safetensors(FILE* fp) {
  size_t N;
  fread(&N, sizeof(size_t), 1, fp);

  size_t size = sizeof(char) * (N+1);
  char* header = (char*) malloc(size);
  fread(header, sizeof(char), size, fp);
  header[N] = 0;

  printf("%ld\n", N);

  free(header);

  fclose(fp);

  //todo replace
  return SSModel();
}

SSModel SSModel::from_pretrained(const std::string& filepath) {
  FILE* fp = fopen(filepath.c_str(),"rb");
  if(!fp) {
    printf("Model file not found\n");
    exit(EXIT_FAILURE);
  }

  unsigned int magic;
  //todo there should be a better way to do this long-term
  if (fread(&magic, sizeof(int), 1, fp) != 1) { exit(EXIT_FAILURE); }
  if (magic != 0x4d616d62) {
    fseek(fp, -sizeof(int), SEEK_CUR);
    return from_safetensors(fp);
  }

  int version;
  if (fread(&version, sizeof(int), 1, fp) != 1) { exit(EXIT_FAILURE); }

  SSModel model;
  
  if (fread(&model.cfg, sizeof(SSMConfig), 1, fp) != 1) { exit(EXIT_FAILURE); }
  if (model.cfg.vocab_size % 8 != 0) {
    model.cfg.rounded_vocab_size = model.cfg.vocab_size + (8 - (model.cfg.vocab_size % 8));
  } else {
    model.cfg.rounded_vocab_size = model.cfg.vocab_size;
  }

  fseek(fp, 0, SEEK_END); // move file pointer to end of file
  model.filesize = ftell(fp); // get the file size, in bytes
  fclose(fp);

  int fd = open(filepath.c_str(), O_RDONLY); // open in read only mode
  if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }

  model.data = (float*) mmap(NULL, model.filesize, PROT_READ, MAP_PRIVATE, fd, 0);

  if (model.data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
  //once mmap is complete, we may close the file
  close(fd);

  float* weights_ptr = model.data + (256 / 4);

  //todo weights

  return model;
}

torch::Tensor forward_layer()

torch::Tensor SSModel::forward(torch::Tensor& input_ids, torch::Tensor& position_ids, InferenceParams& inference_params, int num_last_tokens=0) {
  torch::Tensor hidden_states;

  //TODO implement

  //call forward on each layer (todo parallelize??)
  for(int l = 0; l < cfg.n_layers; l++) {
    // rmsnorm(hidden_state, input, w->norm + l * dim, dim);

    // forward_layer();
  }

  return hidden_states;
}

torch::Tensor SSModel::get_logits(torch::Tensor& input_ids, InferenceParams& inference_params, long batch_size) {
  bool decoding = inference_params.seqlen_offset > 0;
  torch::Tensor position_ids; //todo None?
  if(decoding) {
    auto options = torch::TensorOptions().dtype(torch::kLong).device(input_ids.device()); //TODO use global options
    position_ids = torch::full({batch_size, 1}, inference_params.seqlen_offset, options); //TODO looks like this is unused
  } 

  torch::Tensor logits = forward(input_ids, position_ids, inference_params, 1);

  return logits;
}

torch::Tensor SSModel::sample_tokens(torch::Tensor& logits, Config& cfg) {
  torch::Tensor token = sample(logits, cfg.top_k, cfg.top_p, cfg.min_p, cfg.temperature);
  return token.unsqueeze(1);
}

bool SSModel::should_stop(torch::Tensor current_token, InferenceParams& inference_params, int max_length) {
  if(inference_params.seqlen_offset == 0) return false;

  //TODO eos token ID?

  return inference_params.seqlen_offset >= max_length - 1;
}

void SSModel::generate(torch::Tensor& input_ids, Config cfg) {
  //Decode logic
  //Top-k -> top-p
  long batch_size = input_ids.sizes()[0];
  long seq_len_og = input_ids.sizes()[1];

  //Assume CG

  //TODO
  //decoding cache - only for CUDA
  //inference params
  InferenceParams inference_params(cfg.max_response_length, batch_size);

  std::vector<torch::Tensor> sequences, scores;
  sequences.push_back(input_ids);

  torch::Tensor sequences_cat = input_ids;

  while(!should_stop(sequences.back(), inference_params, cfg.max_response_length)) {
    scores.push_back(get_logits(sequences.back(), inference_params, batch_size));
    inference_params.seqlen_offset += sequences.back().sizes()[1];

    torch::Tensor sampled_tokens;
    if(cfg.repetition_penalty == 1.0) {
      sampled_tokens = sample_tokens(scores.back(), cfg);
    } else {
      torch::Tensor logits = modify_logit_for_repetition_penalty(
          scores.back().clone(), sequences_cat, cfg.repetition_penalty
      );
      sampled_tokens = sample_tokens(logits, cfg);
      sequences_cat = torch::cat({sequences_cat, sampled_tokens}, 1);
    }
    sequences.push_back(sampled_tokens);
  }

  //todo return outputcls with sequences and scores

  //will want timing, but not CUDA
}

#ifdef CUDA_SSM
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
#endif

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

//TODO let's try and improve this
torch::Tensor& modify_logit_for_repetition_penalty(
  torch::Tensor logits, torch::Tensor& prev_output_tokens, float repetition_penalty) {
    if (repetition_penalty == 1.0) return logits;
    torch::Tensor score = torch::gather(logits, 1, prev_output_tokens);
    // if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch::where(score < 0, score * repetition_penalty, score / repetition_penalty);
    logits.scatter_(1, prev_output_tokens, score);
    return logits;
}

torch::Tensor SSModel::sample(
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

MixerModel::MixerModel(int d_model, int n_layer, int vocab_size, 
  Config& ssm_cfg, float norm_epsilon, bool rms_norm, bool fused_add_norm,
  bool residual_in_fp32, torch::Device device, torch::Dtype dtype) : 
    _residual_in_fp32(residual_in_fp32),
    _fused_add_norm(fused_add_norm) {
  //_embedding(vocab_size, d_model, device=device, dtype=dtype);
  
  //TODO layers
  for(int i = 0; i < n_layer; i++) {
    
  }
  
  // _layers();

  //TODO normalization

  //TODO apply partial
}

Block::Block() {
  
}

// static std::tuple<torch::Tensor, std::optional<torch::Tensor>> Block::forward(torch::Tensor hidden_states, std::optional<torch::Tensor> residual, Config& inference_params) {
//   if (!_fused_add_norm) {
//     residual = (residual == NULL ? hidden_states : residual + hidden_states);
//     hidden_states = _layer_norm(residual.to(_layer_norm.weight.dtype));
//     if (_residual_in_fp32) {
//       residual = residual.to(torch::kFloat32);
//     }
//   } else {
//     _layer_norm.fused_add_norm_fn(hidden_states, _layer_norm.weight, _layer_norm.bias, residual, true, _residual_in_fp32, _layer_norm.eps); 
//   }
//   hidden_states = _mixer_norm(hidden_states, inference_params)
//   return std::make_tuple(hidden_states, residual);
// }