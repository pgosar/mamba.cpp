#ifndef UTIL_HPP
#define UTIL_HPP

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "mamba.hpp"

inline void malloc_run_state(RunState *s, Config *p) {
  // memory reused by all layers
  try {
    s->input = static_cast<float *>(malloc(p->dim * sizeof(float)));
    s->hidden_state = static_cast<float *>(malloc(p->dim * sizeof(float)));
    s->xz = static_cast<float *>(malloc(2 * p->d_inner * sizeof(float)));
    s->x_db = static_cast<float *>(
        malloc((p->dt_rank + 2 * p->d_state) * sizeof(float)));
    s->dt = static_cast<float *>(malloc(p->d_inner * sizeof(float)));
    s->dA =
        static_cast<float *>(malloc(p->d_inner * p->d_state * sizeof(float)));
    s->dB =
        static_cast<float *>(malloc(p->d_inner * p->d_state * sizeof(float)));
    s->temp =
        static_cast<float *>(malloc(p->d_inner * p->d_state * sizeof(float)));
    s->y = static_cast<float *>(malloc(p->d_inner * sizeof(float)));
    s->logits =
        static_cast<float *>(malloc(p->rounded_vocab_size * sizeof(float)));
    // internal state, separate memory for each layer
    s->conv_state = static_cast<float *>(
        calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(float)));
    s->ssm_state = static_cast<float *>(
        calloc(p->n_layers * p->d_inner * p->d_state, sizeof(float)));
  } catch (std::bad_alloc &e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline void reset_internal_state(Mamba *mamba) {
  // reset the internal state of the model
  RunState *s = &mamba->state;
  Config *p = &mamba->config;
  memset(s->conv_state, 0,
         p->n_layers * p->d_inner * p->d_conv * sizeof(float));
  memset(s->ssm_state, 0,
         p->n_layers * p->d_inner * p->d_state * sizeof(float));
}

inline void free_run_state(RunState *s) {
  free(s->input);
  free(s->hidden_state);
  free(s->xz);
  free(s->x_db);
  free(s->dt);
  free(s->dA);
  free(s->dB);
  free(s->temp);
  free(s->y);
  free(s->logits);
  free(s->conv_state);
  free(s->ssm_state);
}

inline void memory_map_weights(MambaWeights *w, Config *p, float *ptr) {
  // the multiplications below are done in 64-bit to fit the parameter counts of
  // 13B+ models
  unsigned long long n_layers = p->n_layers;
  // get the pointers to the weights
  w->token_embedding_table = ptr;
  ptr += p->rounded_vocab_size * p->dim;
  w->in_proj = ptr;
  ptr += n_layers * (2 * p->d_inner) * p->dim;
  w->conv1d_weight = ptr;
  ptr += n_layers * p->d_inner * 1 * p->d_conv;
  w->conv1d_bias = ptr;
  ptr += n_layers * p->d_inner;
  w->x_proj = ptr;
  ptr += n_layers * (p->dt_rank + 2 * p->d_state) * p->d_inner;
  w->dt_proj_weight = ptr;
  ptr += n_layers * p->d_inner * p->dt_rank;
  w->dt_proj_bias = ptr;
  ptr += n_layers * p->d_inner;
  w->A = ptr;
  ptr += n_layers * p->d_inner * p->d_state;
  w->D = ptr;
  ptr += n_layers * p->d_inner;
  w->out_proj = ptr;
  ptr += n_layers * p->dim * p->d_inner;
  w->norm = ptr;
  ptr += n_layers * p->dim;
  w->final_norm = ptr;
  ptr += p->dim;
  // the classifier weights can be shared with the token embedding table
  w->lm_head = w->token_embedding_table;
  /*for(int i = 0; i < p->rounded_vocab_size * p->dim; i++) {
	 if(w->token_embedding_table[i] != 0)
	  std::cout << w->token_embedding_table[i] << std::endl;
  }*/
}

inline float* quantized_to_float_tensor(size_t dim, uint8_t num_bits,
                                        uint8_t** ptr_to_ptr, int n_layers = 1) {
  uint8_t* ptr = *ptr_to_ptr;
  float* tensor = (float*) malloc(n_layers * dim * sizeof(float));

  uint8_t val = 0;
  uint8_t pos = 0;

  for(int l = 0; l < n_layers; l++) {
    float* layer = tensor + l * dim;

    float scale = *(float*)ptr;
    ptr += sizeof(float);

    float zeropoint = *(float*)ptr;
    ptr += sizeof(float);

    for(int i = 0; i < dim; i++) {
      // int quantized_value = (*(int*)ptr) & quant_mask;
      // short quantized_val = 0;
      // for(int b = 0; b < num_bits; b++) {
      //   if(pos == 0) {
      //     ptr++;
      //     val = *ptr;
      //   }
      //   quantized_val |= (val & 1) << b;
      //   val >>= 1;
      //   pos = (pos + 1) & 0x7;
      // }
      int16_t quantized_val = *(int16_t*)ptr;
      ptr += sizeof(int16_t);

      layer[i] = (quantized_val - zeropoint) / scale;
    }
  }

  *ptr_to_ptr = ptr;
  return tensor;
}

inline void memory_map_quantized_weights(MambaWeights *w, Config *p, uint8_t *ptr) {
  // the multiplications below are done in 64-bit to fit the parameter counts of
  // 13B+ models
  unsigned long long n_layers = p->n_layers;
  // get the pointers to the weights
  // todo get quantization metadata at start of each tensor, if necessary

  size_t tensor_dim = p->rounded_vocab_size * p->dim;
  w->token_embedding_table = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr);

  tensor_dim = (2 * p->d_inner) * p->dim;
  w->in_proj = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner * 1 * p->d_conv;
  w->conv1d_weight = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner;
  w->conv1d_bias = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = (p->dt_rank + 2 * p->d_state) * p->d_inner;
  w->x_proj = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner * p->dt_rank;
  w->dt_proj_weight = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner;
  w->dt_proj_bias = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner * p->d_state;
  w->A = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->d_inner;
  w->D = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->dim * p->d_inner;
  w->out_proj = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->dim;
  w->norm = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr, n_layers);

  tensor_dim = p->dim;
  w->final_norm = quantized_to_float_tensor(tensor_dim, p->num_bits, &ptr);

  // the classifier weights can be shared with the token embedding table
  w->lm_head = w->token_embedding_table;
  /*for(int i = 0; i < p->rounded_vocab_size * p->dim; i++) {
	 if(w->token_embedding_table[i] != 0)
	  std::cout << w->token_embedding_table[i] << std::endl;
  }*/
}

inline void load_model_file(const std::string& model_path, Config* config,
                            MambaWeights *weights, int *fd, float **data,
                            std::size_t *file_size) {
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Couldn't open file " << model_path << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get the file size
  *file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // read the config
  // don't read rounded vocab size; that's computed here
  if (!file.read(reinterpret_cast<char*>(config), sizeof(Config))) {
    std::exit(EXIT_FAILURE);
  }

  if (config->vocab_size % 8 != 0) {
    config->rounded_vocab_size =
        config->vocab_size + (8 - (config->vocab_size % 8));
  } else {
    config->rounded_vocab_size = config->vocab_size;
  }

  // todo adjust dimensions to account for activation bool tensor

  file.close();

  // memory map the model weights into the data pointer
  // TODO: check on performance of mmap vs alternatives
  *fd = open(model_path.c_str(), O_RDONLY);
  if (*fd == -1) {
    std::cerr << "open failed!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  *data = static_cast<float *>(
      mmap(nullptr, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
  if (*data == MAP_FAILED) {
    std::cerr << "mmap failed!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + (256 / sizeof(float));

  if(config->num_bits < 32) {
    memory_map_quantized_weights(weights, config, (uint8_t*) weights_ptr);
    //we'll need to allocate new data for the dequantized f32 weights
    munmap(data, *file_size);
  } else {
    memory_map_weights(weights, config, weights_ptr);
  }
}

inline void load_model(Mamba *m, char *model_path) {
  // read the Config and the Weights from the model file
  load_model_file(model_path, &m->config, &m->weights, &m->fd, &m->data,
                  &m->file_size);
  // allocate the RunState buffers
  malloc_run_state(&m->state, &m->config);
}

inline void free_model(Mamba *m) {
  // close the memory mapping
  if (m->data != MAP_FAILED) {
    munmap(m->data, m->file_size);
  }
  if (m->fd != -1) {
    close(m->fd);
  }
  // free the RunState buffers
  free_run_state(&m->state);
}

// ----------------------------------------------------------------------------
// utilities: time

inline long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// -------------------
// minimize repetition
inline void apply_repetition_penalty(float *logits,
                                     std::vector<int> &prev_tokens,
                                     float penalty) {
  if (penalty == 1.0)
    return;

  // Gather -> Handle Negative -> Scatter
  for (size_t i = 0; i < prev_tokens.size(); i++) {
    float score = logits[prev_tokens[i]];
    logits[prev_tokens[i]] = score > 0 ? score * penalty : score / penalty;
  }
}

#endif // UTIL_HPP
