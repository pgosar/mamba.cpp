// Mamba model

#ifndef MAMBA_HPP
#define MAMBA_HPP

#include <unistd.h>
#include "math.hpp"

// For int8_t
#define TO_FLOAT_INT8(x) ((float)(x) / 255.0e7f)

#define FROM_FLOAT_INT8(x)                                                     \
  ((x) < 0 ? 0 : (x) > 1e-7f ? 255 : (int8_t)(255.0e7f * (x)))

#define TO_FLOAT_ARRAY_INT8(array, length)                                     \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = TO_FLOAT_INT8((array)[i]);                                    \
  }

#define FROM_FLOAT_ARRAY_INT8(array, length)                                   \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = FROM_FLOAT_INT8((array)[i]);                                  \
  }

// For int16_t
#define TO_FLOAT_INT16(x) ((float)(x) / 32768.0e7f)

#define FROM_FLOAT_INT16(x)                                                    \
  ((x) < 0 ? 0 : (x) > 1e-7f ? 32767 : (int16_t)(32768.0e7f * (x)))

#define TO_FLOAT_ARRAY_INT16(array, length)                                    \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = TO_FLOAT_INT16((array)[i]);                                   \
  }

#define FROM_FLOAT_ARRAY_INT16(array, length)                                  \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = FROM_FLOAT_INT16((array)[i]);                                 \
  }

// For int32_t
#define TO_FLOAT_INT32(x) ((float)(x) / 2147483648.0e7f)

#define FROM_FLOAT_INT32(x)                                                    \
  ((x) < 0 ? 0 : (x) > 1e-7f ? 2147483647 : (int32_t)(2147483648.0e7f * (x)))

#define TO_FLOAT_ARRAY_INT32(array, length)                                    \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = TO_FLOAT_INT32((array)[i]);                                   \
  }

#define FROM_FLOAT_ARRAY_INT32(array, length)                                  \
  for (int i = 0; i < (length); ++i) {                                         \
    (array)[i] = FROM_FLOAT_INT32((array)[i]);                                 \
  }

typedef struct {
  int n_layers;   // number of layers
  int vocab_size; // vocabulary size
  int dim;        // embedding dimension
  int d_inner;
  int dt_rank;
  int d_state;
  int d_conv;
  int num_bits;
  long long longest_tensor_len; // length of the longest weights tensor, flattened
  int rounded_vocab_size; // vocab_size rounded up to the nearest multiple of 8
} Config;

typedef struct {
  float repetition_penalty; // 1 for no penalty, >1 for high repetition, <1 for
                            // low repetition
} UserConfig;

//readonly
template <typename T> struct MambaWeights {
  const Tensor<T> token_embedding_table; // (rounded_vocab_size, dim)              
  const Tensor2D<T> in_proj;               // (layer, 2*d_inner, dim)                
  const Tensor2D<T> conv1d_weight;         // (layer, d_inner, 1, d_conv)
  const Tensor2D<T> conv1d_bias;           // (layer, d_inner)
  const Tensor2D<T> x_proj;                // (layer, dt_rank+2*d_state, d_inner)
  const Tensor2D<T> dt_proj_weight;        // (layer, d_inner, dt_rank)
  const Tensor2D<T> dt_proj_bias;          // (layer, d_inner)
  const Tensor2D<T> A;                     // (layer, d_inner, d_state)
  const Tensor2D<T> D;                     // (layer, d_inner)
  const Tensor2D<T> out_proj;              // (layer, dim, d_inner)
  const Tensor2D<T> norm;                  // (layer, dim)
  const Tensor<T> final_norm;            // (dim)
  const Tensor<T> lm_head;               // (rounded_vocab_size, dim)
};

//read and written to
template <typename T> struct RunState {
  // memory reused by all layers
  T *input;        // (dim)
  T *hidden_state; // (dim)
  T *xz;           // (2*d_inner)          x and z are pointers into this buffer
  T *x_db;   // (dt_rank+2*d_state)  dt, B, C are pointers into this buffer
  T *dt;     // (d_inner)            later, dt is a pointer to this buffer
  T *dA;     // (d_inner, d_state)
  T *dB;     // (d_inner, d_state)
  T *temp;   // (d_inner, d_state)
  T *y;      // (d_inner)
  T *logits; // (rounded_vocab_size)
  // internal state, separate memory for each layer
  T *conv_state; // (n_layers, d_inner, d_conv)
  T *ssm_state;  // (n_layers, d_inner, d_state)
  float *dequantized_buffer;
};

template <typename T> struct Mamba {
  Config config; // the hyperparameters of the architecture (the blueprint)
  MambaWeights<T> weights; // the weights of the model
  RunState<T>
      state; // buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;           // file descriptor for memory mapping
  T *data;          // memory mapped data pointer
  size_t file_size; // size of the checkpoint file in bytes
};

#endif // MAMBA_HPP
