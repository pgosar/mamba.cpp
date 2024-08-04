#ifndef MATH_HPP
#define MATH_HPP

#include "tensor.hpp"

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

template <typename T> inline void rmsnorm(
  EnhancedTensor<T>& o, 
  const Tensor<T>& x, 
  const Tensor<T>& weight, 
  T *tempbuf,
  int size
  ) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    float xj = x[j];
    ss += xj * xj;
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  EnhancedTensor<float> temp(tempbuf, size);
  for (size_t j = 0; j < size; j++) {
    temp[j] = x[j] * weight[j] * ss; //Todo new macro to dequantize
  }

  o.requantize(temp);
}

template <typename T> inline void softmax(EnhancedTensor<T>& x, T* tempbuf, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    max_val = std::max(max_val, x[i]);
  }
  
  EnhancedTensor<float> temp(tempbuf, size);
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    temp[i] = expf(x[i] - max_val);
    sum += temp[i];
  }

  // normalize
  for (int i = 0; i < size; i++) {
    temp[i] /= sum;
  }

  x.requantize(temp);
}

inline float softplus(float x) { return logf(1.0f + expf(x)); }

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

inline float silu(float x) { return x * sigmoid(x); }

template <typename T>
inline void shift_matrix_left(EnhancedTensor2D<T>& matrix, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols - 1; j++) {
      matrix.set(i * cols + j, matrix.get(i * cols + j + 1));
    }
  }
}

template <typename T>
inline void update_last_column(EnhancedTensor2D<T>& matrix, const Tensor<T>& x, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    matrix.set(i * cols + cols - 1, x.get(i));
  }
}

template <typename T>
inline void shift_left_and_update_last(
  EnhancedTensor<T>& matrix, 
  const Tensor<T>& x, 
  float* tempbuf, int rows, int cols) {
  EnhancedTensor<float> temp = matrix.dequantize(tempbuf, rows * cols);

#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols - 1; j++) {
      temp[i * cols + j] = temp[i * cols + j + 1];
    }
    temp[i * cols + cols - 1] = x[i];
  }

  matrix.requantize(temp);
}

template <typename T>
inline void rowwise_dot_product(
  EnhancedTensor<T>& out, 
  const Tensor2D<T>& matrix, 
  const Tensor<T>& weights, 
  float* tempbuf,
  int rows,
  int cols) {
// matrix[rows,cols], weights[cols] -> out[rows]
// this is a dot product of each row of the matrix with the weights
// i.e. out[i] = matrix[i,:] @ weights
EnhancedTensor<float> temp(tempbuf, rows);

#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    float val = 0.0f;
    for (int j = 0; j < cols; j++) {
      val += matrix[i * cols + j] * weights[j];
    }
    temp[i] = val;
  }

  out.requantize(temp);
}

//TODO optimize
template <typename T> inline void matmul(
  EnhancedTensor<T>& xout, 
  const Tensor<T>& x, 
  const Tensor<T>& w, 
  float *tempbuf,
  int d, 
  int n) {
// w[d,n] @ x[n] -> xout[d]
  EnhancedTensor<float> temp(tempbuf, d);

#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    temp[i] = val;
  }

  xout.requantize(temp);
}

template <typename T>
inline void linear(EnhancedTensor<T>& xout, 
                   const Tensor<T>& x, 
                   const Tensor<T>& w, 
                   const Tensor<T>& b, 
                   float* tempbuf, int d, int n) {
// w[d,n] @ x[n] + b[d] -> xout[d]
  EnhancedTensor<float> temp(tempbuf, d);

#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    temp[i] = val + b[i];
  }

  xout.requantize(temp);
}

template <typename T>
inline void broadcast_multiply(
  EnhancedTensor<T>& out, 
  const Tensor<T>& x, const Tensor<T>& y, 
  float* tempbuf, int d, int n) {
// x[d], y[d,n] -> out[d,n]
  EnhancedTensor<float> temp(tempbuf, d * n);

#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      int index = i * n + j;
      temp[index] = x[i] * y[index];
      // out[i * n + j] = x[i] * y[i * n + j];
    }
  }

  out.requantize(temp);
}

template <typename T>
inline void elementwise_multiply(EnhancedTensor<T>& result, 
                                 const Tensor2D<T>& matrix1, 
                                 const Tensor2D<T>& matrix2,
                                 float* tempbuf,
                                 int total_elements) {
  EnhancedTensor<float> temp(tempbuf, total_elements);

#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    temp[i] = matrix1[i] * matrix2[i];
  }

  result.requantize(temp);
}

template <typename T>
inline void elementwise_add(
  EnhancedTensor<T>& result, 
  const Tensor2D<T>& matrix1, 
  const Tensor2D<T>& matrix2,
  float* tempbuf, int total_elements) {
  EnhancedTensor<float> temp(tempbuf, total_elements);

#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    temp[i] = matrix1[i] + matrix2[i];
  }

  result.requantize(temp);
}

template <typename T>
inline void elementwise_multiply_and_add(
  EnhancedTensor<T>& result, 
  const Tensor2D<T>& matrix1, 
  const Tensor2D<T>& matrix2,
  const Tensor2D<T>& matrix3, 
  float* tempbuf, int total_elements) {

  EnhancedTensor<float> temp(tempbuf, total_elements);
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    temp[i] = matrix1[i] * matrix2[i] + matrix3[i];
  }

  result.requantize(temp);
}

template <typename T>
inline void elementwise_multiply_and_add(
  EnhancedTensor2D<T>& result, 
  const Tensor2D<T>& matrix1, 
  const Tensor2D<T>& matrix2,
  const Tensor2D<T>& matrix3, 
  const float* tempbuf, int total_elements) {

  EnhancedTensor2D<float> temp(tempbuf, result._layer_len, result._n_layers);

#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    temp[i] = matrix1[i] * matrix2[i] + matrix3[i];
  }

  result.requantize(temp);
}

template <typename T>
inline void outer_product(
  EnhancedTensor<T>& out, 
  const Tensor<T>& x, 
  const Tensor<T>& y, 
  float* tempbuf, int d, int n) {

  EnhancedTensor<float> temp(tempbuf, d * n);  
// x[d], y[n] -> out[d,n]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      temp[i * n + j] = x[i] * y[j];
    }
  }

  out.requantize(temp);
}

template <typename T>
inline void sum_along_last_dim(
  EnhancedTensor<T>& result, 
  const Tensor<T>& matrix, 
  float* tempbuf, int rows, int cols) {

  EnhancedTensor<float> temp(tempbuf, rows);
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    float val = 0.0f;
    for (int j = 0; j < cols; j++) {
      val += matrix[i * cols + j];
    }
    temp[i] = val;
  }

  result.requantize(temp);
}

#endif // MATH_HPP
