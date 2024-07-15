#ifndef MATH_HPP
#define MATH_HPP

#include <math.h>

// tensors
template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <Number T>
class Tensor {
private:
  const float _scale;      //for dequantization
  const float _zeropoint;  //^
  T* _data;

public:
  Tensor(float scale, float zeropoint, T* data) : 
    _scale(scale), _zeropoint(zeropoint), _data(data)
  {}

  ~Tensor() {
    delete _data;
  }

  //TODO maybe go down to int? 
  //will we even have tensors with such high dim anyways?
  template<typename D=T,
  std::enable_if_t<!std::is_same<D,float>::value>>
  inline float dequantize(size_t i) const {
    return (static_cast<float>(_data[i]) - _zeropoint) / _scale;
  }

    template<typename D=T,
  std::enable_if_t<std::is_same<D,float>::value>>
  inline float dequantize(size_t i) const {
    return _data[i];
  }

  const T* data() const {
    return const_cast<T*>(data);
  }
};

/*
* A Tensor that stores its length and provides tensor-wide operations
* Very slightly higher memory footprint than base Tensor 
* (24 vs 16 bytes, usually negligible)
*
* TODO parallel implementation
*/
template <Number T>
class EnhancedTensor : Tensor<T> {
private:
  const size_t _len;

public:
  EnhancedTensor(float scale, float zeropoint, T* data, size_t len) :
    Tensor<T>(scale, zeropoint, data), _len(len)
  {}

  EnhancedTensor<float> dequantize() {
    //risky but far more efficient than copying here
    if(std::is_same<T, float>::value) return this; 

    float* dequantized = (float*) malloc(_len * sizeof(float));
    for(size_t i = 0; i < _len; i++) {
      dequantized[i] = (this->_data[i] - this->_zeropoint) / this->_scale;
    }

    EnhancedTensor<float> ret(1, 0, dequantized, _len);
    return ret;
  }
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

template <typename T> inline void rmsnorm(T *o, T *x, T *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = x[j] * weight[j] * ss; //Todo new macro to dequantize
  }
}

template <typename T> inline void softmax(T *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

template <typename T> inline T softplus(T x) { return logf(1.0f + expf(x)); }

template <typename T> inline T sigmoid(T x) { return 1.0f / (1.0f + expf(-x)); }

template <typename T> inline T silu(T x) { return x * sigmoid(x); }

template <typename T>
inline void shift_matrix_left(T *matrix, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols - 1; j++) {
      matrix[i * cols + j] = matrix[i * cols + j + 1];
    }
  }
}

template <typename T>
inline void update_last_column(T *matrix, T *x, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    matrix[i * cols + cols - 1] = x[i];
  }
}

template <typename T>
inline void rowwise_dot_product(T *out, T *matrix, T *weights, int rows,
                                int cols) {
// matrix[rows,cols], weights[cols] -> out[rows]
// this is a dot product of each row of the matrix with the weights
// i.e. out[i] = matrix[i,:] @ weights
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    float val = 0.0f;
    for (int j = 0; j < cols; j++) {
      val += matrix[i * cols + j] * weights[j];
    }
    out[i] = val;
  }
}

template <typename T> inline void matmul(T *xout, T *x, T *w, int d, int n) {
// w[d,n] @ x[n] -> xout[d]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

template <typename T>
inline void linear(T *xout, T *x, T *w, T *b, int d, int n) {
// w[d,n] @ x[n] + b[d] -> xout[d]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val + b[i];
  }
}

template <typename T>
inline void broadcast_multiply(T *out, T *x, T *y, int d, int n) {
// x[d], y[d,n] -> out[d,n]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      int index = i * n + j;
      out[index] = x[i] * y[index];
      // out[i * n + j] = x[i] * y[i * n + j];
    }
  }
}

template <typename T>
inline void elementwise_multiply(T *result, T *matrix1, T *matrix2,
                                 int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i];
  }
}

template <typename T>
inline void elementwise_add(T *result, T *matrix1, T *matrix2,
                            int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] + matrix2[i];
  }
}

template <typename T>
inline void elementwise_multiply_and_add(T *result, T *matrix1, T *matrix2,
                                         T *matrix3, int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i] + matrix3[i];
  }
}

template <typename T>
inline void outer_product(T *out, T *x, T *y, int d, int n) {
// x[d], y[n] -> out[d,n]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      out[i * n + j] = x[i] * y[j];
    }
  }
}
template <typename T>
inline void sum_along_last_dim(T *result, T *matrix, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    float val = 0.0f;
    for (int j = 0; j < cols; j++) {
      val += matrix[i * cols + j];
    }
    result[i] = val;
  }
}

#endif // MATH_HPP
