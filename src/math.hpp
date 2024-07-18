#ifndef MATH_HPP
#define MATH_HPP

#include <math.h>
#include <cfloat>

// tensors
template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <Number T>
class Tensor {
protected:
  float _scale;      //for dequantization
  float _zeropoint;  //^
  T* _data;

public:
  Tensor(float scale, float zeropoint, T* data) : 
    _scale(scale), _zeropoint(zeropoint), _data(data)
  {}

  Tensor(T* data) :
    _scale(0.0f), _zeropoint(0.0f), _data(data)
  {}

  Tensor() :
    _scale(0.0f), _zeropoint(0.0f), _data(NULL)
  {}

  //Move
  Tensor(Tensor<T>&& other) :
    _scale(std::exchange(other._scale, 0.0f)),
    _zeropoint(std::exchange(other._zeropoint, 0.0f)),
    _data(std::exchange(other._data, nullptr))
  {}
  
  Tensor<T>& operator=(Tensor<T>&& other)
  {
    _scale = std::exchange(other._scale, 0.0f);
    _zeropoint = std::exchange(other._zeropoint, 0.0f);
    _data = std::exchange(other._data, nullptr);
    return *this;
  }

  //Copy
  Tensor(Tensor<T>& other) :
    _scale(other._scale),
    _zeropoint(other._zeropoint, 0.0f),
    _data(other._data, nullptr)
  {}
  
  Tensor<T>& operator=(Tensor<T>& other)
  {
    _scale = other._scale;
    _zeropoint = other._zeropoint;
    _data = other._data;

    return *this;
  }

  ~Tensor() {
    delete _data;
  }

  //TODO maybe go down to int? 
  //will we even have tensors with such high dim anyways?
  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    return (static_cast<float>(_data[i]) - _zeropoint) / _scale;
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    return _data[i];
  }

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  inline T quantize(size_t i, float value) const {
    _data[i] = static_cast<T>(((value * _scale) + _zeropoint) + .5 * signbit(value));
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  inline T quantize(size_t i, float value) const {
    _data[i] = value;
  }

  float operator[](size_t i) const {
    return dequantize(i);
  }

  const T* data() const {
    return const_cast<T*>(data);
  }

  float scale() const {
    return _scale;
  }

  float zeropoint() const {
    return _zeropoint;
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
  size_t _len;

public:
  EnhancedTensor(float scale, float zeropoint, T* data, size_t len) :
    Tensor<T>(scale, zeropoint, data), _len(len)
  {}

  EnhancedTensor(T* data, size_t len) :
    Tensor<T>(data), _len(len)
  {}

  EnhancedTensor() :
    Tensor<T>(), _len(0)
  {}

    //Move
  EnhancedTensor(EnhancedTensor<T>&& other) :
    Tensor<T>(), _len(std::exchange(other._len, 0))
  {}
  
  EnhancedTensor<T>& operator=(EnhancedTensor<T>&& other)
  {
    Tensor<T>::operator=(other);
    _len = std::exchange(other._len, 0);
    return *this;
  }

  //Copy
  EnhancedTensor(EnhancedTensor<T>& other) :
    Tensor<T>(other), _len(other._len)
  {}
  
  EnhancedTensor<T>& operator=(EnhancedTensor<T>& other)
  {
    Tensor<T>::operator=(other);
    _len = other._len;

    return *this;
  }

  EnhancedTensor<float> dequantize() {
    //todo use templates
    if(std::is_same<T, float>::value) return this;

    float* dequantized = (float*) malloc(_len * sizeof(float));
    for(size_t i = 0; i < _len; i++) {
      dequantized[i] = (this->_data[i] - this->_zeropoint) / this->_scale;
    }

    EnhancedTensor<float> ret(1, 0, dequantized, _len);
    return ret;
  }

  EnhancedTensor<float> dequantize(float* dequantize_buffer) {
    //todo use templates
    if(std::is_same<T, float>::value) return this;

    for(size_t i = 0; i < _len; i++) {
      dequantize_buffer[i] = (this->_data[i] - this->_zeropoint) / this->_scale;
    }

    EnhancedTensor<float> ret(1, 0, dequantize_buffer, _len);
    return ret;
  }

  void requantize(EnhancedTensor<float>& t) {
    if(std::is_same<T, float>::value) return; 

    float max = FLT_MAX;
    float min = -FLT_MAX;
    for(int i = 0; i < _len; i++) {
      if(this->_data[i] > max) max = this->_data[i];
      if(this->_data[i] < min) min = this->_data[i];
    }

    float x_range = max - min;
    x_range = x_range == 0 ? 1 : x_range;

    constexpr T T_MAX = 1 << (sizeof(T)-1); 

    this->_scale = (2.0 * T_MAX) / x_range;
    this->_zeropoint = std::round(-this->_scale * min - T_MAX);

    for(int i = 0; i < _len; i++) {
      float converted = t._data[i] * this->_scale + this->_zeropoint;
      if (converted > T_MAX - 1) this->_data[i] = T_MAX-1;
      else if (converted < -T_MAX) this->_data[i] = -T_MAX;
      else this->_data[i] = static_cast<T>(converted  + .5 * signbit(converted));
    }
  }

  void set(size_t i, T d) {
    if(i > _len) {
      throw std::exception();
    }

    this->_data[i] = d;
  }
};

template <Number T>
class Tensor2D {
protected:
  float* _scales;
  float* _zeropoints;
  size_t _layer_len;
  T* _data;

public:
  Tensor2D(float* scales, float* zeropoints, size_t layer_len, T* data) :
    _scales(scales), _zeropoints(zeropoints), _layer_len(layer_len), _data(data)
  {}

  Tensor2D(T* data, size_t layer_len) :
    _scales(NULL), _zeropoints(NULL), _layer_len(layer_len), _data(data)
  {}

  Tensor2D() :
    _scales(NULL), _zeropoints(NULL), _layer_len(0), _data(NULL)
  {}

  //Move
  Tensor2D(Tensor2D<T>&& other) = default;
  Tensor2D<T>& operator=(Tensor2D<T>&&) = default;

  ~Tensor2D() {
    delete _data;
    delete _scales;
    delete _zeropoints;
  }

  //TODO maybe go down to int? 
  //will we even have tensors with such high dim anyways?
  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    float scale = _scales[i / _layer_len];
    float zeropoint = _zeropoints[i / _layer_len];
    return (static_cast<float>(_data[i]) - zeropoint) / scale;
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    return _data[i];
  }

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  inline T quantize(size_t i, float value) const {
    float scale = _scales[i / _layer_len];
    float zeropoint = _zeropoints[i / _layer_len];
    _data[i] = static_cast<T>(((value * scale) + zeropoint) + .5 * signbit(value));
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  inline T quantize(size_t i, float value) const {
    _data[i] = value;
  }

  float operator[](size_t i) const {
    return dequantize(i);
  }

  const T* data() const {
    return const_cast<T*>(data);
  }
};

template <Number T>
class EnhancedTensor2D : Tensor2D<T> {
private:
  size_t _n_layers;

public:
  EnhancedTensor2D() :
    Tensor2D<T>(), _n_layers(0)
  {}

  EnhancedTensor2D(T* data, size_t layer_len, size_t n_layers) :
    Tensor2D<T>(data, layer_len), _n_layers(n_layers)
  {
    this->_scales = static_cast<float *>(malloc(n_layers * sizeof(float)));
    this->_zeropoints = static_cast<float *>(malloc(n_layers * sizeof(float)));
  }

  EnhancedTensor2D(float* scales, float* zeropoints, T* data, size_t n_layers) :
    Tensor2D<T>(scales, zeropoints, data), _n_layers(n_layers)
  {}

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  void requantize(EnhancedTensor2D<float>& t) {
    for(int l = 0; l < _n_layers; l++) {
      float max = FLT_MAX;
      float min = -FLT_MAX;
      T* layer = this->data + l * this->_layer_len;
      for(int i = 0; i < this->_layer_len; i++) {
        if(layer[i] > max) max = layer[i];
        if(layer[i] < min) min = layer[i];
      }

      float x_range = max - min;
      x_range = x_range == 0 ? 1 : x_range;

      constexpr T T_MAX = 1 << (sizeof(T)-1); 

      this->_scales[l] = (2.0 * T_MAX) / x_range;
      this->_zeropoints[l] = std::round(-this->_scales[l] * min - T_MAX);

      float* dequantized_layer = t._data + l * this->_layer_len;
      for(int i = 0; i < this->_layer_len; i++) {
        float converted = dequantized_layer[i] * this->_scales[l] + this->_zeropoints[l];
        if (converted > T_MAX - 1) layer[i] = T_MAX-1;
        else if (converted < -T_MAX) layer[i] = -T_MAX;
        else layer[i] = static_cast<T>(converted  + .5 * signbit(converted));
      }
    }
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  void requantize(EnhancedTensor2D<float> &t) {
    //nop, data isn't quantized in the first place
  }
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

template <typename T> inline void rmsnorm(T *o, const T *x, const T *weight, int size) {
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
inline void update_last_column(T *matrix, const T *x, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    matrix[i * cols + cols - 1] = x[i];
  }
}

template <typename T>
inline void rowwise_dot_product(T *out, const T *matrix, const T *weights, int rows,
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

//TODO optimize
template <typename T> inline void matmul(T *xout, const T *x, const T *w, int d, int n) {
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
inline void linear(T *xout, const T *x, const T *w, const T *b, int d, int n) {
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
inline void broadcast_multiply(T *out, const T *x, const T *y, int d, int n) {
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
inline void elementwise_multiply(T *result, const T *matrix1, const T *matrix2,
                                 int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i];
  }
}

template <typename T>
inline void elementwise_add(T *result, const T *matrix1, const T *matrix2,
                            int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] + matrix2[i];
  }
}

template <typename T>
inline void elementwise_multiply_and_add(T *result, const T *matrix1, const T *matrix2,
                                         const T *matrix3, int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i] + matrix3[i];
  }
}

template <typename T>
inline void outer_product(T *out, const T *x, const T *y, int d, int n) {
// x[d], y[n] -> out[d,n]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      out[i * n + j] = x[i] * y[j];
    }
  }
}
template <typename T>
inline void sum_along_last_dim(T *result, const T *matrix, int rows, int cols) {
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
