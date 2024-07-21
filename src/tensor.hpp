#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <concepts>

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

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  float operator[](size_t i) const {
    return dequantize(i);
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  float& operator[](size_t i) {
    return _data[i];
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

  T get(size_t i) const {
    return this->_data[i];
  }

  T set(size_t i, T d) {
    this->_data[i] = d;
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
      max = std::max(t.data[i], max);
      min = std::min(t.data[i], min);
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

  //Used to prevent data from being freed on destruction, if using a shared buffer
  void detach() {
    this->_data = nullptr;
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

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  float operator[](size_t i) const {
    return dequantize(i);
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  float& operator[](size_t i) {
    return _data[i];
  }

  const T* data() const {
    return const_cast<T*>(data);
  }

  T get(size_t i) const {
    return this->_data[i];
  }

  T set(size_t i, T d) {
    this->_data[i] = d;
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
      float* dequantized_layer = t._data + l * this->_layer_len;
      for(int i = 0; i < this->_layer_len; i++) {
        max = std::max(dequantized_layer[i], max);
        min = std::min(dequantized_layer[i], min);
      }

      float x_range = max - min;
      x_range = x_range == 0 ? 1 : x_range;

      constexpr T T_MAX = 1 << (sizeof(T)-1); 

      this->_scales[l] = (2.0 * T_MAX) / x_range;
      this->_zeropoints[l] = std::round(-this->_scales[l] * min - T_MAX);

      T* layer = this->data + l * this->_layer_len;
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

#endif