#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cfloat>
#include <concepts>
#include <cmath>
#include <algorithm>

// tensors
template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <Number T>
class TensorBase {

};

template <Number T>
class Tensor : TensorBase<T>{
protected:
  float _scale;      //for dequantization
  float _zeropoint;  //^
  T* _data;

public:
  Tensor(const float scale, const float zeropoint, T* data) :
    _scale(scale), _zeropoint(zeropoint), _data(data)
  {}

  explicit Tensor(T* const data) :
    _scale(0.0f), _zeropoint(0.0f), _data(data)
  {}

  Tensor() :
    _scale(0.0f), _zeropoint(0.0f), _data(nullptr)
  {}

  //Move
  Tensor(Tensor<T>&& other) noexcept :
    _scale(std::exchange(other._scale, 0.0f)),
    _zeropoint(std::exchange(other._zeropoint, 0.0f)),
    _data(std::exchange(other._data, nullptr))
  {}
  
  Tensor<T>& operator=(Tensor<T>&& other) noexcept {
    _scale = std::exchange(other._scale, 0.0f);
    _zeropoint = std::exchange(other._zeropoint, 0.0f);
    _data = std::exchange(other._data, nullptr);
    return *this;
  }

  //Copy
  Tensor(const Tensor<T>& other) :
    _scale(other._scale),
    _zeropoint(other._zeropoint),
    _data(other._data)
  {}
  
  Tensor<T>& operator=(Tensor<T> const& other)
  {
    if(this != &other){
      _scale = other._scale;
      _zeropoint = other._zeropoint;
      _data = other._data;
    }

    return *this;
  }

  ~Tensor() {
    delete _data;
  }

  //TODO maybe go down to int? 
  //will we even have tensors with such high dim anyways?
  template<typename X=T> [[nodiscard]]
  typename std::enable_if_t<!std::is_same_v<X,float>, float>
  dequantize(int i) const {
    return (static_cast<float>(_data[i]) - _zeropoint) / _scale;
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  [[nodiscard]] float dequantize(size_t i) const {
    return _data[i];
  }

  template<typename X=T,
  std::enable_if_t<!std::is_same_v<X,float>>>
  inline void quantize(size_t i, float value) const {
    _data[i] = static_cast<T>(((value * _scale) + _zeropoint) + .5 * signbit(value));
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  inline void quantize(size_t i, float value) const {
    _data[i] = value;
  }

  template<typename X=T>
  typename std::enable_if_t<!std::is_same_v<X,float>, float>
  operator[](size_t i) const {
    return dequantize(i);
  }

  template<typename X=T,
  typename = std::enable_if_t<std::is_same_v<X,float>>>
  const float& operator[](size_t i) const {
    return _data[i];
  }

  template<typename X=T,
  typename = std::enable_if_t<std::is_same_v<X,float>>>
  float& operator[](size_t i) {
    return _data[i];
  }

  [[nodiscard]] const T* data() const {
    return const_cast<T*>(_data);
  }

  [[nodiscard]] T* data() {
    return _data;
  }

  [[nodiscard]] float scale() const {
    return _scale;
  }

  [[nodiscard]] float zeropoint() const {
    return _zeropoint;
  }

  [[nodiscard]] T get(size_t i) const {
    return this->_data[i];
  }

  void set(size_t i, T d) {
    this->_data[i] = d;
  }

  Tensor<T> operator+(const size_t off) {
    return Tensor<T>(this->_scale, this->_zeropoint,
							this->_data + off);
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
class EnhancedTensor : public Tensor<T> {
private:
  size_t _len;

public:
  EnhancedTensor(float scale, float zeropoint, T* data, const size_t len) :
    Tensor<T>(scale, zeropoint, data), _len(len)
  {}

  EnhancedTensor(T* const data, size_t len) :
    Tensor<T>(data), _len(len)
  {}

  EnhancedTensor() :
    Tensor<T>(), _len(0)
  {}

    //Move
  EnhancedTensor(EnhancedTensor<T>&& other) noexcept :
    Tensor<T>(), _len(std::exchange(other._len, 0))
  {}
  
  EnhancedTensor<T>& operator=(EnhancedTensor<T>&& other) noexcept
  {
    Tensor<T>::operator=(other);
    _len = std::exchange(other._len, 0);
    return *this;
  }

  //Copy
  EnhancedTensor(EnhancedTensor<T>& other) :
    Tensor<T>(other), _len(other._len)
  {}
  
  EnhancedTensor<T>& operator=(EnhancedTensor<T> const& other)
  {
    if(this==&other) return *this;

    Tensor<T>::operator=(other);
    _len = other._len;

    return *this;
  }

  template<typename X=T>
  typename std::enable_if_t<!std::is_same_v<X,float>, EnhancedTensor<float>>
  dequantize() {
    //todo use templates
    auto dequantized = static_cast<float*>(malloc(_len * sizeof(float)));
    for(size_t i = 0; i < _len; i++) {
      dequantized[i] = (this->_data[i] - this->_zeropoint) / this->_scale;
    }

    EnhancedTensor<float> ret(1, 0, dequantized, _len);
    return ret;
  }

  template<typename X=T>
  typename std::enable_if_t<std::is_same_v<X,float>, EnhancedTensor<float>>
  dequantize() {
    return *this;
  }

  template<typename X=T>
  typename std::enable_if_t<!std::is_same_v<X,float>, EnhancedTensor<float>>
  dequantize(float* dequantize_buffer) {
    for(size_t i = 0; i < _len; i++) {
      dequantize_buffer[i] = (this->_data[i] - this->_zeropoint) / this->_scale;
    }

    EnhancedTensor<float> ret(1, 0, dequantize_buffer, _len);
    return ret;
  }

  template<typename X=T>
  typename std::enable_if_t<std::is_same_v<X,float>, EnhancedTensor<float>>
  dequantize(float* dequantize_buffer) {
    return *this;
  }

  template<typename X=T>
  typename std::enable_if_t<!std::is_same_v<X,float>>
  requantize(EnhancedTensor<float>& t) {
    float max = FLT_MAX;
    float min = -FLT_MAX;
    for(int i = 0; i < _len; i++) {
      max = std::max(t._data[i], max);
      min = std::min(t._data[i], min);
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

    t.detach();  //todo could use std::shared_ptr instead
  }

  template<typename X=T>
  typename std::enable_if_t<std::is_same_v<X,float>>
  requantize(EnhancedTensor<float>& t) {
    memcpy(this->_data, t._data, _len);
    t.detach();
  }

  template<typename X=T>
  typename std::enable_if_t<!std::is_same_v<X,float>>
  requantize(const float* t) {
    float max = FLT_MAX;
    float min = -FLT_MAX;
    for(int i = 0; i < _len; i++) {
      max = std::max(t[i], max);
      min = std::min(t[i], min);
    }

    float x_range = max - min;
    x_range = x_range == 0 ? 1 : x_range;

    constexpr T T_MAX = 1 << (sizeof(T)-1);

    this->_scale = (2.0 * T_MAX) / x_range;
    this->_zeropoint = std::round(-this->_scale * min - T_MAX);

    for(int i = 0; i < _len; i++) {
      float converted = t[i] * this->_scale + this->_zeropoint;
      if (converted > T_MAX - 1) this->_data[i] = T_MAX-1;
      else if (converted < -T_MAX) this->_data[i] = -T_MAX;
      else this->_data[i] = static_cast<T>(converted  + .5 * signbit(converted));
    }
  }

  template<typename X=T>
  typename std::enable_if_t<std::is_same_v<X,float>>
  requantize(const float* t) {
    memcpy(this->_data, t, _len);
  }

  //Used to prevent data from being freed on destruction, if using a shared buffer
  void detach() {
    this->_data = nullptr;
  }

  [[nodiscard]] size_t len() const {
    return _len;
  }

  EnhancedTensor<T> operator+(const size_t off) {
    return EnhancedTensor<T>(this->_scale, this->_zeropoint,
							this->_data + off, _len - off);
  }

  EnhancedTensor<T> subset(const size_t off, const size_t len) {
    return EnhancedTensor<T>(this->_scale, this->_zeropoint,
                                                        this->_data + off, len);
  }
};


template <Number T>
class SubTensor {
private:
  float* _scale_ptr;
  float* _zeropoint_ptr;
  T* _data;
  size_t _len;

public:
  SubTensor(float* scale_ptr, float* zeropoint_ptr, T* data, size_t len) :
    _scale_ptr(scale_ptr), _zeropoint_ptr(zeropoint_ptr), _data(data), _len(len)
  {}

  SubTensor(EnhancedTensor<T>& source, size_t offset) :
    _scale_ptr(&source._scale),
    _zeropoint_ptr(&source._zeropoint),
    _data(source._data + offset),
    _len(source._len)
  {}

  //Move
  SubTensor(SubTensor<T>&& other) noexcept :
    _scale_ptr(std::exchange(other._scale_ptr, nullptr)),
    _zeropoint_ptr(std::exchange(other._zeropoint_ptr, nullptr)),
    _data(std::exchange(other._data, nullptr)),
    _len(std::exchange(other._len, 0))
  {}

  SubTensor<T>& operator=(SubTensor<T>&& other) noexcept
  {
    _scale_ptr = std::exchange(other._scale_ptr, nullptr);
    _zeropoint_ptr = std::exchange(other._zeropoint_ptr, nullptr);
    _data = std::exchange(other._data, nullptr);
    _len = std::exchange(other._len, 0);
    return *this;
  }

  //Copy (Shallow, this tensor is effectively a 1D reference into a 2D tensor)
  SubTensor(SubTensor<T> const& other) :
    _scale_ptr(other._scale_ptr),
    _zeropoint_ptr(other._zeropoint_ptr),
    _data(other._data),
    _len(other._len)
  {}

  SubTensor<T>& operator=(SubTensor<T> const& other)
  {
    _scale_ptr = other._scale_ptr;
    _zeropoint_ptr = other._zeropoint_ptr;
    _data = other._data;
    _len = other._len;

    return *this;
  }

  //No destructor other than default; this tensor does not own any of its pointers
  ~SubTensor() = default;

  template<typename X=T,
  std::enable_if_t<!std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    return (static_cast<float>(_data[i]) - *_zeropoint_ptr) / *_scale_ptr;
  }

  template<typename X=T,
  std::enable_if_t<std::is_same<X,float>::value>>
  inline float dequantize(size_t i) const {
    return _data[i];
  }

    template<typename X=T,
  std::enable_if_t<!std::is_same_v<X,float>>>
  float operator[](size_t i) const {
    return dequantize<T>(i);
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  float& operator[](size_t i) {
    return _data[i];
  }

  const T* data() const {
    return const_cast<T*>(data);
  }

  T get(size_t i) const {
    return this->_data[i];
  }

  void set(size_t i, T d) {
    this->_data[i] = d;
  }

  template<typename X=T,
  std::enable_if_t<!std::is_same_v<X,float>>>
  void requantize(EnhancedTensor<float>& t) {
    float max = FLT_MAX;
    float min = -FLT_MAX;
    T* other_data = t.data();
    for(int i = 0; i < _len; i++) {
      max = std::max(other_data[i], max);
      min = std::min(other_data[i], min);
    }

    float x_range = max - min;
    x_range = x_range == 0 ? 1 : x_range;

    constexpr T T_MAX = 1 << (sizeof(T)-1);

    float scale = (2.0 * T_MAX) / x_range;
    float zeropoint = std::round(-scale * min - T_MAX);

    for(int i = 0; i < _len; i++) {
      float converted = other_data[i] * scale + zeropoint;
      if (converted > T_MAX - 1) _data[i] = T_MAX-1;
      else if (converted < -T_MAX) _data[i] = -T_MAX;
      else _data[i] = static_cast<T>(converted  + .5 * signbit(converted));
    }

    *_scale_ptr = scale;
    *_zeropoint_ptr = zeropoint;

    t.detach();
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  void requantize(EnhancedTensor<float>& t) {
    memcpy(_data, t.data(), _len);
    t.detach();
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
  Tensor2D(float* scales, float* zeropoints, const size_t layer_len, T* data) :
    _scales(scales), _zeropoints(zeropoints), _layer_len(layer_len), _data(data)
  {}

  Tensor2D(T* data, const size_t layer_len) :
    _scales(nullptr), _zeropoints(nullptr), _layer_len(layer_len), _data(data)
  {}

  Tensor2D() :
    _scales(nullptr), _zeropoints(nullptr), _layer_len(0), _data(nullptr)
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
  std::enable_if_t<!std::is_same_v<X,float>>>
  [[nodiscard]] float dequantize(size_t i) const {
    float scale = _scales[i / _layer_len];
    float zeropoint = _zeropoints[i / _layer_len];
    return (static_cast<float>(_data[i]) - zeropoint) / scale;
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  [[nodiscard]] float dequantize(size_t i) const {
    return _data[i];
  }

  template<typename X=T,
  std::enable_if_t<!std::is_same_v<X,float>>>
  void quantize(size_t i, const float value) const {
    float scale = _scales[i / _layer_len];
    float zeropoint = _zeropoints[i / _layer_len];
    _data[i] = static_cast<T>(((value * scale) + zeropoint) + .5 * signbit(value));
  }

  template<typename X=T,
  std::enable_if_t<std::is_same_v<X,float>>>
  void quantize(size_t i, const float value) const {
    _data[i] = value;
  }

  template<typename X=T,
  typename = std::enable_if_t<!std::is_same_v<X,float>>>
  float operator[](size_t i) const {
    return dequantize<T>(i);
  }

  template<typename X=T,
  typename = std::enable_if_t<std::is_same_v<X,float>>>
  float& operator[](size_t i) {
    return _data[i];
  }

  [[nodiscard]] const T* data() const {
    return const_cast<T*>(data);
  }

  [[nodiscard]] T get(size_t i) const {
    return this->_data[i];
  }

  void set(size_t i, T d) {
    this->_data[i] = d;
  }

  EnhancedTensor<T> layer(const size_t l) {
    return EnhancedTensor<T>(_scales[l], _zeropoints[l], _data + l * _layer_len, _layer_len);
  }

  void update_layer(const size_t l, const EnhancedTensor<T>& layer) {
    _scales[l] = layer.scale();
    _zeropoints[l] = layer.zeropoint();
  }

  SubTensor<T> layer_ref(const size_t l) {
    return SubTensor<T>(_scales + l, _zeropoints + l, _data + l * _layer_len, _layer_len);
  }
};

template <Number T>
class EnhancedTensor2D : public Tensor2D<T> {
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

  EnhancedTensor2D(const float* scales, const float* zeropoints, T* data, size_t n_layers) :
    Tensor2D<T>(scales, zeropoints, data), _n_layers(n_layers)
  {}

  template<typename X=T,
  std::enable_if_t<!std::is_same_v<X,float>>>
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
  std::enable_if_t<std::is_same_v<X,float>>>
  void requantize(EnhancedTensor2D<float> &t) {
    //nop, data isn't quantized in the first place
  }
};

#endif