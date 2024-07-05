#ifndef MATH_HPP
#define MATH_HPP

#include <math.h>
// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

inline void rmsnorm(float *o, float *x, float *weight, int size) {
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
    o[j] = x[j] * weight[j] * ss;
  }
}

inline void softmax(float *x, int size) {
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

inline float softplus(float x) { return logf(1.0f + expf(x)); }

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

inline float silu(float x) { return x * sigmoid(x); }

inline void shift_matrix_left(float *matrix, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols - 1; j++) {
      matrix[i * cols + j] = matrix[i * cols + j + 1];
    }
  }
}

inline void update_last_column(float *matrix, float *x, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    matrix[i * cols + cols - 1] = x[i];
  }
}

inline void rowwise_dot_product(float *out, float *matrix, float *weights,
                                int rows, int cols) {
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

inline void matmul(float *xout, float *x, float *w, int d, int n) {
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

inline void linear(float *xout, float *x, float *w, float *b, int d, int n) {
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

inline void broadcast_multiply(float *out, float *x, float *y, int d, int n) {
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

inline void elementwise_multiply(float *result, float *matrix1, float *matrix2,
                                 int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i];
  }
}

inline void elementwise_add(float *result, float *matrix1, float *matrix2,
                            int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] + matrix2[i];
  }
}

inline void elementwise_multiply_and_add(float *result, float *matrix1,
                                         float *matrix2, float *matrix3,
                                         int total_elements) {
#pragma omp parallel for
  for (int i = 0; i < total_elements; i++) {
    result[i] = matrix1[i] * matrix2[i] + matrix3[i];
  }
}

inline void outer_product(float *out, float *x, float *y, int d, int n) {
// x[d], y[n] -> out[d,n]
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
      out[i * n + j] = x[i] * y[j];
    }
  }
}

inline void sum_along_last_dim(float *result, float *matrix, int rows,
                               int cols) {
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
