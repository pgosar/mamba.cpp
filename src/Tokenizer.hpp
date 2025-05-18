#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "math.hpp"

#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Tokenizer {
private:
  struct TokenIndex {
    const char *str;
    int id;
  };

  char **vocab = nullptr;
  TokenIndex *sorted_vocab = nullptr;
  int vocab_size = 0;
  unsigned int max_token_length = 0;
  unsigned char byte_pieces[512]; // stores all single-byte strings

  static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
  }

  static int str_lookup(const char *str, TokenIndex *sorted_vocab,
                        int vocab_size) {
    TokenIndex tok = {str, 0};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size,
                                            sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
  }

public:
  static constexpr int BOS = 0;
  static constexpr int EOS = 0;

  explicit Tokenizer(const std::string &tokenizer_path) {
    // Initialize byte_pieces array
    for (int i = 0; i < 256; i++) {
      byte_pieces[i * 2] = static_cast<unsigned char>(i);
      byte_pieces[i * 2 + 1] = '\0';
    }

    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("couldn't load " + tokenizer_path);
    }

    // Read vocab_size
    if (!file.read(reinterpret_cast<char *>(&vocab_size), sizeof(int))) {
      throw std::runtime_error("failed read");
    }

    // Read max_token_length
    if (!file.read(reinterpret_cast<char *>(&max_token_length), sizeof(int))) {
      throw std::runtime_error("failed read");
    }

    // Malloc space for the vocab
    vocab = static_cast<char **>(malloc(vocab_size * sizeof(char *)));
    if (!vocab) {
      throw std::runtime_error("malloc failed");
    }

    // Read vocab
    for (int i = 0; i < vocab_size; i++) {
      int len;
      if (!file.read(reinterpret_cast<char *>(&len), sizeof(int))) {
        throw std::runtime_error("failed read");
      }

      vocab[i] = static_cast<char *>(malloc(len + 1));
      if (!file.read(vocab[i], len)) {
        throw std::runtime_error("failed read");
      }
      vocab[i][len] = '\0';
    }
  }

  ~Tokenizer() {
    for (int i = 0; i < vocab_size; i++) {
      free(vocab[i]);
    }
    free(vocab);
    free(sorted_vocab);
  }

  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;

  // Add move operations
  Tokenizer(Tokenizer &&) noexcept = default;
  Tokenizer &operator=(Tokenizer &&) noexcept = default;

  char *decode(int prev_token, int token) const {
    const char *piece = vocab[token];
    // Discard initial space if prev_token was EOS
    if (prev_token == EOS && piece[0] == ' ') {
      piece++;
    }
    // Handle raw bytes
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
      piece = reinterpret_cast<const char *>(byte_pieces + byte_val * 2);
    }
    return const_cast<char *>(piece);
  }

  void encode(const std::string &text, int8_t add_bos, int8_t add_eos,
              int *tokens, int *n_tokens) {
    if (sorted_vocab == nullptr) {
      sorted_vocab =
          static_cast<TokenIndex *>(malloc(vocab_size * sizeof(TokenIndex)));
      for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i];
        sorted_vocab[i].id = i;
      }
      qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    size_t buffer_size = max_token_length * 2 + 1 + 2;
    char *str_buffer = static_cast<char *>(malloc(buffer_size));
    size_t str_len = 0;

    *n_tokens = 0;

    if (add_bos) {
      tokens[(*n_tokens)++] = BOS;
    }

    constexpr int add_dummy_prefix = 0;
    if (add_dummy_prefix && !text.empty()) {
      int dummy_prefix = str_lookup(" ", sorted_vocab, vocab_size);
      tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (const char c : text) {
      if ((c & 0xC0) != 0x80) {
        str_len = 0;
      }

      str_buffer[str_len++] = c;
      str_buffer[str_len] = '\0';

      if ((str_len < 4) && (str_len < text.length()) &&
          ((text[str_len] & 0xC0) == 0x80)) {
        continue;
      }

      int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
      if (id != -1) {
        tokens[(*n_tokens)++] = id;
      } else {
        for (size_t i = 0; i < str_len; i++) {
          tokens[(*n_tokens)++] = static_cast<unsigned char>(str_buffer[i]) + 3;
        }
      }
      str_len = 0;
    }

    while (true) {
      int best_id = -1;
      int best_idx = -1;

      for (int i = 0; i < (*n_tokens - 1); i++) {
        snprintf(str_buffer, buffer_size, "%s%s", vocab[tokens[i]],
                 vocab[tokens[i + 1]]);
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
        if (id != -1) {
          best_id = id;
          best_idx = i;
          break;
        }
      }

      if (best_idx == -1) {
        break;
      }

      tokens[best_idx] = best_id;
      for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
        tokens[i] = tokens[i + 1];
      }
      (*n_tokens)--;
    }

    if (add_eos) {
      tokens[(*n_tokens)++] = EOS;
    }

    free(str_buffer);
  }
};

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token

class Sampler {
private:
  struct ProbIndex {
    float prob;
    int index;

    bool operator<(const ProbIndex &other) const noexcept {
      return prob > other.prob; // Reverse order for descending sort
    }
  };

  const int vocab_size;
  const float temperature;
  const float topp;
  std::vector<ProbIndex> probindex;

  inline static thread_local std::mt19937 gen{std::random_device{}()};
  inline static thread_local std::uniform_real_distribution<float> dis{0.0f,
                                                                       1.0f};

  template <typename T>
  static int sample_argmax(const T *probabilities, int n) noexcept {
    return static_cast<int>(std::distance(
        probabilities, std::max_element(probabilities, probabilities + n)));
  }

  static int sample_mult(const float *probabilities, int n,
                         float coin) noexcept {
    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
      cdf += probabilities[i];
      if (coin < cdf) {
        return i;
      }
    }
    return n - 1;
  }

public:
  Sampler(int vocab_size, float temperature, float topp)
      : vocab_size(vocab_size), temperature(temperature), topp(topp),
        probindex(vocab_size) {}

  Sampler(const Sampler &) = delete;
  Sampler &operator=(const Sampler &) = delete;

  template <typename T>
  int sample(EnhancedTensor<T> &logits, float *templogits) {
    if (temperature == 0.0f) {
      return sample_argmax(logits.data(), vocab_size);
    }

    logits.dequantize(templogits);
    for (int q = 0; q < vocab_size; ++q) {
      templogits[q] /= temperature;
    }
    softmax(templogits, vocab_size);

    const float coin = dis(gen);
    if (topp <= 0.0f || topp >= 1.0f) {
      const int result = sample_mult(templogits, vocab_size, coin);
      logits.requantize(templogits);
      return result;
    }

    // Top-p sampling
    const float cutoff = (1.0f - topp) / (vocab_size - 1);
    int n0 = 0;
    for (int i = 0; i < vocab_size; ++i) {
      if (templogits[i] >= cutoff) {
        probindex[n0].index = i;
        probindex[n0].prob = templogits[i];
        ++n0;
      }
    }

    std::sort(probindex.begin(), probindex.begin() + n0);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; ++i) {
      cumulative_prob += probindex[i].prob;
      if (cumulative_prob > topp) {
        last_idx = i;
        break;
      }
    }

    const float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; ++i) {
      cdf += probindex[i].prob;
      if (r < cdf) {
        const int result = probindex[i].index;
        logits.requantize(templogits);
        return result;
      }
    }

    const int result = probindex[last_idx].index;
    logits.requantize(templogits);
    return result;
  }
};

inline void safe_printf(const char *piece) {
  if (!piece || piece[0] == '\0')
    return;
  if (piece[1] == '\0') {
    const unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val)))
      return;
  }
  printf("%s", piece);
}

#endif // TOKENIZER
