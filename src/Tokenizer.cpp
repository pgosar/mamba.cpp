#include "Tokenizer.hpp"

Tokenizer::Tokenizer(std::string& prompt) {
  _input_ids = std::vector<unsigned long>(100);
}

const std::vector<unsigned long>& Tokenizer::inputIds() {
  return _input_ids;
}
