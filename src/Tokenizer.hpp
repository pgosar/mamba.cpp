/**
 * C++ implementation of GPTNeoX Tokenizer by EleutherAI and The HuggingFace Inc. team, 2022.
 * Implements HuggingFace Inc. Team 2020
*/

/**
 * need:
 *  input_ids
 *  attn_mask
 * (Thus) model_input_names ?
 *  prompt->tokens->input_ids & attn_mask
 * 
 *  batch_decode
*/

#include <vector>
#include <string>

class Tokenizer {
private:
  std::vector<unsigned long> _input_ids;

public:
  Tokenizer(std::string&);
  const std::vector<unsigned long>& inputIds();
};