#include "util.hpp"
#if defined(__arm__) || defined(__aarch64__)
#include "sse2neon/sse2neon.h"
#endif

#include "unistd.h"

#include "SSModel.hpp"
#include "Tokenizer.hpp"

int main(int argc, char *argv[]) {
  Opts opts = parse_args(argc, argv);
  const std::string CONFIG_FILE = "../mamba_configuration.yaml";
  Config cfg = parse_config(CONFIG_FILE, opts.configuration);

  //First assume mamba
  const std::string MODEL_FILE = "../models/hub/models--state-spaces--mamba-130m-hf/snapshots/1e76775f628fbf1350fbe4dbb3d971ba64af25a1/model.safetensors";
  // SSMModel model = 
  Tokenizer tokenizer = Tokenizer(opts.prompt);
  SSModel model = SSModel::from_pretrained(MODEL_FILE);

  // std::vector<unsigned long> input_ids = tokenizer.inputIds();
  torch::Tensor input_tensor = torch::tensor(
    {8262, 849, 403, 368, 2509, 32},  // Hey how are you doing?
    {torch::kInt64});
  model.generate(input_tensor, cfg);
  /**
   * TODO basic translation
   * 1. tokenize
   * 2. extract model
   * 3. generate attention mask
   * 4. generate model
   * 5. quantize
   * 
   * 6. dequantize
   * 7. inference
  */

  

  return 0;
}
