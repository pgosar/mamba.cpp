#ifndef UTIL_HPP
#define UTIL_HPP

#include <args.hxx>
#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

/*
 * global program options
 */
struct Opts {
  bool verbose;
  std::string prompt;
  std::string configuration;
};

/*
 * model configuration options
 */
struct Config {
  int max_prompt_length;
  int max_response_length;
  float temperature;
  int top_k;
  float top_p;
  float min_p;
  float repetition_penalty;
  int batch_size;
};

struct InferenceParams {
  int max_seqlen;
  int max_batch_size;
  int seqlen_offset;
  int batch_size_offset;
  //TODO key-value memory
  torch::Tensor lengths_per_sample;

  InferenceParams(int max_seqlen, int max_batch_size) {
    this->max_seqlen = max_seqlen;
    this->max_batch_size = max_batch_size;
    this->seqlen_offset = 0;
    if(this->lengths_per_sample.numel() != 0)
      this->lengths_per_sample = torch::zero_(this->lengths_per_sample);
  }
};

/*
 * parses command line arguments
 */
Opts parse_args(int argc, char *argv[]);

/*
 * parses yaml model configurations
 */
Config parse_config(const std::string path, std::string config_name);

#endif // UTIL_HPP
