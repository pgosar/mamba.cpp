#ifndef UTIL_HPP
#define UTIL_HPP

#include <args.hxx>
#include <yaml-cpp/yaml.h>

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
struct Inference {
  int max_prompt_length;
  int max_response_length;
  float temperature;
  int top_k;
  float top_p;
  float min_p;
  int repetition_penalty;
  int batch_size;
};

/*
 * parses command line arguments
 */
Opts parse_args(int argc, char *argv[]);

/*
 * parses yaml model configurations
 */
Inference parse_config(const std::string path, std::string config_name);

#endif // UTIL_HPP
