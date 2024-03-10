#include "util.hpp"

int main(int argc, char *argv[]) {
  Opts opts = parse_args(argc, argv);
  const std::string CONFIG_FILE = "../mamba_configuration.yaml";
  Inference inf = parse_config(CONFIG_FILE, opts.configuration);
  return 0;
}
