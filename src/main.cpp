#include "util.hpp"
#if defined(__arm__) || defined(__aarch64__)
#include "sse2neon/sse2neon.h"
#endif

int main(int argc, char *argv[]) {
  Opts opts = parse_args(argc, argv);
  const std::string CONFIG_FILE = "../mamba_configuration.yaml";
  Inference inf = parse_config(CONFIG_FILE, opts.configuration);
  return 0;
}
