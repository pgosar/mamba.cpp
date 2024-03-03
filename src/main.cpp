#include "util.h"

int main(int argc, char *argv[]) {
  Opts opts = parse_args(argc, argv);
  std::cout << "Verbosity: " << opts.verbose << std::endl;
  return 0;
}
