#include "util.h"

Opts parse_args(int argc, char *argv[]) {
  Opts opts;
  args::ArgumentParser parser("fast inference for mamba", "Authors");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Flag verbose(parser, "verbose", "Verbose output", {'v', "verbose"});
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    std::exit(0);
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  return opts;
}
