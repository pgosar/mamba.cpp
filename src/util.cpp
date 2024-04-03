#include "util.hpp"

Opts parse_args(int argc, char *argv[]) {
  Opts opts;
  args::ArgumentParser parser("fast inference for mamba", "Authors");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Flag verbose(parser, "verbose", "Verbose output", {'v', "verbose"});
  args::ValueFlag<std::string> config(parser, "config", "Configuration file",
                                      {'c', "config"});
  args::ValueFlag<std::string> prompt(parser, "prompt", "Prompt",
                                      {'p', "prompt"});

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &e) {
    std::cout << parser;
    std::exit(0);
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  } catch (args::ValidationError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  if (!config || !prompt) {
    std::cerr << "Missing required arguments.\n" << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  opts.verbose = args::get(verbose);
  opts.configuration = args::get(config);
  opts.prompt = args::get(prompt);

  return opts;
}

Config parse_config(const std::string config_file,
                       const std::string config_name) {
  YAML::Node config = YAML::LoadFile(config_file);
  if (!config[config_name]) {
    std::cerr << "Configuration '" << config_name << "' not found in YAML file."
              << std::endl;
    exit(1);
  }

  Config inf;
  inf.max_prompt_length = config[config_name]["max_prompt_length"].as<int>();
  inf.max_response_length =
      config[config_name]["max_response_length"].as<int>();
  inf.temperature = config[config_name]["temperature"].as<float>();
  inf.top_k = config[config_name]["top_k"].as<int>();
  inf.top_p = config[config_name]["top_p"].as<float>();
  inf.min_p = config[config_name]["min_p"].as<float>();
  inf.repetition_penalty = config[config_name]["repetition_penalty"].as<int>();
  inf.batch_size = config[config_name]["batch_size"].as<int>();

  return inf;
}
