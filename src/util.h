#ifndef UTIL_HPP
#define UTIL_HPP

#include <args.hxx>

/*
 * global program options
 */
struct Opts {
  bool verbose;
};

/**
 * parses command line arguments
 */
Opts parse_args(int argc, char *argv[]);

#endif // UTIL_HPP
