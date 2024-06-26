#include <cstddef>
#include <unistd.h>

inline size_t get_total_memory() {
  return sysconf(_SC_PAGESIZE) * sysconf(_SC_PHYS_PAGES);
}

inline size_t get_model_size(char *path) {

}
