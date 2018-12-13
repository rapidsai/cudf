#ifndef RMM_H
#define RMM_H

#include <cuda_runtime_api.h>
#include "memory.hpp"

extern "C" {
#include "memory.h"
}

/** ---------------------------------------------------------------------------*
 * @brief Device memory alloc / realloc / free macros that pass the calling file
 * and line number to RMM for tracking.
 * ---------------------------------------------------------------------------**/
#define RMM_ALLOC(ptr, sz, stream) \
  rmm::alloc((ptr), (sz), (stream), __FILE__, __LINE__)

#define RMM_REALLOC(ptr, new_sz, stream) \
  rmm::realloc((ptr), (new_sz), (stream), __FILE__, __LINE__)

#define RMM_FREE(ptr, stream) rmm::free((ptr), (stream), __FILE__, __LINE__)

#endif // RMM_H
