#ifndef RMM_H
#define RMM_H

#include <cuda_runtime_api.h>

extern "C" {
#include "memory.h"
}

/** ---------------------------------------------------------------------------*
 * @brief Device memory alloc / realloc / free macros that pass the calling file
 * and line number to RMM for tracking.
 * ---------------------------------------------------------------------------**/
#define RMM_ALLOC(ptr, sz, stream) rmmAlloc((ptr), (sz), (stream), \
                                            __FILE__, __LINE__)
#define RMM_REALLOC(ptr, new_sz, stream) rmmRealloc((ptr), (new_sz), (stream), \
                                                    __FILE__, __LINE__)
#define RMM_FREE(ptr, stream) rmmFree((ptr), (stream), __FILE__, __LINE__)

#endif // RMM_H