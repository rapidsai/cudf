/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 Device Memory Manager public interface. Efficient allocation, deallocation and
 tracking of GPU memory.

 Author: Mark Harris
 */

#pragma once

#ifdef __cplusplus
extern "c" {
#endif

typedef struct CUstream_st *cudaStream_t;

/// Result codes returned by the memory manager
typedef enum
{
  RMM_SUCCESS = 0,
  RMM_ERROR_CUDA_ERROR,
  RMM_ERROR_INVALID_ARGUMENT,
  RMM_ERROR_NOT_INITIALIZED,
  RMM_ERROR_OUT_OF_MEMORY,
  RMM_ERROR_UNKNOWN
} rmmError_t;

/// Initialize memory manager state and storage.
rmmError_t rmmInitialize();

/// Shutdown memory manager.
rmmError_t rmmFinalize();

/// Allocate memory and initialize a pointer to device memory.
rmmError_t rmmAlloc(void **ptr, size_t size, cudaStream_t stream);

/// Reallocate device memory block to new size and recycle any remaining memory as a new block.
rmmError_t rmmRealloc(void **ptr, size_t new_size, cudaStream_t stream);

/// Release device memory and recycle the associated memory block.
rmmError_t rmmFree(void *ptr, cudaStream_t stream);

/// Get amounts of free and total memory managed by a manager associated with the stream.
rmmError_t rmmGetInfo(size_t *freeSize, size_t *totalSize, cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
