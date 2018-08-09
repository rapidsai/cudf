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
 Device Memory Manager implementation. Efficient allocation, deallocation and
 tracking of GPU memory.

 Author: Mark Harris
 */

#include "memory.h"

#define RMM_CHECK_CUDA(call) do { \
    cudaError_t cudaError = (call); \
    if( cudaError == cudaErrorMemoryAllocation ) { \
        return RMM_ERROR_OUT_OF_MEMORY; \
    } \
    else if( cudaError != cudaSuccess ) { \
        return RMM_ERROR_CUDA_ERROR; \
    } \
} while(0)


/// Initialize memory manager state and storage.
rmmError_t rmmInitialize()
{
    RMM_CHECK_CUDA(cudaFree(0));
    return RMM_SUCCESS;
}

/// Shutdown memory manager.
rmmError_t rmmFinalize()
{

}

/// Allocate memory and initialize a pointer to device memory.
rmmError_t rmmAlloc(void **ptr, size_t size, cudaStream_t stream)
{
	if (!ptr && !size) {
        return RMM_SUCCESS;
    }
    else if (!size) {
        ptr[0] = NULL;
        return RMM_SUCCESS;
    }

    if (!ptr) 
    	return RMM_ERROR_INVALID_ARGUMENT;

    RMM_CHECK_CUDA(cudaMallocManaged(ptr, size));

    return RMM_SUCCESS;
}

#include <cstdio>

/// Reallocate device memory block to new size and recycle any remaining memory as a new block.
rmmError_t rmmRealloc(void **ptr, size_t new_size, cudaStream_t stream)
{
	if (!ptr && !new_size) {
        return RMM_SUCCESS;
    }

    if (!ptr) 
    	return RMM_ERROR_INVALID_ARGUMENT;

	RMM_CHECK_CUDA(cudaFree(*ptr));
	RMM_CHECK_CUDA(cudaMallocManaged(ptr, new_size));

    return RMM_SUCCESS;
}

/// Release device memory and recycle the associated memory block.
rmmError_t rmmFree(void *ptr, cudaStream_t stream)
{
	RMM_CHECK_CUDA(cudaFree(ptr));
	return RMM_SUCCESS;
}

/// Get amounts of free and total memory managed by a manager associated with the stream.
rmmError_t rmmGetInfo(size_t *freeSize, size_t *totalSize, cudaStream_t stream)
{
	RMM_CHECK_CUDA(cudaMemGetInfo(freeSize, totalSize));
	return RMM_SUCCESS;
}