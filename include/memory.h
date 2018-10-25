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

/** ---------------------------------------------------------------------------*
 * @brief Device Memory Manager public interface. 
 * 
 * Efficient allocation, deallocation and tracking of GPU memory.
 * ---------------------------------------------------------------------------**/

#pragma once

typedef struct CUstream_st *cudaStream_t;
typedef long int offset_t; // would prefer ptrdiff_t but can't #include <stddef.h>
                           // due to CFFI limitations

/** ---------------------------------------------------------------------------*
 * @brief RMM error codes
 * ---------------------------------------------------------------------------**/
typedef enum
{
  RMM_SUCCESS = 0,            //< Success result
  RMM_ERROR_CUDA_ERROR,       //< A CUDA error occurred
  RMM_ERROR_INVALID_ARGUMENT, //< An invalid argument was passed (e.g.null pointer)
  RMM_ERROR_NOT_INITIALIZED,  //< RMM API called before rmmInitialize()
  RMM_ERROR_OUT_OF_MEMORY,    //< The memory manager was unable to allocate more memory
  RMM_ERROR_UNKNOWN,          //< An unknown error occurred
  RMM_ERROR_IO,               //< Stats output error
  N_RMM_ERROR                 //< Count of error types
} rmmError_t;

typedef enum
{
  CudaDefaultAllocation = 0,  //< Use cudaMalloc for allocation
  PoolAllocation,             //< Use pool suballocation strategy
} rmmAllocationMode_t;

typedef struct
{
  rmmAllocationMode_t allocation_mode; //< Allocation strategy to use
  size_t initial_pool_size;            //< When pool suballocation is enabled, 
                                       //< this is the initial pool size in bytes
  bool enable_logging;                 //< Enable logging memory manager events
} rmmOptions_t;

/** ---------------------------------------------------------------------------*
 * @brief Initialize memory manager state and storage.
 * 
 * @param[in] options Structure of options for the memory manager. Defaults are 
 *                    used if it is null.
 * @return rmmError_t RMM_SUCCESS or RMM_ERROR_CUDA_ERROR on any CUDA error.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmInitialize(rmmOptions_t *options);

/** ---------------------------------------------------------------------------*
 * @brief Shutdown memory manager.
 * 
 * @return rmmError_t RMM_SUCCESS, or RMM_NOT_INITIALIZED if rmmInitialize() has
 *                    not been called, or RMM_ERROR_CUDA_ERROR on any CUDA error.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmFinalize(); 

/** ---------------------------------------------------------------------------*
 * @brief Stringify RMM error code.
 * 
 * @param errcode The error returned by an RMM function
 * @return const char* The input error code in string form
 * ---------------------------------------------------------------------------**/
const char * rmmGetErrorString(rmmError_t errcode);

/** ---------------------------------------------------------------------------*
 * @brief Allocate memory and return a pointer to device memory. 
 * 
 * @param[out] ptr Returned pointer
 * @param[in] size The size in bytes of the allocated memory region
 * @param[in] stream The stream in which to synchronize this command
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called, RMM_ERROR_INVALID_ARGUMENT if ptr is
 *                    null, RMM_ERROR_OUT_OF_MEMORY if unable to allocate the
 *                    requested size, or RMM_CUDA_ERROR on any other CUDA error.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmAlloc(void **ptr, size_t size, cudaStream_t stream);

/** ---------------------------------------------------------------------------*
 * @brief Reallocate device memory block to new size and recycle any remaining
 *        memory.
 * 
 * @param[out] ptr Returned pointer
 * @param[in] new_size The size in bytes of the allocated memory region
 * @param[in] stream The stream in which to synchronize this command
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called, RMM_ERROR_INVALID_ARGUMENT if ptr is
 *                    null, RMM_ERROR_OUT_OF_MEMORY if unable to allocate the
 *                    requested size, or RMM_ERROR_CUDA_ERROR on any other CUDA
 *                    error.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmRealloc(void **ptr, size_t new_size, cudaStream_t stream);

/** ---------------------------------------------------------------------------*
 * @brief Release device memory and recycle the associated memory.
 * 
 * @param[in] ptr The pointer to free
 * @param[in] stream The stream in which to synchronize this command
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called,or RMM_ERROR_CUDA_ERROR on any CUDA
 *                    error.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmFree(void *ptr, cudaStream_t stream);

/** ---------------------------------------------------------------------------*
 * @brief Get the offset of ptr from its base allocation.
 * 
 * This offset is the difference between the address stored in ptr and the
 * base device memory allocation that it is a sub-allocation of within the pool.
 * This is useful for, e.g. IPC, where cudaIpcOpenMemHandle() returns a pointer
 * to the base  * allocation, so the user needs the offset of the sub-allocation
 * in order to correctly access its data.
 * 
 * @param[out] offset The difference between ptr and the base allocation that ptr
 *                    is sub-allocated from.
 * @param[in] ptr The ptr to find the base allocation of.
 * @param[in] stream The stream originally passed to rmmAlloc or rmmRealloc for
 *                   ptr.
 * @return rmmError_t RMM_SUCCESS if all goes well
 * ---------------------------------------------------------------------------**/
rmmError_t rmmGetAllocationOffset(offset_t *offset,
                                  void *ptr,
                                  cudaStream_t stream);

/** ---------------------------------------------------------------------------*
 * @brief Get amounts of free and total memory managed by a manager associated
 *        with the stream.
 * 
 * Returns in *free and *total, respectively, the free and total amount of
 * memory available for allocation by the device in bytes.
 * 
 * @param[out] freeSize The free memory in bytes available to the manager
 *                      associated with stream
 * @param[out] totalSize The total memory managed by the manager associated with
 *                       stream
 * @param[in] stream 
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called, or RMM_ERROR_CUDA_ERROR on any CUDA
 *                    error
 * ---------------------------------------------------------------------------**/
rmmError_t rmmGetInfo(size_t *freeSize, size_t *totalSize, cudaStream_t stream);

/** ---------------------------------------------------------------------------*
 * @brief Write the memory event stats log to specified path/filename
 * 
 * Note: will overwrite the specified file.
 * 
 * @param filename The full path and filename to write.
 * @return rmmError_t RMM_SUCCESS or RMM_ERROR_IO on output failure.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmWriteLog(const char* filename);

/** ---------------------------------------------------------------------------*
 * @brief Get the size of the CSV log string in memory.
 * 
 * @return size_t The size of the log (as a C string) in memory.
 * ---------------------------------------------------------------------------**/
size_t rmmLogSize();

/** ---------------------------------------------------------------------------*
 * @brief Get the RMM log as CSV in a C string.
 * 
 * @param[out] buffer The buffer into which to store the CSV log string.
 * @param[in] buffer_size The size allocated for buffer.
 * @return rmmError_t RMM_SUCCESS, or RMM_IO_ERROR on any failure.
 * ---------------------------------------------------------------------------**/
rmmError_t rmmGetLog(char* buffer, size_t buffer_size);