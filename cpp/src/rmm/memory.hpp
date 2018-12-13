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
 * @brief Device Memory Manager public C++ interface.
 *
 * Efficient allocation, deallocation and tracking of GPU memory.
 * --------------------------------------------------------------------------**/

#ifndef MEMORY_HPP
#define MEMORY_HPP
extern "C" {
#include "memory.h"
}
#include "memory_manager.h"

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper to check for error in RMM API calls.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK(call)                     \
  do {                                      \
    rmmError_t error = (call);              \
    if (error != RMM_SUCCESS) return error; \
  } while (0)

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for RMM API calls to return appropriate RMM errors.
 * ---------------------------------------------------------------------------**/
#define RMM_CHECK_CUDA(call)                    \
  do {                                          \
    cudaError_t cudaError = (call);             \
    if (cudaError == cudaErrorMemoryAllocation) \
      return RMM_ERROR_OUT_OF_MEMORY;           \
    else if (cudaError != cudaSuccess)          \
      return RMM_ERROR_CUDA_ERROR;              \
  } while (0)

namespace rmm {

// Set true to enable free/total memory logging at each RMM call (expensive)
constexpr bool RMM_USAGE_LOGGING{false};

// RAII logger class
class LogIt {
 public:
  LogIt(Logger::MemEvent_t event, void* ptr, size_t size, cudaStream_t stream,
        const char* filename, unsigned int line,
        bool usageLogging = RMM_USAGE_LOGGING)
      : event(event),
        device(0),
        ptr(ptr),
        size(size),
        stream(stream),
        usageLogging(usageLogging),
        line(line) {
    if (filename) file = filename;
    if (Manager::getOptions().enable_logging) {
      cudaGetDevice(&device);
      start = std::chrono::system_clock::now();
    }
  }

  /// Sometimes you need to start logging before the pointer address is
  /// known
  inline void setPointer(void* p) {
    if (Manager::getOptions().enable_logging) ptr = p;
  }

  ~LogIt() {
    if (Manager::getOptions().enable_logging) {
      Logger::TimePt end = std::chrono::system_clock::now();
      size_t freeMem = 0, totalMem = 0;
      if (usageLogging) rmmGetInfo(&freeMem, &totalMem, stream);
      Manager::getLogger().record(event, device, ptr, start, end, freeMem,
                                  totalMem, size, stream, file, line);
    }
  }

 private:
  rmm::Logger::MemEvent_t event;
  int device;
  void* ptr;
  size_t size;
  cudaStream_t stream;
  rmm::Logger::TimePt start;
  std::string file;
  unsigned int line;
  bool usageLogging;
};

inline bool usePoolAllocator() {
  return Manager::getOptions().allocation_mode == PoolAllocation;
}

/** ---------------------------------------------------------------------------*
 * @brief Allocate memory and return a pointer to device memory.
 *
 * @param[out] ptr Returned pointer
 * @param[in] size The size in bytes of the allocated memory region
 * @param[in] stream The stream in which to synchronize this command
 * @param[in] file The filename location of the call to this function, for
 * tracking
 * @param[in] line The line number of the call to this function, for tracking
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called, RMM_ERROR_INVALID_ARGUMENT if ptr is
 *                    null, RMM_ERROR_OUT_OF_MEMORY if unable to allocate the
 *                    requested size, or RMM_CUDA_ERROR on any other CUDA error.
 * --------------------------------------------------------------------------**/
template <typename T>
inline rmmError_t alloc(T** ptr, size_t size, cudaStream_t stream, const char* file,
                 unsigned int line) {
  rmm::LogIt log(rmm::Logger::Alloc, 0, size, stream, file, line);

  if (!ptr && !size) {
    return RMM_SUCCESS;
  }

  if (!ptr) return RMM_ERROR_INVALID_ARGUMENT;

  if (rmm::usePoolAllocator()) {
    RMM_CHECK(rmm::Manager::getInstance().registerStream(stream));
    RMM_CHECK_CNMEM(cnmemMalloc(reinterpret_cast<void**>(ptr), size, stream));
  } else
    RMM_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));

  log.setPointer(*ptr);
  return RMM_SUCCESS;
}

/** ---------------------------------------------------------------------------*
 * @brief Reallocate device memory block to new size and recycle any remaining
 *        memory.
 *
 * @param[out] ptr Returned pointer
 * @param[in] new_size The size in bytes of the allocated memory region
 * @param[in] stream The stream in which to synchronize this command
 * @param[in] file The filename location of the call to this function, for
 * tracking
 * @param[in] line The line number of the call to this function, for tracking
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called, RMM_ERROR_INVALID_ARGUMENT if ptr is
 *                    null, RMM_ERROR_OUT_OF_MEMORY if unable to allocate the
 *                    requested size, or RMM_ERROR_CUDA_ERROR on any other CUDA
 *                    error.
 * --------------------------------------------------------------------------**/
template <typename T>
inline rmmError_t realloc(T** ptr, size_t new_size, cudaStream_t stream,
                   const char* file, unsigned int line) {
  rmm::LogIt log(rmm::Logger::Realloc, ptr, new_size, stream, file, line);

  if (!ptr && !new_size) {
    return RMM_SUCCESS;
  }

  if (!ptr) return RMM_ERROR_INVALID_ARGUMENT;

  if (rmm::usePoolAllocator()) {
    RMM_CHECK(rmm::Manager::getInstance().registerStream(stream));
    RMM_CHECK_CNMEM(cnmemFree(*reinterpret_cast<void**>(ptr), stream));
    RMM_CHECK_CNMEM(
        cnmemMalloc(reinterpret_cast<void**>(ptr), new_size, stream));
  } else {
    RMM_CHECK_CUDA(cudaFree(*ptr));
    RMM_CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(ptr), new_size));
  }
  log.setPointer(*ptr);
  return RMM_SUCCESS;
}

/** ---------------------------------------------------------------------------*
 * @brief Release device memory and recycle the associated memory.
 *
 * \note Doesn't need to be a template because T* is implicitly conertible to
 * void*
 *
 * @param[in] ptr The pointer to free
 * @param[in] stream The stream in which to synchronize this command
 * @param[in] file The filename location of the call to this function, for
 * tracking
 * @param[in] line The line number of the call to this function, for tracking
 * @return rmmError_t RMM_SUCCESS, or RMM_ERROR_NOT_INITIALIZED if rmmInitialize
 *                    has not been called,or RMM_ERROR_CUDA_ERROR on any CUDA
 *                    error.
 * --------------------------------------------------------------------------**/
inline rmmError_t free(void* ptr, cudaStream_t stream, const char* file,
                   unsigned int line) {
  rmm::LogIt log(rmm::Logger::Free, ptr, 0, stream, file, line);
  if (rmm::usePoolAllocator())
    RMM_CHECK_CNMEM(cnmemFree(ptr, stream));
  else
    RMM_CHECK_CUDA(cudaFree(ptr));
  return RMM_SUCCESS;
}

}  // namespace rmm
#endif
