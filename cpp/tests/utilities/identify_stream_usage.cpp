/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

/**
 * @brief Print a backtrace and raise an error if stream is a default stream.
 */
void check_stream_and_error(cudaStream_t stream)
{
  // We explicitly list the possibilities rather than using
  // `cudf::get_default_stream().value()` for two reasons:
  // 1. There is no guarantee that `thrust::device` and the default value of
  //    `cudf::get_default_stream().value()` are actually the same. At present,
  //    the former is `cudaStreamLegacy` while the latter is 0.
  // 2. Using the cudf default stream would require linking against cudf, which
  //    adds unnecessary complexity to the build process (especially in CI)
  //    when this simple approach is sufficient.
  if (stream == cudaStreamDefault || (stream == cudaStreamLegacy) ||
      (stream == cudaStreamPerThread)) {
#ifdef __GNUC__
    // If we're on the wrong stream, print the stack trace from the current frame.
    // Adapted from from https://panthema.net/2008/0901-stacktrace-demangled/
    constexpr int kMaxStackDepth = 64;
    void* stack[kMaxStackDepth];
    auto depth   = backtrace(stack, kMaxStackDepth);
    auto strings = backtrace_symbols(stack, depth);

    if (strings == nullptr) {
      std::cout << "No stack trace could be found!" << std::endl;
    } else {
      // If we were able to extract a trace, parse it, demangle symbols, and
      // print a readable output.

      // allocate string which will be filled with the demangled function name
      size_t funcnamesize = 256;
      char* funcname      = (char*)malloc(funcnamesize);

      // Start at frame 1 to skip print_trace itself.
      for (int i = 1; i < depth; ++i) {
        char* begin_name   = nullptr;
        char* begin_offset = nullptr;
        char* end_offset   = nullptr;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char* p = strings[i]; *p; ++p) {
          if (*p == '(') {
            begin_name = p;
          } else if (*p == '+') {
            begin_offset = p;
          } else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
          }
        }

        if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
          *begin_name++   = '\0';
          *begin_offset++ = '\0';
          *end_offset     = '\0';

          // mangled name is now in [begin_name, begin_offset) and caller offset
          // in [begin_offset, end_offset). now apply __cxa_demangle():

          int status;
          char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
          if (status == 0) {
            funcname =
              ret;  // use possibly realloc()-ed string (__cxa_demangle may realloc funcname)
            std::cout << "#" << i << " in " << strings[i] << " : " << funcname << "+"
                      << begin_offset << std::endl;
          } else {
            // demangling failed. Output function name as a C function with no arguments.
            std::cout << "#" << i << " in " << strings[i] << " : " << begin_name << "()+"
                      << begin_offset << std::endl;
          }
        } else {
          std::cout << "#" << i << " in " << strings[i] << std::endl;
        }
      }

      free(funcname);
    }
    free(strings);
#else
    std::cout << "Backtraces are only when built with a GNU compiler." << std::endl;
#endif  // __GNUC__
    throw std::runtime_error("Found unexpected default stream!");
  }
}

/**
 * @brief Container for CUDA APIs that have been overloaded using DEFINE_OVERLOAD.
 *
 * This variable must be initialized before everything else.
 *
 * @see find_originals for a description of the priorities
 */
__attribute__((init_priority(1001))) std::unordered_map<std::string, void*> originals;

/**
 * @brief Macro for generating functions to override existing CUDA functions.
 *
 * Define a new function with the provided signature that checks the used
 * stream and raises an exception if it is one of CUDA's default streams. If
 * not, the new function forwards all arguments to the original function.
 *
 * Note that since this only defines the function, we do not need default
 * parameter values since those will be provided by the original declarations
 * in CUDA itself.
 *
 * @see find_originals for a description of the priorities
 *
 * @param function The function to overload.
 * @param signature The function signature (must include names, not just types).
 * @parameter arguments The function arguments (names only, no types).
 */
#define DEFINE_OVERLOAD(function, signature, arguments)     \
  using function##_t = cudaError_t (*)(signature);          \
                                                            \
  cudaError_t function(signature)                           \
  {                                                         \
    check_stream_and_error(stream);                         \
    return ((function##_t)originals[#function])(arguments); \
  }                                                         \
  __attribute__((constructor(1002))) void queue_##function() { originals[#function] = nullptr; }

/**
 * @brief Helper macro to define macro arguments that contain a comma.
 */
#define ARG(...) __VA_ARGS__

// clang-format off
/*
   We need to overload all the functions from the runtime API (assuming that we
   don't use the driver API) that accept streams. The main webpage for APIs is
   https://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules. Here are
   the modules containing any APIs using streams as of 9/20/2022:
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL
 */
// clang-format on

// Event APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT
DEFINE_OVERLOAD(cudaEventRecord, ARG(cudaEvent_t event, cudaStream_t stream), ARG(event, stream));

DEFINE_OVERLOAD(cudaEventRecordWithFlags,
                ARG(cudaEvent_t event, cudaStream_t stream, unsigned int flags),
                ARG(event, stream, flags));

// Execution APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION
DEFINE_OVERLOAD(cudaLaunchKernel,
                ARG(const void* func,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t sharedMem,
                    cudaStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD(cudaLaunchCooperativeKernel,
                ARG(const void* func,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t sharedMem,
                    cudaStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD(cudaLaunchHostFunc,
                ARG(cudaStream_t stream, cudaHostFn_t fn, void* userData),
                ARG(stream, fn, userData));

// Memory transfer APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
DEFINE_OVERLOAD(cudaMemPrefetchAsync,
                ARG(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream),
                ARG(devPtr, count, dstDevice, stream));
DEFINE_OVERLOAD(cudaMemcpy2DAsync,
                ARG(void* dst,
                    size_t dpitch,
                    const void* src,
                    size_t spitch,
                    size_t width,
                    size_t height,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, dpitch, src, spitch, width, height, kind, stream));
DEFINE_OVERLOAD(cudaMemcpy2DFromArrayAsync,
                ARG(void* dst,
                    size_t dpitch,
                    cudaArray_const_t src,
                    size_t wOffset,
                    size_t hOffset,
                    size_t width,
                    size_t height,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream));
DEFINE_OVERLOAD(cudaMemcpy2DToArrayAsync,
                ARG(cudaArray_t dst,
                    size_t wOffset,
                    size_t hOffset,
                    const void* src,
                    size_t spitch,
                    size_t width,
                    size_t height,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, wOffset, hOffset, src, spitch, width, height, kind, stream));
DEFINE_OVERLOAD(cudaMemcpy3DAsync,
                ARG(const cudaMemcpy3DParms* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(cudaMemcpy3DPeerAsync,
                ARG(const cudaMemcpy3DPeerParms* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(
  cudaMemcpyAsync,
  ARG(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream),
  ARG(dst, src, count, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyFromSymbolAsync,
                ARG(void* dst,
                    const void* symbol,
                    size_t count,
                    size_t offset,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, symbol, count, offset, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyToSymbolAsync,
                ARG(const void* symbol,
                    const void* src,
                    size_t count,
                    size_t offset,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(symbol, src, count, offset, kind, stream));
DEFINE_OVERLOAD(
  cudaMemset2DAsync,
  ARG(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream),
  ARG(devPtr, pitch, value, width, height, stream));
DEFINE_OVERLOAD(
  cudaMemset3DAsync,
  ARG(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream),
  ARG(pitchedDevPtr, value, extent, stream));
DEFINE_OVERLOAD(cudaMemsetAsync,
                ARG(void* devPtr, int value, size_t count, cudaStream_t stream),
                ARG(devPtr, value, count, stream));

// Memory allocation APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS
DEFINE_OVERLOAD(cudaFreeAsync, ARG(void* devPtr, cudaStream_t stream), ARG(devPtr, stream));
DEFINE_OVERLOAD(cudaMallocAsync,
                ARG(void** devPtr, size_t size, cudaStream_t stream),
                ARG(devPtr, size, stream));
DEFINE_OVERLOAD(cudaMallocFromPoolAsync,
                ARG(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream),
                ARG(ptr, size, memPool, stream));

namespace cudf {

/**
 * @brief Get the current default stream
 *
 * Overload the default function to return a new stream here.
 *
 * @return The current default stream.
 */
rmm::cuda_stream_view const get_default_stream()
{
  static rmm::cuda_stream stream{};
  return {stream};
}

}  // namespace cudf

/**
 * @brief Function to collect all the original CUDA symbols corresponding to overloaded functions.
 *
 * Note on priorities:
 * - `originals` must be initialized first, so it is 1001.
 * - The function names must be added to originals next in the macro, so those are 1002.
 * - Finally, this function actually finds the original symbols so it is 1003.
 */
__attribute__((constructor(1003))) void find_originals()
{
  for (auto it : originals) {
    originals[it.first] = dlsym(RTLD_NEXT, it.first.data());
  }
}
