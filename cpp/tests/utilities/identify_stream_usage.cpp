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

#include <cudf/detail/utilities/stacktrace.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// We control whether to override cudf::test::get_default_stream or
// cudf::get_default_stream with a compile-time flag. Thesee are the two valid
// options:
// 1. STREAM_MODE_TESTING=OFF: In this mode, cudf::get_default_stream will
//    return a custom stream and stream_is_invalid will return true if any CUDA
//    API is called using any of CUDA's default stream constants
//    (cudaStreamLegacy, cudaStreamDefault, or cudaStreamPerThread). This check
//    is sufficient to ensure that cudf is using cudf::get_default_stream
//    everywhere internally rather than implicitly using stream 0,
//    cudaStreamDefault, cudaStreamLegacy, thrust execution policies, etc. It
//    is not sufficient to guarantee a stream-ordered API because it will not
//    identify places in the code that use cudf::get_default_stream instead of
//    properly forwarding along a user-provided stream.
// 2. STREAM_MODE_TESTING=ON: In this mode, cudf::test::get_default_stream
//    returns a custom stream and stream_is_invalid returns true if any CUDA
//    API is called using any stream other than cudf::test::get_default_stream.
//    This is a necessary and sufficient condition to ensure that libcudf is
//    properly passing streams through all of its (tested) APIs.

namespace cudf {

#ifdef STREAM_MODE_TESTING
namespace test {
#endif

rmm::cuda_stream_view const get_default_stream()
{
  static rmm::cuda_stream stream{};
  return {stream};
}

#ifdef STREAM_MODE_TESTING
}  // namespace test
#endif

}  // namespace cudf

bool stream_is_invalid(cudaStream_t stream)
{
#ifdef STREAM_MODE_TESTING
  // In this mode the _only_ valid stream is the one returned by cudf::test::get_default_stream.
  return (stream != cudf::test::get_default_stream().value());
#else
  // We explicitly list the possibilities rather than using
  // `cudf::get_default_stream().value()` because there is no guarantee that
  // `thrust::device` and the default value of
  // `cudf::get_default_stream().value()` are actually the same. At present, the
  // former is `cudaStreamLegacy` while the latter is 0.
  return (stream == cudaStreamDefault) || (stream == cudaStreamLegacy) ||
         (stream == cudaStreamPerThread);
#endif
}

/**
 * @brief Print a backtrace and raise an error if stream is a default stream.
 */
void check_stream_and_error(cudaStream_t stream)
{
  if (stream_is_invalid(stream)) {
    // Exclude the current function from stacktrace.
    std::cout << cudf::detail::get_stacktrace(cudf::detail::capture_last_stackframe::NO)
              << std::endl;

    char const* env_stream_error_mode{std::getenv("GTEST_CUDF_STREAM_ERROR_MODE")};
    if (env_stream_error_mode && !strcmp(env_stream_error_mode, "print")) {
      std::cout << "cudf_identify_stream_usage found unexpected stream!" << std::endl;
    } else {
      throw std::runtime_error("cudf_identify_stream_usage found unexpected stream!");
    }
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
                ARG(void const* func,
                    dim3 gridDim,
                    dim3 blockDim,
                    void** args,
                    size_t sharedMem,
                    cudaStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD(cudaLaunchCooperativeKernel,
                ARG(void const* func,
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
                ARG(void const* devPtr, size_t count, int dstDevice, cudaStream_t stream),
                ARG(devPtr, count, dstDevice, stream));
DEFINE_OVERLOAD(cudaMemcpy2DAsync,
                ARG(void* dst,
                    size_t dpitch,
                    void const* src,
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
                    void const* src,
                    size_t spitch,
                    size_t width,
                    size_t height,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, wOffset, hOffset, src, spitch, width, height, kind, stream));
DEFINE_OVERLOAD(cudaMemcpy3DAsync,
                ARG(cudaMemcpy3DParms const* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(cudaMemcpy3DPeerAsync,
                ARG(cudaMemcpy3DPeerParms const* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(
  cudaMemcpyAsync,
  ARG(void* dst, void const* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream),
  ARG(dst, src, count, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyFromSymbolAsync,
                ARG(void* dst,
                    void const* symbol,
                    size_t count,
                    size_t offset,
                    cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, symbol, count, offset, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyToSymbolAsync,
                ARG(void const* symbol,
                    void const* src,
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
