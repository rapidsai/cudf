/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime.h>

/**
 * @brief Print a backtrace and raise and error if stream is a default stream.
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
  if (stream == static_cast<cudaStream_t>(0) || (stream == cudaStreamLegacy) ||
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
    std::cout << "Backtraces are only support on GNU systems." << std::endl;
#endif  // __GNUC__
    throw std::runtime_error("Found unexpected default stream!");
  }
}

/**
 * @brief Container for CUDA APIs that have been overloaded using DEFINE_OVERLOAD.
 */
static std::unordered_map<std::string, void*> originals;

// Note: This macro assumes that defaults can be "inherited" from the declarations in
// cuda_runtime.h.
/**
 * @brief Macro for generating functions to override existing CUDA functions.
 *
 * Define a new function with the provided signature that checks the used
 * stream and raises an exception if it is one of CUDA's default streams. If
 * not, the new function forwards all arguments to the original function.
 *
 * @param function The function to overload.
 * @param ret_type The return type of the function
 * @param signature The function signature (must include names, not just types).
 * @parameter arguments The function arguments (names only, no types).
 */
#define DEFINE_OVERLOAD(function, ret_type, signature, arguments, attributes) \
  using function##_t = ret_type (*)(signature);                               \
                                                                              \
  attributes ret_type function(signature)                                     \
  {                                                                           \
    check_stream_and_error(stream);                                           \
    return ((function##_t)originals["function"])(arguments);                  \
  }

/**
 * @brief Helper macro to define macro arguments that contain a comma.
 */
#define ARG(...) __VA_ARGS__

#define DEFINE_OVERLOAD_HOST(function, ret_type, signature, arguments) \
  DEFINE_OVERLOAD(function, ret_type, ARG(signature), ARG(arguments), __host__)

#define DEFINE_OVERLOAD_HOST_DEVICE(function, ret_type, signature, arguments) \
  DEFINE_OVERLOAD(function, ret_type, ARG(signature), ARG(arguments), __host__ __device__)

// clang-format off
/*
   We need to overload all the functions from the runtime API (assuming that we
   don't use the driver API) that accept streams. The main webpage for APIs is
   https://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules. Here are
   the modules containing any APIs using streams as of 9/20/2022:
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL
 */
// clang-format on

// Event APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT

DEFINE_OVERLOAD_HOST_DEVICE(cudaEventRecord,
                            cudaError_t,
                            ARG(cudaEvent_t event, cudaStream_t stream),
                            ARG(event, stream));

DEFINE_OVERLOAD_HOST(cudaEventRecordWithFlags,
                     cudaError_t,
                     ARG(cudaEvent_t event, cudaStream_t stream, unsigned int flags),
                     ARG(event, stream, flags));
// Execution APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION
DEFINE_OVERLOAD_HOST(cudaLaunchKernel,
                     cudaError_t,
                     ARG(const void* func,
                         dim3 gridDim,
                         dim3 blockDim,
                         void** args,
                         size_t sharedMem,
                         cudaStream_t stream),
                     ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD_HOST(cudaLaunchCooperativeKernel,
                     cudaError_t,
                     ARG(const void* func,
                         dim3 gridDim,
                         dim3 blockDim,
                         void** args,
                         size_t sharedMem,
                         cudaStream_t stream),
                     ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD_HOST(cudaLaunchHostFunc,
                     cudaError_t,
                     ARG(cudaStream_t stream, cudaHostFn_t fn, void* userData),
                     ARG(stream, fn, userData));

__attribute__((constructor)) void init()
{
  for (auto it : originals) {
    originals[it.first] = dlsym(RTLD_NEXT, it.first.data());
  }
}
