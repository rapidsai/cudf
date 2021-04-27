/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace cuco {
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  cuda_error(const char* message) : std::runtime_error(message) {}
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};
}  // namespace cuco

#define STRINGIFY_DETAIL(x) #x
#define CUCO_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing `cuco::cuda_error`, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws `rmm::cuda_error` if `cudaMalloc` fails
 * CUCO_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws `std::runtime_error` if `cudaMalloc` fails
 * CUCO_CUDA_TRY(cudaMalloc(&p, 100), std::runtime_error);
 * ```
 *
 */
#define CUCO_CUDA_TRY(...)                                               \
  GET_CUCO_CUDA_TRY_MACRO(__VA_ARGS__, CUCO_CUDA_TRY_2, CUCO_CUDA_TRY_1) \
  (__VA_ARGS__)
#define GET_CUCO_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUCO_CUDA_TRY_2(_call, _exception_type)                                                    \
  do {                                                                                             \
    cudaError_t const error = (_call);                                                             \
    if (cudaSuccess != error) {                                                                    \
      cudaGetLastError();                                                                          \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + CUCO_STRINGIFY(__LINE__) + \
                            ": " + cudaGetErrorName(error) + " " + cudaGetErrorString(error)};     \
    }                                                                                              \
  } while (0);
#define CUCO_CUDA_TRY_1(_call) CUCO_CUDA_TRY_2(_call, cuco::cuda_error)

/**
 * @brief Error checking macro for CUDA runtime API that asserts the result is
 * equal to `cudaSuccess`.
 *
 */
#define CUCO_ASSERT_CUDA_SUCCESS(expr) \
  do {                                 \
    cudaError_t const status = (expr); \
    assert(cudaSuccess == status);     \
  } while (0)
