/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace cudf {
/**
 * @addtogroup utility_error
 * @{
 * @file
 */

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUDF_EXPECTS macro.
 */
struct logic_error : public std::logic_error {
  /**
   * @brief Constructs a logic_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  logic_error(char const* const message) : std::logic_error(message) {}

  /**
   * @brief Construct a new logic error object with error message
   *
   * @param message Message to be associated with the exception
   */
  logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error {
  /**
   * @brief Construct a new cuda error object with error message and code.
   *
   * @param message Error message
   * @param error CUDA error code
   */
  cuda_error(std::string const& message, cudaError_t const& error)
    : std::runtime_error(message), _cudaError(error)
  {
  }

 public:
  /**
   * @brief Returns the CUDA error code associated with the exception.
   *
   * @return CUDA error code
   */
  cudaError_t error_code() const { return _cudaError; }

 protected:
  cudaError_t _cudaError;  //!< CUDA error code
};

struct fatal_cuda_error : public cuda_error {
  using cuda_error::cuda_error;  // Inherit constructors
};
/** @} */

}  // namespace cudf

#define STRINGIFY_DETAIL(x) #x                   ///< Stringify a macro argument
#define CUDF_STRINGIFY(x)   STRINGIFY_DETAIL(x)  ///< Stringify a macro argument

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Example usage:
 *
 * @code
 * CUDF_EXPECTS(lhs->dtype == rhs->dtype, "Column type mismatch");
 * @endcode
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw cudf::logic_error if the condition evaluates to false.
 */
#define CUDF_EXPECTS(cond, reason)                                  \
  (!!(cond)) ? static_cast<void>(0)                                 \
             : throw cudf::logic_error("cuDF failure at: " __FILE__ \
                                       ":" CUDF_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `cudf::logic_error`.
 *
 *
 * Example usage:
 * ```
 * CUDF_FAIL("Non-arithmetic operation is not supported");
 * ```
 *
 * @param[in] reason String literal description of the reason
 */
#define CUDF_FAIL(reason) \
  throw cudf::logic_error("cuDF failure at: " __FILE__ ":" CUDF_STRINGIFY(__LINE__) ": " reason)

namespace cudf {
namespace detail {
// @cond
inline void throw_cuda_error(cudaError_t error, const char* file, unsigned int line)
{
  // Calls cudaGetLastError to clear the error status. It is nearly certain that a fatal error
  // occurred if it still returns the same error after a cleanup.
  cudaGetLastError();
  auto const last = cudaFree(0);
  auto const msg  = std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                               std::to_string(line) + ": " + std::to_string(error) + " " +
                               cudaGetErrorName(error) + " " + cudaGetErrorString(error)};
  // Call cudaDeviceSynchronize to ensure `last` did not result from an asynchronous error.
  // between two calls.
  if (error == last && last == cudaDeviceSynchronize()) {
    throw fatal_cuda_error{"Fatal " + msg, error};
  } else {
    throw cuda_error{msg, error};
  }
}
// @endcond
}  // namespace detail
}  // namespace cudf

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 */
#define CUDF_CUDA_TRY(call)                                                                    \
  do {                                                                                         \
    cudaError_t const status = (call);                                                         \
    if (cudaSuccess != status) { cudf::detail::throw_cuda_error(status, __FILE__, __LINE__); } \
  } while (0);

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#ifndef NDEBUG
#define CUDF_CHECK_CUDA(stream)                   \
  do {                                            \
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream)); \
    CUDF_CUDA_TRY(cudaPeekAtLastError());         \
  } while (0);
#else
#define CUDF_CHECK_CUDA(stream) CUDF_CUDA_TRY(cudaPeekAtLastError());
#endif
/** @} */
