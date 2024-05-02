/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/stacktrace.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace cudf {
/**
 * @addtogroup utility_error
 * @{
 * @file
 */

/**
 * @brief The struct to store the current stacktrace upon its construction.
 */
struct stacktrace_recorder {
  stacktrace_recorder()
    // Exclude the current stackframe, as it is this constructor.
    : _stacktrace{cudf::detail::get_stacktrace(cudf::detail::capture_last_stackframe::NO)}
  {
  }

 public:
  /**
   * @brief Get the stored stacktrace captured during object construction.
   *
   * @return The pointer to a null-terminated string storing the output stacktrace
   */
  char const* stacktrace() const { return _stacktrace.c_str(); }

 protected:
  std::string const _stacktrace;  //!< The whole stacktrace stored as one string.
};

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUDF_EXPECTS macro.
 */
struct logic_error : public std::logic_error, public stacktrace_recorder {
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

  ~logic_error()
  {
    // Needed so that the first instance of the implicit destructor for any TU isn't 'constructed'
    // from a host+device function marking the implicit version also as host+device
  }
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : public std::runtime_error, public stacktrace_recorder {
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

/**
 * @brief Exception thrown when an operation is attempted on an unsupported dtype.
 *
 * This exception should be thrown when an operation is attempted on an
 * unsupported data_type. This exception should not be thrown directly and is
 * instead thrown by the CUDF_EXPECTS or CUDF_FAIL macros.
 */
struct data_type_error : public std::invalid_argument, public stacktrace_recorder {
  /**
   * @brief Constructs a data_type_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  data_type_error(char const* const message) : std::invalid_argument(message) {}

  /**
   * @brief Construct a new data_type_error object with error message
   *
   * @param message Message to be associated with the exception
   */
  data_type_error(std::string const& message) : std::invalid_argument(message) {}
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
 * Defaults to throwing `cudf::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws cudf::logic_error
 * CUDF_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * CUDF_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `cudf::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define CUDF_EXPECTS(...)                                             \
  GET_CUDF_EXPECTS_MACRO(__VA_ARGS__, CUDF_EXPECTS_3, CUDF_EXPECTS_2) \
  (__VA_ARGS__)

/// @cond

#define GET_CUDF_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define CUDF_EXPECTS_3(_condition, _reason, _exception_type)                    \
  do {                                                                          \
    static_assert(std::is_base_of_v<std::exception, _exception_type>);          \
    (_condition) ? static_cast<void>(0)                                         \
                 : throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/ \
      {"CUDF failure at: " __FILE__ ":" CUDF_STRINGIFY(__LINE__) ": " _reason}; \
  } while (0)

#define CUDF_EXPECTS_2(_condition, _reason) CUDF_EXPECTS_3(_condition, _reason, cudf::logic_error)

/// @endcond

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws `cudf::logic_error`
 * CUDF_FAIL("Unsupported code path");
 *
 * // Throws `std::runtime_error`
 * CUDF_FAIL("Unsupported code path", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to `cudf::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define CUDF_FAIL(...)                                       \
  GET_CUDF_FAIL_MACRO(__VA_ARGS__, CUDF_FAIL_2, CUDF_FAIL_1) \
  (__VA_ARGS__)

/// @cond

#define GET_CUDF_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define CUDF_FAIL_2(_what, _exception_type)      \
  /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/ \
  throw _exception_type { "CUDF failure at:" __FILE__ ":" CUDF_STRINGIFY(__LINE__) ": " _what }

#define CUDF_FAIL_1(_what) CUDF_FAIL_2(_what, cudf::logic_error)

/// @endcond

namespace cudf {
namespace detail {
// @cond
inline void throw_cuda_error(cudaError_t error, char const* file, unsigned int line)
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
