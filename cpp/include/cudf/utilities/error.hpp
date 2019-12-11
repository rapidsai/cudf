#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string>
#include <stdexcept>

#include <rmm/rmm.h>

#define RMM_TRY(call)                                             \
  do {                                                            \
    rmmError_t const status = (call);                             \
    if (RMM_SUCCESS != status) {                                  \
      cudf::detail::throw_rmm_error(status, __FILE__, __LINE__);  \
    }                                                             \
  } while (0);

#define RMM_TRY_CUDAERROR(x) \
  if ((x) != RMM_SUCCESS) CUDA_TRY(cudaPeekAtLastError());

namespace cudf {
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUDF_EXPECTS macro.
 *
 *---------------------------------------------------------------------------**/
struct logic_error : public std::logic_error {
  logic_error(char const* const message) : std::logic_error(message) {}

  logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::runtime_error {
  cuda_error(std::string const& message) : std::runtime_error(message) {}
};
}  // namespace cudf

#define STRINGIFY_DETAIL(x) #x
#define CUDF_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
#define CUDF_EXPECTS(cond, reason)                           \
  (!!(cond))                                                 \
      ? static_cast<void>(0)                                 \
      : throw cudf::logic_error("cuDF failure at: " __FILE__ \
                                ":" CUDF_STRINGIFY(__LINE__) ": " reason)

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
#define CUDF_FAIL(reason)                              \
  throw cudf::logic_error("cuDF failure at: " __FILE__ \
                          ":" CUDF_STRINGIFY(__LINE__) ": " reason)

namespace cudf {
namespace detail {

inline void throw_rmm_error(rmmError_t error, const char* file,
                             unsigned int line) {
  // todo: throw cuda_error if the error is from cuda
  throw cudf::logic_error(
      std::string{"RMM error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  rmmGetErrorString(error)});
}

inline void throw_cuda_error(cudaError_t error, const char* file,
                             unsigned int line) {
  throw cudf::cuda_error(
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

template <bool sync_on_stream>
inline std::enable_if_t<sync_on_stream, void>
check_cuda_error(char const* file, unsigned int line, cudaStream_t stream=0) {
  cudaError_t error{cudaSuccess};

  error = cudaStreamSynchronize(stream);
  if (cudaSuccess != error) {
    throw_cuda_error(error, file, line);
  }

  error = cudaGetLastError();
  if (cudaSuccess != error) {
    throw_cuda_error(error, file, line);
  }
}

template <bool sync_on_stream>
inline std::enable_if_t<not sync_on_stream, void>
check_cuda(char const* file, unsigned int line, cudaStream_t stream=0) {
  auto error = cudaGetLastError();
  if (cudaSuccess != error) {
    throw_cuda_error(error, file, line);
  }
}
}  // namespace detail
}  // namespace cudf

/**---------------------------------------------------------------------------*
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, throws an exception detailing the CUDA error that occurred.
 *
 * This macro supersedes GDF_REQUIRE and should be preferred in all instances.
 * GDF_REQUIRE should be considered deprecated.
 *
 *---------------------------------------------------------------------------**/
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
      cudf::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0);

/**---------------------------------------------------------------------------*
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any CUDA errors returned from cudaGetLastError. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 *
 *---------------------------------------------------------------------------**/
#ifndef NDEBUG
#define CHECK_CUDA(stream) \
  cudf::detail::check_cuda<true>(__FILE__, __LINE__, stream)
#else
#define CHECK_CUDA(stream) \
  cudf::detail::check_cuda<false>(__FILE__, __LINE__, stream)
#endif
