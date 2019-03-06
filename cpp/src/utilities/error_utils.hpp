#ifndef ERRORUTILS_HPP
#define ERRORUTILS_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

#define CUDA_TRY(call)                                                    \
  {                                                                       \
    cudaError_t cudaStatus = call;                                        \
    if (cudaSuccess != cudaStatus) {                                      \
      std::cerr << "ERROR: CUDA Runtime call " << #call << " in line "    \
                << __LINE__ << " of file " << __FILE__ << " failed with " \
                << cudaGetErrorString(cudaStatus) << " (" << cudaStatus   \
                << ").\n";                                                \
      return GDF_CUDA_ERROR;                                              \
    }                                                                     \
  }

#define RMM_TRY(x) \
  if ((x) != RMM_SUCCESS) return GDF_MEMORYMANAGER_ERROR;

#define RMM_TRY_CUDAERROR(x) \
  if ((x) != RMM_SUCCESS) return cudaPeekAtLastError();

#define CUDA_CHECK_LAST() CUDA_TRY(cudaPeekAtLastError())

#define GDF_REQUIRE(F, S) \
  if (!(F)) return (S);

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

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::runtime_error {
  cuda_error(char const* const message) : std::runtime_error(message) {}
};
}  // namespace cudf

#define STRINGIFY_DETAIL(x) #x
#define CUDF_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**---------------------------------------------------------------------------*
 * @brief Error checking macro that throws an exception when a condition is
 * violated.
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

#define CUDA_EXPECTS(call)                                                 \
  do {                                                                     \
    cudaError_t const status = (call);                                     \
    if (cudaSuccess != status) {                                           \
      std::string const msg{                                               \
          "CUDA error encountered at: " + std::string{__FILE__} + ":" +    \
          CUDF_STRINGIFY(__LINE__) + ": " + std::to_string(status) + " " + \
          cudaGetErrorName(status) + " " + cudaGetErrorString(status)};    \
      throw cudf::cuda_error(msg.c_str());                                 \
    }                                                                      \
  } while (0)
#endif
