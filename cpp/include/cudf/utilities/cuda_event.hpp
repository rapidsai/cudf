#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda_runtime_api.h>

namespace CUDF_EXPORT cudf {

namespace detail {
/**
 * @brief RAII struct to wrap a cuda event and ensure its proper destruction.
 */
struct cuda_event {
  cuda_event();
  virtual ~cuda_event();

  // Moveable but not copyable.
  cuda_event(const cuda_event&)            = delete;
  cuda_event& operator=(const cuda_event&) = delete;

  cuda_event(cuda_event&&)            = default;
  cuda_event& operator=(cuda_event&&) = default;

  operator cudaEvent_t() const;

 private:
  cudaEvent_t e_;
};
}  // namespace detail

}  // namespace CUDF_EXPORT cudf
