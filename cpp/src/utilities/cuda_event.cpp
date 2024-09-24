#include <cudf/utilities/cuda_event.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/detail/error.hpp>

namespace cudf::detail {

cuda_event::cuda_event() { CUDF_CUDA_TRY(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }

cuda_event::~cuda_event() { RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(e_)); }

cuda_event::operator cudaEvent_t() const { return e_; }

}  // namespace cudf::detail
