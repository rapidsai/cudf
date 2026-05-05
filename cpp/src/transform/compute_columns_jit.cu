


#include <cudf/transform.hpp>

namespace CUDF_EXPORT cudf {

std::unique_ptr<column> compute_columns(
  std::span<transform_input const> inputs,
  std::span<uint8_t const> udf,
  std::span<data_type const> output_types,
  void* user_data,
  null_aware is_null_aware          = null_aware::NO,
  std::optional<size_type> row_size = std::nullopt,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}