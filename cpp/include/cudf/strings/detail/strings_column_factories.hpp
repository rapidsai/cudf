#include <cudf/column/column.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> make_strings_column(
  device_span<string_view> const& string_views,
  string_view const null_placeholder,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<column> make_strings_column(
  device_span<char> const& chars,
  device_span<size_type> const& offsets,
  size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail

}  // namespace cudf
