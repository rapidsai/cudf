#include <cudf/column/column.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> make_strings_column(
  const device_span<string_view>& string_views,
  const string_view null_placeholder,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}
}  // namespace cudf
