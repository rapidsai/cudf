#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <iostream>
#include <memory>

namespace cudf {
namespace io {
namespace text {

std::unique_ptr<cudf::column> multibyte_split(
  std::istream& input,
  std::string delimeter,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}
}  // namespace io
}  // namespace cudf
