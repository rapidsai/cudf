#include <cudf/io/text/device_istream.hpp>

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <iostream>
#include <memory>

namespace cudf {
namespace io {
namespace text {

std::unique_ptr<cudf::column> multibyte_split(
  cudf::string_scalar const& input,
  std::vector<std::string> const& delimeters,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> multibyte_split(
  cudf::io::text::device_istream& input,
  std::vector<std::string> const& delimeters,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace text
}  // namespace io
}  // namespace cudf
