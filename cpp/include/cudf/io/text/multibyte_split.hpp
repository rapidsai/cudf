#include <cudf/io/text/data_chunk_source.hpp>

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <iostream>
#include <memory>

namespace cudf {
namespace io {
namespace text {

std::unique_ptr<cudf::column> multibyte_split(
  data_chunk_source& source,
  std::vector<std::string> const& delimiters,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace text
}  // namespace io
}  // namespace cudf
