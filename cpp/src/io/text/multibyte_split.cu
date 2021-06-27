#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <bitset>
#include <iostream>
#include <memory>

namespace {

}

namespace cudf {
namespace io {
namespace text {
namespace detail {

std::unique_ptr<cudf::column> multibyte_split(std::istream& input,
                                              std::string delimeter,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FAIL();
}

}  // namespace detail

std::unique_ptr<cudf::column> multibyte_split(std::istream& input,
                                              std::string delimeter,
                                              rmm::mr::device_memory_resource* mr)
{
  char c;
  while (input.readsome(&c, 1) > 0) { std::cout << std::bitset<8>(c) << std::endl; }
  std::cout << std::endl;

  CUDF_FAIL();
}

}  // namespace text
}  // namespace io
}  // namespace cudf
