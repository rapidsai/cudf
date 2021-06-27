#pragma once

#include <cudf/io/text/input_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <istream>

namespace cudf {
namespace io {
namespace text {

class host_input_stream : public cudf::io::text::input_stream {
 public:
  host_input_stream(std::istream& source_stream) : _source_stream(source_stream) {}

  uint32_t readsome(cudf::device_span<char> destination, rmm::cuda_stream_view stream) override;

 private:
  std::istream& _source_stream;
  thrust::host_vector<char> _host_buffer{};
};

}  // namespace text
}  // namespace io
}  // namespace cudf
