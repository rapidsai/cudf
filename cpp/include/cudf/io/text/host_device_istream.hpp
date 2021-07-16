#pragma once

#include <cudf/io/text/device_istream.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <istream>

namespace cudf {
namespace io {
namespace text {

class host_device_istream : public cudf::io::text::device_istream {
 public:
  host_device_istream(std::istream& source_stream) : _source_stream(source_stream) {}

  uint32_t readsome(cudf::device_span<char> destination, rmm::cuda_stream_view stream) override;

  uint32_t tellg() override;

  void seekg(uint32_t pos) override;

 private:
  std::istream& _source_stream;
  thrust::host_vector<char> _host_buffer{};
};

}  // namespace text
}  // namespace io
}  // namespace cudf
