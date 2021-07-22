#pragma once

#include <cudf/io/text/device_istream.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <istream>

namespace cudf {
namespace io {
namespace text {

class host_device_istream : public cudf::io::text::device_istream {
 public:
  host_device_istream(std::istream& source_stream) : _source_stream(source_stream) {}

  uint32_t read(cudf::device_span<char> destination, rmm::cuda_stream_view stream) override;

  void reset() override;

 private:
  std::istream& _source_stream;
  thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>>
    _host_buffer{};
};

}  // namespace text
}  // namespace io
}  // namespace cudf
