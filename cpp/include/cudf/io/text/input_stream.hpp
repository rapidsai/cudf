#pragma once

#include <rmm/device_buffer.hpp>

#include <cudf/utilities/span.hpp>

namespace cudf {
namespace io {
namespace text {

class input_stream {
 public:
  virtual uint32_t readsome(cudf::device_span<char> destination, rmm::cuda_stream_view stream) = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
