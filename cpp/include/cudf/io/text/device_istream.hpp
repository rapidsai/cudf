#pragma once

#include <rmm/device_buffer.hpp>

#include <cudf/utilities/span.hpp>

namespace cudf {
namespace io {
namespace text {

class device_istream {
 public:
  virtual uint32_t read(cudf::device_span<char> destination, rmm::cuda_stream_view stream) = 0;
  virtual void reset()                                                                     = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
