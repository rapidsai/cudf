#pragma once

#include <rmm/device_buffer.hpp>

#include <cudf/utilities/span.hpp>

namespace cudf {
namespace io {
namespace text {

class device_istream {
 public:
  virtual uint32_t readsome(cudf::device_span<char> destination, rmm::cuda_stream_view stream) = 0;
  virtual uint32_t tellg()                                                                     = 0;
  virtual void seekg(uint32_t pos)                                                             = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
