#pragma once

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {
namespace text {

struct data_chunk {
  data_chunk(rmm::device_buffer&& buffer, std::size_t size)
    : _buffer(std::move(buffer)), _size(size)
  {
  }

  operator cudf::device_span<char const>()
  {
    return cudf::device_span<char const>(static_cast<char const*>(_buffer.data()), _size);
  }

  uint32_t size() const { return _size; }

  rmm::cuda_stream_view stream() const { return _buffer.stream(); }

 private:
  rmm::device_buffer _buffer;
  std::size_t _size;
};

class data_chunk_reader {
 public:
  virtual data_chunk get_next_chunk(uint32_t size, rmm::cuda_stream_view stream) = 0;
};

class data_chunk_source {
 public:
  virtual std::unique_ptr<data_chunk_reader> create_reader() = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
