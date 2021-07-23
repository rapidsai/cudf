#pragma once

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace io {
namespace text {

/**
 * @brief represents a possibly-shared view over device memory.
 */
struct data_chunk {
  data_chunk(device_span<char const> data) : _data(data) {}

  operator cudf::device_span<char const>() { return _data; }

  uint32_t size() const { return _data.size(); }

 private:
  device_span<char const> _data;
};

/**
 * @brief a reader capable of producing views over device memory
 *
 */
class data_chunk_reader {
 public:
  virtual data_chunk get_next_chunk(uint32_t size, rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief a data source capable of creating a reader which can produce views of the data source in
 * device memory.
 *
 */
class data_chunk_source {
 public:
  virtual std::unique_ptr<data_chunk_reader> create_reader() = 0;
};

}  // namespace text
}  // namespace io
}  // namespace cudf
