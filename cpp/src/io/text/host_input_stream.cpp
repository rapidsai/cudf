#include <cudf/io/text/host_input_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <istream>

namespace cudf {
namespace io {
namespace text {

uint32_t host_input_stream::readsome(cudf::device_span<char> destination,
                                     rmm::cuda_stream_view stream)
{
  auto read_size = destination.size();

  if (_host_buffer.size() < read_size) { _host_buffer.resize(read_size); }

  read_size = _source_stream.readsome(_host_buffer.data(), read_size);

  CUDA_TRY(cudaMemcpyAsync(  //
    destination.data(),
    _host_buffer.data(),
    read_size,
    cudaMemcpyHostToDevice,
    stream.value()));

  return read_size;
}

}  // namespace text
}  // namespace io
}  // namespace cudf
