#include <cudf/io/text/host_device_istream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <istream>

namespace cudf {
namespace io {
namespace text {

uint32_t host_device_istream::read(cudf::device_span<char> destination,
                                   rmm::cuda_stream_view stream)
{
  auto read_size = destination.size();

  if (_host_buffer.size() < read_size) { _host_buffer.resize(read_size); }

  _source_stream.read(_host_buffer.data(), read_size);

  auto read_size_actual = _source_stream.gcount();

  CUDA_TRY(cudaMemcpyAsync(  //
    destination.data(),
    _host_buffer.data(),
    read_size_actual,
    cudaMemcpyHostToDevice,
    stream.value()));

  // std::cout << "tried to read: " << read_size << ", and got: " << read_size_actual << std::endl;

  return read_size_actual;
}

void host_device_istream::reset()
{
  _source_stream.clear();
  _source_stream.seekg(0, _source_stream.beg);  //
}

}  // namespace text
}  // namespace io
}  // namespace cudf
