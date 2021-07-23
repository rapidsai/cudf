#pragma once

#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/io/text/device_istream.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <fstream>
#include <memory>
#include <string>

namespace cudf {
namespace io {
namespace text {

namespace {

class file_data_chunk_reader : public data_chunk_reader {
 public:
  file_data_chunk_reader(std::string const& filename)
    : _filestream(std::ifstream(filename, std::ifstream::in))
  {
    CUDA_TRY(cudaEventCreate(&prev_host_copy_event));  //
  }

  ~file_data_chunk_reader()
  {
    CUDA_TRY(cudaEventDestroy(prev_host_copy_event));  //
  }

  data_chunk get_next_chunk(uint32_t read_size, rmm::cuda_stream_view stream) override
  {
    CUDA_TRY(cudaEventSynchronize(prev_host_copy_event));

    if (_host_buffer.size() < read_size) { _host_buffer.resize(read_size); }

    _filestream.read(_host_buffer.data(), read_size);

    read_size = _filestream.gcount();

    auto chunk_buffer = rmm::device_buffer(read_size, stream);

    CUDA_TRY(cudaMemcpyAsync(  //
      chunk_buffer.data(),
      _host_buffer.data(),
      read_size,
      cudaMemcpyHostToDevice,
      stream.value()));

    CUDA_TRY(cudaEventRecord(prev_host_copy_event, stream.value()));

    return data_chunk(std::move(chunk_buffer), read_size);
  }

 private:
  cudaEvent_t prev_host_copy_event;
  std::ifstream _filestream;
  thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>>
    _host_buffer{};
};

class file_data_chunk_source : public data_chunk_source {
 public:
  file_data_chunk_source(std::string filename) : _filename(filename) {}
  std::unique_ptr<data_chunk_reader> create_reader() override
  {
    return std::make_unique<file_data_chunk_reader>(_filename);
  }

 private:
  std::string _filename;
};

}  // namespace

std::unique_ptr<data_chunk_source> make_source(std::string& data);
std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data);
std::unique_ptr<data_chunk_source> make_source_from_file(std::string filename)
{
  return std::make_unique<file_data_chunk_source>(filename);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
