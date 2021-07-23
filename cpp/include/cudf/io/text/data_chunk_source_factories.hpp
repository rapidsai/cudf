#pragma once

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace cudf {
namespace io {
namespace text {

namespace {

class istream_data_chunk_reader : public data_chunk_reader {
 public:
  istream_data_chunk_reader(std::unique_ptr<std::istream> datastream)
    : _datastream(std::move(datastream)), _buffers()
  {
    CUDA_TRY(cudaEventCreate(&prev_host_copy_event));  //
  }

  ~istream_data_chunk_reader()
  {
    CUDA_TRY(cudaEventDestroy(prev_host_copy_event));  //
  }

  device_span<char> find_or_create_data(uint32_t size, rmm::cuda_stream_view stream)
  {
    auto search = _buffers.find(stream.value());

    if (search == _buffers.end() || search->second.size() < size) {
      _buffers[stream.value()] = rmm::device_buffer(size, stream);
    }

    return device_span<char>(static_cast<char*>(_buffers[stream.value()].data()), size);
  }

  data_chunk get_next_chunk(uint32_t read_size, rmm::cuda_stream_view stream) override
  {
    CUDF_FUNC_RANGE();
    CUDA_TRY(cudaEventSynchronize(prev_host_copy_event));

    if (_host_buffer.size() < read_size) { _host_buffer.resize(read_size); }

    _datastream->read(_host_buffer.data(), read_size);

    read_size = _datastream->gcount();

    auto chunk_span = find_or_create_data(read_size, stream);

    CUDA_TRY(cudaMemcpyAsync(  //
      chunk_span.data(),
      _host_buffer.data(),
      read_size,
      cudaMemcpyHostToDevice,
      stream.value()));

    CUDA_TRY(cudaEventRecord(prev_host_copy_event, stream.value()));

    return data_chunk(chunk_span);
  }

 private:
  std::unique_ptr<std::istream> _datastream;
  std::unordered_map<cudaStream_t, rmm::device_buffer> _buffers;
  cudaEvent_t prev_host_copy_event;
  thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>>
    _host_buffer{};
};

class device_span_data_chunk_reader : public data_chunk_reader {
 public:
  device_span_data_chunk_reader(device_span<char const> data) : _data(data) {}

  data_chunk get_next_chunk(uint32_t read_size, rmm::cuda_stream_view stream) override
  {
    if (read_size > _data.size() - _position) { read_size = _data.size() - _position; }

    auto chunk_span = _data.subspan(_position, read_size);

    _position += read_size;

    return data_chunk(chunk_span);
  }

 private:
  device_span<char const> _data;
  uint64_t _position = 0;
};

class file_data_chunk_source : public data_chunk_source {
 public:
  file_data_chunk_source(std::string filename) : _filename(filename) {}
  std::unique_ptr<data_chunk_reader> create_reader() override
  {
    return std::make_unique<istream_data_chunk_reader>(
      std::make_unique<std::ifstream>(_filename, std::ifstream::in));
  }

 private:
  std::string _filename;
};

class string_data_chunk_source : public data_chunk_source {
 public:
  string_data_chunk_source(std::string const& data) : _data(data) {}
  std::unique_ptr<data_chunk_reader> create_reader() override
  {
    return std::make_unique<istream_data_chunk_reader>(std::make_unique<std::istringstream>(_data));
  }

 private:
  std::string const& _data;
};

class device_span_data_chunk_source : public data_chunk_source {
 public:
  device_span_data_chunk_source(device_span<char const> data) : _data(data) {}
  std::unique_ptr<data_chunk_reader> create_reader() override
  {
    return std::make_unique<device_span_data_chunk_reader>(_data);
  }

 private:
  device_span<char const> _data;
};

}  // namespace

std::unique_ptr<data_chunk_source> make_source(std::string const& data)
{
  return std::make_unique<string_data_chunk_source>(data);
}

std::unique_ptr<data_chunk_source> make_source_from_file(std::string const& filename)
{
  return std::make_unique<file_data_chunk_source>(filename);
}

std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data)
{
  auto data_span = device_span<char const>(data.data(), data.size());
  return std::make_unique<device_span_data_chunk_source>(data_span);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
