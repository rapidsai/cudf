/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

/**
 * @brief a reader which produces views of device memory which contain a copy of the data from an
 * istream.
 *
 */
class istream_data_chunk_reader : public data_chunk_reader {
  struct host_ticket {
    cudaEvent_t event;
    thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>> buffer;
  };

 public:
  istream_data_chunk_reader(std::unique_ptr<std::istream> datastream)
    : _datastream(std::move(datastream)), _buffers(), _tickets(2)
  {
    // create an event to track the completion of the last device-to-host copy.
    for (uint32_t i = 0; i < _tickets.size(); i++) {
      CUDA_TRY(cudaEventCreate(&(_tickets[i].event)));
    }
  }

  ~istream_data_chunk_reader()
  {
    for (uint32_t i = 0; i < _tickets.size(); i++) {
      CUDA_TRY(cudaEventDestroy(_tickets[i].event));
    }
  }

  device_span<char> find_or_create_data(uint32_t size, rmm::cuda_stream_view stream)
  {
    auto search = _buffers.find(stream.value());

    if (search == _buffers.end() || search->second.size() < size) {
      _buffers[stream.value()] = rmm::device_buffer(size, stream);
    }

    return device_span<char>(static_cast<char*>(_buffers[stream.value()].data()), size);
  }

  device_span<char const> get_next_chunk(uint32_t read_size, rmm::cuda_stream_view stream) override
  {
    CUDF_FUNC_RANGE();

    auto& ticket = _tickets[_next_ticket_idx];

    _next_ticket_idx = (_next_ticket_idx + 1) % _tickets.size();

    // synchronize on the last host-to-device copy, so we don't clobber the host buffer.
    CUDA_TRY(cudaEventSynchronize(ticket.event));

    // resize the host buffer as necessary to contain the requested number of bytes
    if (ticket.buffer.size() < read_size) { ticket.buffer.resize(read_size); }

    // read data from the host istream in to the pinned host memory buffer
    _datastream->read(ticket.buffer.data(), read_size);

    // adjust the read size to reflect how many bytes were actually read from the data stream
    read_size = _datastream->gcount();

    // get a view over some device memory we can use to buffer the read data on to device.
    auto chunk_span = find_or_create_data(read_size, stream);

    // copy the host-pinned data on to device
    CUDA_TRY(cudaMemcpyAsync(  //
      chunk_span.data(),
      ticket.buffer.data(),
      read_size,
      cudaMemcpyHostToDevice,
      stream.value()));

    // record the host-to-device copy.
    CUDA_TRY(cudaEventRecord(ticket.event, stream.value()));

    // return the view over device memory so it can be processed.
    return chunk_span;
  }

 private:
  uint32_t _next_ticket_idx = 0;
  std::unique_ptr<std::istream> _datastream;
  std::unordered_map<cudaStream_t, rmm::device_buffer> _buffers;
  std::vector<host_ticket> _tickets;
};

/**
 * @brief a reader which produces view of device memory which represent a subset of the input device
 * span
 *
 */
class device_span_data_chunk_reader : public data_chunk_reader {
 public:
  device_span_data_chunk_reader(device_span<char const> data) : _data(data) {}

  device_span<char const> get_next_chunk(uint32_t read_size, rmm::cuda_stream_view stream) override
  {
    // limit the read size to the number of bytes remaining in the device_span.
    if (read_size > _data.size() - _position) { read_size = _data.size() - _position; }

    // create a view over the device span
    auto chunk_span = _data.subspan(_position, read_size);

    // increment position
    _position += read_size;

    // return the view over device memory so it can be processed.
    return chunk_span;
  }

 private:
  device_span<char const> _data;
  uint64_t _position = 0;
};

/**
 * @brief a file data source which creates an istream_data_chunk_reader
 *
 */
class file_data_chunk_source : public data_chunk_source {
 public:
  file_data_chunk_source(std::string filename) : _filename(filename) {}
  std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<istream_data_chunk_reader>(
      std::make_unique<std::ifstream>(_filename, std::ifstream::in));
  }

 private:
  std::string _filename;
};

/**
 * @brief a host string data source which creates an istream_data_chunk_reader
 */
class string_data_chunk_source : public data_chunk_source {
 public:
  string_data_chunk_source(std::string const& data) : _data(data) {}
  std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<istream_data_chunk_reader>(std::make_unique<std::istringstream>(_data));
  }

 private:
  std::string const& _data;
};

/**
 * @brief a device span data source which creates an istream_data_chunk_reader
 */
class device_span_data_chunk_source : public data_chunk_source {
 public:
  device_span_data_chunk_source(device_span<char const> data) : _data(data) {}
  std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<device_span_data_chunk_reader>(_data);
  }

 private:
  device_span<char const> _data;
};

}  // namespace

/**
 * @brief Creates a data source capable of producing device-buffered views of the given string.
 */
std::unique_ptr<data_chunk_source> make_source(std::string const& data)
{
  return std::make_unique<string_data_chunk_source>(data);
}

/**
 * @brief Creates a data source capable of producing device-buffered views of the file
 */
std::unique_ptr<data_chunk_source> make_source_from_file(std::string const& filename)
{
  return std::make_unique<file_data_chunk_source>(filename);
}

/**
 * @brief Creates a data source capable of producing views of the given device string scalar
 */
std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data)
{
  auto data_span = device_span<char const>(data.data(), data.size());
  return std::make_unique<device_span_data_chunk_source>(data_span);
}

}  // namespace text
}  // namespace io
}  // namespace cudf
