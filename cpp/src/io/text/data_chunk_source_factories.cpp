/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "io/text/device_data_chunks.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/pinned_host_vector.hpp>
#include <cudf/io/text/data_chunk_source_factories.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>

#include <fstream>

namespace cudf::io::text {

namespace {

struct host_ticket {
  cudaEvent_t event;
  cudf::detail::pinned_host_vector<char> buffer;
};

/**
 * @brief A reader which produces owning chunks of device memory which contain a copy of the data
 * from an istream.
 */
class datasource_chunk_reader : public data_chunk_reader {
  constexpr static int num_tickets = 2;

 public:
  datasource_chunk_reader(datasource* source) : _source(source)
  {
    // create an event to track the completion of the last device-to-host copy.
    for (auto& ticket : _tickets) {
      CUDF_CUDA_TRY(cudaEventCreate(&(ticket.event)));
    }
  }

  ~datasource_chunk_reader() override
  {
    for (auto& ticket : _tickets) {
      CUDF_CUDA_TRY(cudaEventDestroy(ticket.event));
    }
  }

  void skip_bytes(std::size_t size) override
  {
    _offset += std::min(_source->size() - _offset, size);
  };

  std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t read_size,
                                                    rmm::cuda_stream_view stream) override
  {
    CUDF_FUNC_RANGE();

    read_size = std::min(_source->size() - _offset, read_size);

    // get a device buffer containing read data on the device.
    auto chunk = rmm::device_uvector<char>(read_size, stream);

    if (_source->supports_device_read() && _source->is_device_read_preferred(read_size)) {
      _source->device_read(_offset, read_size, reinterpret_cast<uint8_t*>(chunk.data()), stream);
    } else {
      auto& h_ticket = _tickets[_next_ticket_idx];

      _next_ticket_idx = (_next_ticket_idx + 1) % num_tickets;

      // synchronize on the last host-to-device copy, so we don't clobber the host buffer.
      CUDF_CUDA_TRY(cudaEventSynchronize(h_ticket.event));

      // resize the host buffer as necessary to contain the requested number of bytes
      if (h_ticket.buffer.size() < read_size) { h_ticket.buffer.resize(read_size); }

      _source->host_read(_offset, read_size, reinterpret_cast<uint8_t*>(h_ticket.buffer.data()));

      // copy the host-pinned data on to device
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk.data(), h_ticket.buffer.data(), read_size, cudaMemcpyDefault, stream.value()));

      // record the host-to-device copy.
      CUDF_CUDA_TRY(cudaEventRecord(h_ticket.event, stream.value()));
    }

    _offset += read_size;

    // return the device buffer so it can be processed.
    return std::make_unique<device_uvector_data_chunk>(std::move(chunk));
  }

 private:
  std::size_t _offset          = 0;
  std::size_t _next_ticket_idx = 0;
  std::array<host_ticket, num_tickets> _tickets{};
  datasource* _source;
};

/**
 * @brief A reader which produces owning chunks of device memory which contain a copy of the data
 * from an istream.
 */
class istream_data_chunk_reader : public data_chunk_reader {
  constexpr static int num_tickets = 2;

 public:
  istream_data_chunk_reader(std::unique_ptr<std::istream> datastream)
    : _datastream(std::move(datastream))
  {
    // create an event to track the completion of the last device-to-host copy.
    for (auto& ticket : _tickets) {
      CUDF_CUDA_TRY(cudaEventCreate(&(ticket.event)));
    }
  }

  ~istream_data_chunk_reader() override
  {
    for (auto& ticket : _tickets) {
      CUDF_CUDA_TRY(cudaEventDestroy(ticket.event));
    }
  }

  void skip_bytes(std::size_t size) override { _datastream->ignore(size); };

  std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t read_size,
                                                    rmm::cuda_stream_view stream) override
  {
    CUDF_FUNC_RANGE();

    auto& h_ticket = _tickets[_next_ticket_idx];

    _next_ticket_idx = (_next_ticket_idx + 1) % num_tickets;

    // synchronize on the last host-to-device copy, so we don't clobber the host buffer.
    CUDF_CUDA_TRY(cudaEventSynchronize(h_ticket.event));

    // resize the host buffer as necessary to contain the requested number of bytes
    if (h_ticket.buffer.size() < read_size) { h_ticket.buffer.resize(read_size); }

    // read data from the host istream in to the pinned host memory buffer
    _datastream->read(h_ticket.buffer.data(), read_size);

    // adjust the read size to reflect how many bytes were actually read from the data stream
    read_size = _datastream->gcount();

    // get a device buffer containing read data on the device.
    auto chunk = rmm::device_uvector<char>(read_size, stream);

    // copy the host-pinned data on to device
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      chunk.data(), h_ticket.buffer.data(), read_size, cudaMemcpyDefault, stream.value()));

    // record the host-to-device copy.
    CUDF_CUDA_TRY(cudaEventRecord(h_ticket.event, stream.value()));

    // return the device buffer so it can be processed.
    return std::make_unique<device_uvector_data_chunk>(std::move(chunk));
  }

 private:
  std::size_t _next_ticket_idx = 0;
  std::array<host_ticket, num_tickets> _tickets{};
  std::unique_ptr<std::istream> _datastream;
};

/**
 * @brief A reader which produces owning chunks of device memory which contain a copy of the data
 * from a host span.
 */
class host_span_data_chunk_reader : public data_chunk_reader {
 public:
  host_span_data_chunk_reader(cudf::host_span<char const> data) : _data(data) {}

  void skip_bytes(std::size_t read_size) override
  {
    _position += std::min(read_size, _data.size() - _position);
  }

  std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t read_size,
                                                    rmm::cuda_stream_view stream) override
  {
    CUDF_FUNC_RANGE();

    read_size = std::min(read_size, _data.size() - _position);

    // get a device buffer containing read data on the device.
    auto chunk = rmm::device_uvector<char>(read_size, stream);

    // copy the host data to device
    CUDF_CUDA_TRY(cudaMemcpyAsync(  //
      chunk.data(),
      _data.data() + _position,
      read_size,
      cudaMemcpyDefault,
      stream.value()));

    _position += read_size;

    // return the device buffer so it can be processed.
    return std::make_unique<device_uvector_data_chunk>(std::move(chunk));
  }

 private:
  std::size_t _position = 0;
  cudf::host_span<char const> _data;
};

/**
 * @brief A reader which produces view of device memory which represent a subset of the input device
 * span.
 */
class device_span_data_chunk_reader : public data_chunk_reader {
 public:
  device_span_data_chunk_reader(device_span<char const> data) : _data(data) {}

  void skip_bytes(std::size_t read_size) override
  {
    _position += std::min(read_size, _data.size() - _position);
  }

  std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t read_size,
                                                    rmm::cuda_stream_view stream) override
  {
    // limit the read size to the number of bytes remaining in the device_span.
    read_size = std::min(read_size, _data.size() - _position);

    // create a view over the device span
    auto chunk_span = _data.subspan(_position, read_size);

    // increment position
    _position += read_size;

    // return the view over device memory so it can be processed.
    return std::make_unique<device_span_data_chunk>(chunk_span);
  }

 private:
  device_span<char const> _data;
  uint64_t _position = 0;
};

/**
 * @brief A datasource-based data chunk source which creates a datasource_chunk_reader.
 */
class datasource_chunk_source : public data_chunk_source {
 public:
  datasource_chunk_source(datasource& source) : _source(&source) {}
  [[nodiscard]] std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<datasource_chunk_reader>(_source);
  }

 private:
  datasource* _source;
};

/**
 * @brief A file data source which creates an istream_data_chunk_reader.
 */
class file_data_chunk_source : public data_chunk_source {
 public:
  file_data_chunk_source(std::string_view filename) : _filename(filename) {}
  [[nodiscard]] std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<istream_data_chunk_reader>(
      std::make_unique<std::ifstream>(_filename, std::ifstream::in));
  }

 private:
  std::string _filename;
};

/**
 * @brief A host string data source which creates an host_span_data_chunk_reader.
 */
class host_span_data_chunk_source : public data_chunk_source {
 public:
  host_span_data_chunk_source(host_span<char const> data) : _data(data) {}
  [[nodiscard]] std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<host_span_data_chunk_reader>(_data);
  }

 private:
  host_span<char const> _data;
};

/**
 * @brief A device span data source which creates an istream_data_chunk_reader.
 */
class device_span_data_chunk_source : public data_chunk_source {
 public:
  device_span_data_chunk_source(device_span<char const> data) : _data(data) {}
  [[nodiscard]] std::unique_ptr<data_chunk_reader> create_reader() const override
  {
    return std::make_unique<device_span_data_chunk_reader>(_data);
  }

 private:
  device_span<char const> _data;
};

}  // namespace

std::unique_ptr<data_chunk_source> make_source(datasource& data)
{
  return std::make_unique<datasource_chunk_source>(data);
}

std::unique_ptr<data_chunk_source> make_source(host_span<char const> data)
{
  return std::make_unique<host_span_data_chunk_source>(data);
}

std::unique_ptr<data_chunk_source> make_source_from_file(std::string_view filename)
{
  return std::make_unique<file_data_chunk_source>(filename);
}

std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data)
{
  auto data_span = device_span<char const>(data.data(), data.size());
  return std::make_unique<device_span_data_chunk_source>(data_span);
}

}  // namespace cudf::io::text
