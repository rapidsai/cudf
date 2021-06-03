/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "../file_io_utilities.hpp"

#include <cudf/io/data_destination.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <fstream>
#include <memory>

namespace cudf {
namespace io {

class file_destination_writer : public data_destination_writer {
 public:
  file_destination_writer(std::string filepath, rmm::cuda_stream_view stream)
    : _cufile_out(detail::make_cufile_output(filepath)), _stream(stream)
  {
    _output_stream.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(_output_stream.is_open(), "Cannot open output file");
  }

  ~file_destination_writer() { _output_stream.flush(); }

  void write(cudf::host_span<char const> data)
  {
    _output_stream.seekp(_bytes_written);
    _output_stream.write(data.data(), data.size());
    _bytes_written += data.size();
  };

  void write(cudf::device_span<char const> data)
  {
    if (_cufile_out != nullptr && _cufile_out->is_cufile_io_preferred(data.size())) {
      _cufile_out->write(data.data(), _bytes_written, data.size());
      _bytes_written += data.size();
      return;
    }

    grow_host_buffer(data.size());

    CUDA_TRY(cudaMemcpyAsync(_host_buffer,
                             data.data(),
                             data.size() * sizeof(char),
                             cudaMemcpyDeviceToHost,
                             _stream.value()));

    _stream.synchronize();

    write(cudf::host_span<char>(_host_buffer, data.size()));
  };

 private:
  void grow_host_buffer(size_type min_size)
  {
    if (_host_buffer_size >= min_size) {
      return;  //
    }

    if (_host_buffer != nullptr) {
      cudaFreeHost(_host_buffer);  //
    }

    // optionally replace cudaMallocHost to specify flags with cudaHostAlloc
    cudaMallocHost(reinterpret_cast<void**>(&_host_buffer), min_size);
    _host_buffer_size = min_size;
  }

  std::ofstream _output_stream;
  size_t _bytes_written = 0;
  std::unique_ptr<detail::cufile_output_impl> _cufile_out;
  rmm::cuda_stream_view _stream;
  char* _host_buffer;
  size_type _host_buffer_size;
};

class file_destination : public data_destination {
 public:
  file_destination(std::string const& filepath) : _filepath(filepath) {}

  std::unique_ptr<data_destination_writer> create_writer(rmm::cuda_stream_view stream)
  {
    return std::make_unique<file_destination_writer>(_filepath, stream);
  }

 private:
  std::string _filepath;
};

std::unique_ptr<data_destination> create_file_destination(std::string const& filepath)
{
  return std::make_unique<file_destination>(filepath);
}

}  // namespace io
}  // namespace cudf
