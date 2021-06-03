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

#include <thrust/host_vector.h>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <memory>

namespace cudf {
namespace io {

class file_destination : public data_destination {
 public:
  file_destination(std::string filepath) : _cufile_out(detail::make_cufile_output(filepath))
  {
    _output_stream.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(_output_stream.is_open(), "Cannot open output file");
  }

  void flush() override { _output_stream.flush(); }

  void write(cudf::host_span<char const> data, rmm::cuda_stream_view stream) override
  {
    _output_stream.seekp(_bytes_written);
    _output_stream.write(data.data(), data.size());
    _bytes_written += data.size();
  };

  void write(cudf::device_span<char const> data, rmm::cuda_stream_view stream) override
  {
    if (_cufile_out != nullptr && _cufile_out->is_cufile_io_preferred(data.size())) {
      _cufile_out->write(data.data(), _bytes_written, data.size());
      _bytes_written += data.size();
      return;
    }

    if (_host_buffer.size() < data.size()) { _host_buffer.resize(data.size()); }

    CUDA_TRY(cudaMemcpyAsync(
      _host_buffer.data(), data.data(), data.size(), cudaMemcpyDeviceToHost, stream.value()));

    stream.synchronize();  // why even use cudaMemcpyAsync instead of cudaMemcpy?

    write(cudf::host_span<char>(_host_buffer.data(), data.size()), stream);
  };

  size_t bytes_written() const override { return _bytes_written; }

 private:
  std::ofstream _output_stream;
  size_t _bytes_written = 0;
  std::unique_ptr<detail::cufile_output_impl> _cufile_out;
  rmm::cuda_stream_view _stream;
  thrust::host_vector<char> _host_buffer;
};

std::unique_ptr<data_destination> create_file_destination(std::string const& filepath)
{
  return std::make_unique<file_destination>(filepath);
}

}  // namespace io
}  // namespace cudf
