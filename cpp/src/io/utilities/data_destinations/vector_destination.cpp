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

#include <cudf/io/data_destination.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <memory>

namespace cudf {
namespace io {

class vector_destination : public data_destination {
 public:
  vector_destination(std::vector<char>* buffer) : _buffer(buffer) {}

  void write(cudf::host_span<char const> data, rmm::cuda_stream_view stream)
  {
    _buffer->insert(_buffer->end(), data.data(), data.data() + data.size());
  };

  void write(cudf::device_span<char const> data, rmm::cuda_stream_view stream)
  {
    if (_host_buffer.size() < data.size()) { _host_buffer.resize(data.size()); }

    CUDA_TRY(cudaMemcpyAsync(
      _host_buffer.data(), data.data(), data.size(), cudaMemcpyDeviceToHost, stream.value()));

    stream.synchronize();  // why even use cudaMemcpyAsync instead of cudaMemcpy?

    write(cudf::host_span<char>(_host_buffer.data(), data.size()), stream);
  };

  size_t bytes_written() const override { return _buffer->size(); }

 private:
  std::vector<char>* _buffer;
  thrust::host_vector<char> _host_buffer;
};

std::unique_ptr<data_destination> create_vector_destination(std::vector<char>* buffer)
{
  return std::make_unique<vector_destination>(buffer);
}

}  // namespace io
}  // namespace cudf
