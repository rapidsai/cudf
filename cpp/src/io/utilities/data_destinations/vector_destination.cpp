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
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace io {

class vector_destination_writer : public data_destination_writer {
 public:
  vector_destination_writer(std::vector<char>* buffer, rmm::cuda_stream_view stream)
    : _buffer(buffer), _stream(stream)
  {
  }

  void write(cudf::host_span<uint8_t> data)
  {
    auto char_array = reinterpret_cast<char const*>(data.data());
    _buffer->insert(_buffer->end(), char_array, char_array + data.size());
  };

  void write(cudf::device_span<uint8_t> data)
  {
    _buffer->resize(_buffer->size() + data.size());

    RMM_CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<void*>(_buffer->data()),  //
                                 data.data(),
                                 data.size(),
                                 cudaMemcpyDeviceToHost,
                                 _stream.value()));
  };

 private:
  std::vector<char>* _buffer;
  rmm::cuda_stream_view _stream;
};

class vector_destination : public data_destination {
 public:
  vector_destination(std::vector<char>* buffer) : _buffer(buffer) {}

  static std::unique_ptr<data_destination> create(std::vector<char>* buffer)
  {
    return std::make_unique<vector_destination>(buffer);
  }

  std::unique_ptr<data_destination_writer> create_writer(rmm::cuda_stream_view stream)
  {
    return std::make_unique<vector_destination_writer>(_buffer, stream);
  }

 private:
  std::vector<char>* _buffer;
};

}  // namespace io
}  // namespace cudf
