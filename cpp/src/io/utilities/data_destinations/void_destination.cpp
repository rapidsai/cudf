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

class void_destination : public data_destination {
 public:
  void write(cudf::host_span<char const> data, rmm::cuda_stream_view stream)
  {
    _bytes_written += data.size();
  };
  void write(cudf::device_span<char const> data, rmm::cuda_stream_view stream)
  {
    _bytes_written += data.size();
  };

  size_t bytes_written() const override { return _bytes_written; }

 private:
  size_t _bytes_written = 0;
};

std::unique_ptr<data_destination> create_void_destination()
{
  return std::make_unique<void_destination>();
}

}  // namespace io
}  // namespace cudf
