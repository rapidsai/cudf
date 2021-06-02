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

class void_destination_writer : public data_destination_writer {
 public:
  void write(cudf::host_span<uint8_t> data){};
  void write(cudf::device_span<uint8_t> data){};
};

class void_destination : public data_destination {
 public:
  void_destination() {}

  static std::unique_ptr<data_destination> create() { return std::make_unique<void_destination>(); }

  std::unique_ptr<data_destination_writer> create_writer(rmm::cuda_stream_view)
  {
    return std::make_unique<void_destination_writer>();
  }
};

}  // namespace io
}  // namespace cudf
