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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace io {

class data_destination_writer {
 public:
  virtual ~data_destination_writer()                  = 0;
  virtual void write(cudf::host_span<uint8_t> data)   = 0;
  virtual void write(cudf::device_span<uint8_t> data) = 0;
};

class data_destination {
 public:
  virtual ~data_destination()                                                           = 0;
  virtual std::unique_ptr<data_destination_writer> create_writer(rmm::cuda_stream_view) = 0;
};

}  // namespace io
}  // namespace cudf
