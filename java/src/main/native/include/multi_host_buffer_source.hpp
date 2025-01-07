/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "jni_utils.hpp"

#include <cudf/io/datasource.hpp>

#include <vector>

namespace cudf {
namespace jni {

/**
 * @brief A custom datasource providing data from an array of host memory buffers.
 */
class multi_host_buffer_source : public cudf::io::datasource {
  std::vector<uint8_t const*> addrs_;
  std::vector<size_t> offsets_;

  size_t locate_offset_index(size_t offset);

 public:
  explicit multi_host_buffer_source(native_jlongArray const& addrs_sizes);
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override;
  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;
  bool supports_device_read() const override { return true; }
  bool is_device_read_preferred(size_t size) const override { return true; }
  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override;
  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;
  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;
  size_t size() const override { return offsets_.back(); }
};

}  // namespace jni
}  // namespace cudf
