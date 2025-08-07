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

#include <cudf/io/datasource.hpp>

#include <future>
#include <vector>

namespace cudf::test {

/**
 * @brief Custom exception for device read async testing
 */
class DeviceReadAsyncException : public std::exception {};

/**
 * @brief Datasource that throws an exception in device_read_async for testing
 */
class ThrowingDeviceReadDatasource : public cudf::io::datasource {
 private:
  std::vector<char> const& data_;

 public:
  explicit ThrowingDeviceReadDatasource(std::vector<char> const& data) : data_(data) {}

  std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset, size_t size) override
  {
    size = std::min(size, data_.size() - offset);
    // Convert char data to bytes for the buffer
    std::vector<std::byte> byte_data;
    byte_data.reserve(size);
    std::transform(data_.begin() + offset,
                   data_.begin() + offset + size,
                   std::back_inserter(byte_data),
                   [](char c) { return static_cast<std::byte>(c); });
    return cudf::io::datasource::buffer::create(std::move(byte_data));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, data_.size() - offset);
    std::memcpy(dst, data_.data() + offset, read_size);
    return read_size;
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    // This datasource returns a future that throws a custom exception when accessed for testing
    std::promise<size_t> promise;
    promise.set_exception(std::make_exception_ptr(DeviceReadAsyncException()));
    return promise.get_future();
  }

  [[nodiscard]] size_t size() const override { return data_.size(); }
};

}  // namespace cudf::test