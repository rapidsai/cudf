/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>

#include <future>
#include <vector>

namespace cudf::test {

/**
 * @brief Custom exception for device read async testing
 */
class AsyncException : public std::exception {};

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
    std::vector<std::byte> byte_data(size);
    std::memcpy(byte_data.data(), data_.data() + offset, size);
    return cudf::io::datasource::buffer::create(std::move(byte_data));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = std::min(size, data_.size() - offset);
    std::memcpy(dst, data_.data() + offset, read_size);
    return read_size;
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::unique_ptr<cudf::io::datasource::buffer> device_read(size_t offset,
                                                            size_t size,
                                                            rmm::cuda_stream_view stream) override
  {
    // For testing, just copy the data from the host buffer into a new buffer
    size = std::min(size, data_.size() - offset);
    rmm::device_buffer out_data(size, stream);
    cudaMemcpyAsync(
      out_data.data(), data_.data() + offset, size, cudaMemcpyHostToDevice, stream.value());
    cudaStreamSynchronize(stream.value());
    return cudf::io::datasource::buffer::create(std::move(out_data));
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    // This datasource returns a future that throws a custom exception when accessed for testing
    std::promise<size_t> promise;
    promise.set_exception(std::make_exception_ptr(AsyncException()));
    return promise.get_future();
  }

  [[nodiscard]] size_t size() const override { return data_.size(); }
};

/**
 * @brief Data sink that throws an exception in device_write_async for testing
 */
class ThrowingDeviceWriteDataSink : public cudf::io::data_sink {
 private:
  size_t buffer_size_ = 0;

 public:
  void host_write(void const* data, size_t size) override { buffer_size_ += size; }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    buffer_size_ += size;
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    // This data sink returns a future that throws a custom exception when accessed for testing
    std::promise<void> promise;
    promise.set_exception(std::make_exception_ptr(AsyncException()));
    return promise.get_future();
  }

  void flush() override {}

  size_t bytes_written() override { return buffer_size_; }
};

}  // namespace cudf::test
