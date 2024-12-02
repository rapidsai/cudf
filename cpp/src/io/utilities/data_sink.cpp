/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "file_io_utilities.hpp"

#include <cudf/detail/utilities/logger.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/file_handle.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <fstream>

namespace cudf {
namespace io {

/**
 * @brief Implementation class for storing data into a local file.
 */
class file_sink : public data_sink {
 public:
  explicit file_sink(std::string const& filepath)
  {
    detail::force_init_cuda_context();
    _output_stream.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!_output_stream.is_open()) { detail::throw_on_file_open_failure(filepath, true); }

    if (cufile_integration::is_kvikio_enabled()) {
      cufile_integration::set_up_kvikio();
      _kvikio_file = kvikio::FileHandle(filepath, "w");
      CUDF_LOG_INFO("Writing a file using kvikIO, with compatibility mode {}.",
                    _kvikio_file.is_compat_mode_preferred() ? "on" : "off");
    } else {
      _cufile_out = detail::make_cufile_output(filepath);
    }
  }

  // Marked as NOLINT because we are calling a virtual method in the destructor
  ~file_sink() override { flush(); }  // NOLINT

  void host_write(void const* data, size_t size) override
  {
    _output_stream.seekp(_bytes_written);
    _output_stream.write(static_cast<char const*>(data), size);
    _bytes_written += size;
  }

  void flush() override { _output_stream.flush(); }

  size_t bytes_written() override { return _bytes_written; }

  [[nodiscard]] bool supports_device_write() const override
  {
    return !_kvikio_file.closed() || _cufile_out != nullptr;
  }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override
  {
    if (!supports_device_write()) { return false; }

    // Always prefer device writes if kvikio is enabled
    if (!_kvikio_file.closed()) { return true; }

    return size >= _gds_write_preferred_threshold;
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    if (!supports_device_write()) CUDF_FAIL("Device writes are not supported for this file.");

    size_t offset = _bytes_written;
    _bytes_written += size;

    if (!_kvikio_file.closed()) {
      // KvikIO's `pwrite()` returns a `std::future<size_t>` so we convert it
      // to `std::future<void>`
      return std::async(std::launch::deferred, [this, gpu_data, size, offset] {
        _kvikio_file.pwrite(gpu_data, size, offset).get();
      });
    }
    return _cufile_out->write_async(gpu_data, offset, size);
  }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    return device_write_async(gpu_data, size, stream).get();
  }

 private:
  std::ofstream _output_stream;
  size_t _bytes_written = 0;
  std::unique_ptr<detail::cufile_output_impl> _cufile_out;
  kvikio::FileHandle _kvikio_file;
  // The write size above which GDS is faster then d2h-copy + posix-write
  static constexpr size_t _gds_write_preferred_threshold = 128 << 10;  // 128KB
};

/**
 * @brief Implementation class for storing data into a std::vector.
 */
class host_buffer_sink : public data_sink {
 public:
  explicit host_buffer_sink(std::vector<char>* buffer) : buffer_(buffer) {}

  // Marked as NOLINT because we are calling a virtual method in the destructor
  ~host_buffer_sink() override { flush(); }  // NOLINT

  void host_write(void const* data, size_t size) override
  {
    auto char_array = static_cast<char const*>(data);
    buffer_->insert(buffer_->end(), char_array, char_array + size);
  }

  void flush() override {}

  size_t bytes_written() override { return buffer_->size(); }

 private:
  std::vector<char>* buffer_;
};

/**
 * @brief Implementation class for voiding data (no io performed)
 */
class void_sink : public data_sink {
 public:
  explicit void_sink() {}

  ~void_sink() override {}

  void host_write(void const* data, size_t size) override { _bytes_written += size; }

  [[nodiscard]] bool supports_device_write() const override { return true; }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    _bytes_written += size;
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    _bytes_written += size;
    return std::async(std::launch::deferred, [] {});
  }

  void flush() override {}

  size_t bytes_written() override { return _bytes_written; }

 private:
  size_t _bytes_written;
};

class user_sink_wrapper : public data_sink {
 public:
  explicit user_sink_wrapper(cudf::io::data_sink* const user_sink_) : user_sink(user_sink_) {}

  ~user_sink_wrapper() override {}

  void host_write(void const* data, size_t size) override { user_sink->host_write(data, size); }

  [[nodiscard]] bool supports_device_write() const override
  {
    return user_sink->supports_device_write();
  }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(user_sink->supports_device_write(),
                 "device_write() was called on a data_sink that doesn't support it");
    user_sink->device_write(gpu_data, size, stream);
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(user_sink->supports_device_write(),
                 "device_write_async() was called on a data_sink that doesn't support it");
    return user_sink->device_write_async(gpu_data, size, stream);
  }

  [[nodiscard]] bool is_device_write_preferred(size_t size) const override
  {
    return user_sink->is_device_write_preferred(size);
  }

  void flush() override { user_sink->flush(); }

  size_t bytes_written() override { return user_sink->bytes_written(); }

 private:
  cudf::io::data_sink* const user_sink;
};

std::unique_ptr<data_sink> data_sink::create(std::string const& filepath)
{
  return std::make_unique<file_sink>(filepath);
}

std::unique_ptr<data_sink> data_sink::create(std::vector<char>* buffer)
{
  return std::make_unique<host_buffer_sink>(buffer);
}

std::unique_ptr<data_sink> data_sink::create() { return std::make_unique<void_sink>(); }

std::unique_ptr<data_sink> data_sink::create(cudf::io::data_sink* const user_sink)
{
  return std::make_unique<user_sink_wrapper>(user_sink);
}

}  // namespace io
}  // namespace cudf
