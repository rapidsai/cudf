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

#include <fstream>

#include <cudf/io/data_sink.hpp>
#include <cudf/utilities/error.hpp>
#include <io/utilities/file_io_utilities.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
/**
 * @brief Implementation class for storing data into a local file.
 */
class file_sink : public data_sink {
 public:
  explicit file_sink(std::string const& filepath)
    : _cufile_out(detail::make_cufile_output(filepath))
  {
    _output_stream.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(_output_stream.is_open(), "Cannot open output file");
  }

  virtual ~file_sink() { flush(); }

  void host_write(void const* data, size_t size) override
  {
    _output_stream.seekp(_bytes_written);
    _output_stream.write(static_cast<char const*>(data), size);
    _bytes_written += size;
  }

  void flush() override { _output_stream.flush(); }

  size_t bytes_written() override { return _bytes_written; }

  bool supports_device_write() const override { return _cufile_out != nullptr; }

  bool is_device_write_preferred(size_t size) const override
  {
    return _cufile_out != nullptr && _cufile_out->is_cufile_io_preferred(size);
  }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    if (!supports_device_write()) CUDF_FAIL("Device writes are not supported for this file.");

    _cufile_out->write(gpu_data, _bytes_written, size);
    _bytes_written += size;
  }

 private:
  std::ofstream _output_stream;
  size_t _bytes_written = 0;
  std::unique_ptr<detail::cufile_output_impl> _cufile_out;
};

/**
 * @brief Implementation class for storing data into a std::vector.
 */
class host_buffer_sink : public data_sink {
 public:
  explicit host_buffer_sink(std::vector<char>* buffer) : buffer_(buffer) {}

  virtual ~host_buffer_sink() { flush(); }

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
  explicit void_sink() : _bytes_written(0) {}

  virtual ~void_sink() {}

  void host_write(void const* data, size_t size) override { _bytes_written += size; }

  bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    _bytes_written += size;
  }

  void flush() override {}

  size_t bytes_written() override { return _bytes_written; }

 private:
  size_t _bytes_written;
};

class user_sink_wrapper : public data_sink {
 public:
  explicit user_sink_wrapper(cudf::io::data_sink* const user_sink_) : user_sink(user_sink_) {}

  virtual ~user_sink_wrapper() {}

  void host_write(void const* data, size_t size) override { user_sink->host_write(data, size); }

  bool supports_device_write() const override { return user_sink->supports_device_write(); }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(user_sink->supports_device_write(),
                 "device_write() being called on a data_sink that doesn't support it");
    user_sink->device_write(gpu_data, size, stream);
  }

  void flush() override { user_sink->flush(); }

  size_t bytes_written() override { return user_sink->bytes_written(); }

 private:
  cudf::io::data_sink* const user_sink;
};

std::unique_ptr<data_sink> data_sink::create(const std::string& filepath)
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
