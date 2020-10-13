/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace io {
/**
 * @brief Implementation class for storing data into a local file.
 *
 */
class file_sink : public data_sink {
 public:
  explicit file_sink(std::string const& filepath)
  {
    outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
  }

  virtual ~file_sink() { flush(); }

  void host_write(void const* data, size_t size) override
  {
    outfile_.write(static_cast<char const*>(data), size);
  }

  void flush() override { outfile_.flush(); }

  size_t bytes_written() override { return outfile_.tellp(); }

 private:
  std::ofstream outfile_;
};

/**
 * @brief Implementation class for storing data into a std::vector.
 *
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
 *
 */
class void_sink : public data_sink {
 public:
  explicit void_sink() : bytes_written_(0) {}

  virtual ~void_sink() {}

  void host_write(void const* data, size_t size) override { bytes_written_ += size; }

  bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, cudaStream_t stream) override
  {
    bytes_written_ += size;
  }

  void flush() override {}

  size_t bytes_written() override { return bytes_written_; }

 private:
  size_t bytes_written_;
};

class user_sink_wrapper : public data_sink {
 public:
  explicit user_sink_wrapper(cudf::io::data_sink* const user_sink_) : user_sink(user_sink_) {}

  virtual ~user_sink_wrapper() {}

  void host_write(void const* data, size_t size) override { user_sink->host_write(data, size); }

  bool supports_device_write() const override { return user_sink->supports_device_write(); }

  void device_write(void const* gpu_data, size_t size, cudaStream_t stream) override
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
