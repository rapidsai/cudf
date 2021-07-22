/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifdef CUFILE_FOUND
#include <cufile.h>
#include <cudf_test/file_utilities.hpp>
#endif

#include <rmm/cuda_stream_view.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>

#include <string>

namespace cudf {
namespace io {
namespace detail {

/**
 * @brief Class that provides RAII for file handling.
 */
class file_wrapper {
  int fd = -1;
  size_t _size;

 public:
  explicit file_wrapper(std::string const& filepath, int flags);
  explicit file_wrapper(std::string const& filepath, int flags, mode_t mode);
  ~file_wrapper();
  auto size() const { return _size; }
  auto desc() const { return fd; }
};

/**
 * @brief Base class for cuFile input/output.
 *
 * Contains the common API for cuFile input and output classes.
 */
class cufile_io_base {
 public:
  /**
   * @brief Returns an estimate of whether the cuFile operation is the optimal option.
   *
   * @param size Read/write operation size, in bytes.
   * @return Whether a cuFile operation with the given size is expected to be faster than a host
   * read + H2D copy
   */
  static bool is_cufile_io_preferred(size_t size) { return size > op_size_threshold; }

 protected:
  /**
   * @brief The read/write size above which cuFile is faster then host read + copy
   *
   * This may not be the optimal threshold for all systems. Derived `is_cufile_io_preferred`
   * implementations can use a different logic.
   */
  static constexpr size_t op_size_threshold = 128 << 10;
};

/**
 * @brief Interface class for cufile input.
 */
class cufile_input : public cufile_io_base {
 public:
  /**
   * @brief Reads into a new device buffer.
   *
   *  @throws cudf::logic_error on cuFile error
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param stream CUDA stream to use
   *
   * @return The data buffer in the device memory
   */
  virtual std::unique_ptr<datasource::buffer> read(size_t offset,
                                                   size_t size,
                                                   rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Reads into existing device memory.
   *
   *  @throws cudf::logic_error on cuFile error
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param dst Address of the existing device memory
   * @param stream CUDA stream to use
   *
   * @return The number of bytes read
   */
  virtual size_t read(size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief Interface class for cufile output.
 */
class cufile_output : public cufile_io_base {
 public:
  /**
   * @brief Writes the data from a device buffer into a file.
   *
   *  @throws cudf::logic_error on cuFile error
   *
   * @param data Pointer to the buffer to be written into the output file
   * @param offset Number of bytes from the start
   * @param size Number of bytes to write
   */
  virtual void write(void const* data, size_t offset, size_t size) = 0;
};

#ifdef CUFILE_FOUND

class cufile_shim;

/**
 * @brief Class that manages cuFile configuration.
 */
class cufile_config {
  std::string const default_policy    = "OFF";
  std::string const json_path_env_var = "CUFILE_ENV_PATH_JSON";

  std::string const policy = default_policy;
  temp_directory tmp_config_dir{"cudf_cufile_config"};

  cufile_config();

 public:
  /**
   * @brief Returns true when cuFile use is enabled.
   */
  bool is_enabled() const { return policy == "ALWAYS" or policy == "GDS"; }

  /**
   * @brief Returns true when cuDF should not fall back to host IO.
   */
  bool is_required() const { return policy == "ALWAYS"; }

  static cufile_config const* instance();
};

/**
 * @brief Class that provides RAII for cuFile file registration.
 */
struct cufile_registered_file {
  void register_handle();

 public:
  cufile_registered_file(cufile_shim const* shim, std::string const& filepath, int flags)
    : _file(filepath, flags), shim{shim}
  {
    register_handle();
  }

  cufile_registered_file(cufile_shim const* shim,
                         std::string const& filepath,
                         int flags,
                         mode_t mode)
    : _file(filepath, flags, mode), shim{shim}
  {
    register_handle();
  }

  auto const& handle() const noexcept { return cf_handle; }

  ~cufile_registered_file();

 private:
  file_wrapper const _file;
  CUfileHandle_t cf_handle = nullptr;
  cufile_shim const* shim  = nullptr;
};

/**
 * @brief Adapter for the `cuFileRead` API.
 *
 * Exposes APIs to read directly from a file into device memory.
 */
class cufile_input_impl final : public cufile_input {
 public:
  cufile_input_impl(std::string const& filepath);

  std::unique_ptr<datasource::buffer> read(size_t offset,
                                           size_t size,
                                           rmm::cuda_stream_view stream) override;

  size_t read(size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream) override;

 private:
  cufile_shim const* shim = nullptr;
  cufile_registered_file const cf_file;
};

/**
 * @brief Adapter for the `cuFileWrite` API.
 *
 * Exposes an API to write directly into a file from device memory.
 */
class cufile_output_impl final : public cufile_output {
 public:
  cufile_output_impl(std::string const& filepath);

  void write(void const* data, size_t offset, size_t size) override;

 private:
  cufile_shim const* shim = nullptr;
  cufile_registered_file const cf_file;
};
#else

class cufile_input_impl final : public cufile_input {
 public:
  std::unique_ptr<datasource::buffer> read(size_t offset,
                                           size_t size,
                                           rmm::cuda_stream_view stream) override
  {
    CUDF_FAIL("Only used to compile without cufile library, should not be called");
  }

  size_t read(size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream) override
  {
    CUDF_FAIL("Only used to compile without cufile library, should not be called");
  }
};

class cufile_output_impl final : public cufile_output {
 public:
  void write(void const* data, size_t offset, size_t size) override
  {
    CUDF_FAIL("Only used to compile without cufile library, should not be called");
  }
};
#endif

/**
 * @brief Creates a `cufile_input_impl` object
 *
 * Returns a null pointer if an exception occurs in the `cufile_input_impl` constructor, or if the
 * cuFile library is not installed.
 */
std::unique_ptr<cufile_input_impl> make_cufile_input(std::string const& filepath);

/**
 * @brief Creates a `cufile_output_impl` object
 *
 * Returns a null pointer if an exception occurs in the `cufile_output_impl` constructor, or if the
 * cuFile library is not installed.
 */
std::unique_ptr<cufile_output_impl> make_cufile_output(std::string const& filepath);

}  // namespace detail
}  // namespace io
}  // namespace cudf
