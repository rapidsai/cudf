/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#ifdef CUDF_CUFILE_FOUND
#include <cudf_test/file_utilities.hpp>

#include <BS_thread_pool.hpp>
#include <cufile.h>
#endif

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string>

namespace cudf {
namespace io {
namespace detail {

[[noreturn]] void throw_on_file_open_failure(std::string const& filepath, bool is_create);

// Call before any cuFile API calls to ensure the CUDA context is initialized.
void force_init_cuda_context();

/**
 * @brief Class that provides RAII for file handling.
 */
class file_wrapper {
  int fd       = -1;
  size_t _size = 0;

 public:
  explicit file_wrapper(std::string const& filepath, int flags, mode_t mode = 0);
  ~file_wrapper();
  [[nodiscard]] auto size() const { return _size; }
  [[nodiscard]] auto desc() const { return fd; }
};

/**
 * @brief Interface class for cufile input.
 */
class cufile_input {
 public:
  /**
   * @brief Asynchronously reads into existing device memory.
   *
   *  @throws cudf::logic_error on cuFile error
   *
   * @param offset Number of bytes from the start
   * @param size Number of bytes to read
   * @param dst Address of the existing device memory
   * @param stream CUDA stream to use
   *
   * @return The number of bytes read as an std::future
   */
  virtual std::future<size_t> read_async(size_t offset,
                                         size_t size,
                                         uint8_t* dst,
                                         rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief Interface class for cufile output.
 */
class cufile_output {
 public:
  /**
   * @brief Asynchronously writes the data from a device buffer into a file.
   *
   * It is the caller's responsibility to not invalidate `data` until the result from this function
   * is synchronized.
   *
   * @throws cudf::logic_error on cuFile error
   *
   * @param data Pointer to the buffer to be written into the output file
   * @param offset Number of bytes from the start
   * @param size Number of bytes to write
   */
  virtual std::future<void> write_async(void const* data, size_t offset, size_t size) = 0;
};

#ifdef CUDF_CUFILE_FOUND

class cufile_shim;

/**
 * @brief Class that provides RAII for cuFile file registration.
 */
class cufile_registered_file {
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

  [[nodiscard]] auto const& handle() const noexcept { return cf_handle; }

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

  std::future<size_t> read_async(size_t offset,
                                 size_t size,
                                 uint8_t* dst,
                                 rmm::cuda_stream_view stream) override;

 private:
  cufile_shim const* shim = nullptr;
  cufile_registered_file const cf_file;
  BS::thread_pool pool;
};

/**
 * @brief Adapter for the `cuFileWrite` API.
 *
 * Exposes an API to write directly into a file from device memory.
 */
class cufile_output_impl final : public cufile_output {
 public:
  cufile_output_impl(std::string const& filepath);

  std::future<void> write_async(void const* data, size_t offset, size_t size) override;

 private:
  cufile_shim const* shim = nullptr;
  cufile_registered_file const cf_file;
  BS::thread_pool pool;
};
#else

class cufile_input_impl final : public cufile_input {
 public:
  cufile_input_impl(std::string const& filepath);
  std::future<size_t> read_async(size_t offset,
                                 size_t size,
                                 uint8_t* dst,
                                 rmm::cuda_stream_view stream) override
  {
    CUDF_FAIL("Only used to compile without cufile library, should not be called");
  }
};

class cufile_output_impl final : public cufile_output {
 public:
  cufile_output_impl(std::string const& filepath);
  std::future<void> write_async(void const* data, size_t offset, size_t size) override
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

/**
 * @brief Byte range to be read/written in a single operation.
 */
CUDF_EXPORT struct file_io_slice {
  size_t offset;
  size_t size;
};

/**
 * @brief Split the total number of bytes to read/write into slices to enable parallel IO.
 *
 * If `max_slice_size` is below 1024, 1024 will be used instead to prevent potential misuse.
 */
CUDF_EXPORT std::vector<file_io_slice> make_file_io_slices(size_t size, size_t max_slice_size);

}  // namespace detail
}  // namespace io
}  // namespace cudf
