/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "getenv_or.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/logger.hpp>

#include <dlfcn.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace cudf {
namespace io {
namespace detail {
namespace {

[[nodiscard]] int open_file_checked(std::string const& filepath, int flags, mode_t mode)
{
  auto const fd = open(filepath.c_str(), flags, mode);
  if (fd == -1) { throw_on_file_open_failure(filepath, flags & O_CREAT); }

  return fd;
}

[[nodiscard]] size_t get_file_size(int file_descriptor)
{
  struct stat st {};
  CUDF_EXPECTS(fstat(file_descriptor, &st) != -1, "Cannot query file size");
  return static_cast<size_t>(st.st_size);
}

}  // namespace

void force_init_cuda_context()
{
  // Workaround for https://github.com/rapidsai/cudf/issues/14140, where cuFileDriverOpen errors
  // out if no CUDA calls have been made before it. This is a no-op if the CUDA context is already
  // initialized.
  cudaFree(nullptr);
}

[[noreturn]] void throw_on_file_open_failure(std::string const& filepath, bool is_create)
{
  // save errno because it may be overwritten by subsequent calls
  auto const err = errno;

  if (auto const path = std::filesystem::path(filepath); is_create) {
    CUDF_EXPECTS(std::filesystem::exists(path.parent_path()),
                 "Cannot create output file; directory does not exist");

  } else {
    CUDF_EXPECTS(std::filesystem::exists(path), "Cannot open file; it does not exist");
  }

  std::array<char, 1024> error_msg_buffer{};
  auto const error_msg = strerror_r(err, error_msg_buffer.data(), 1024);
  CUDF_FAIL("Cannot open file; failed with errno: " + std::string{error_msg});
}

file_wrapper::file_wrapper(std::string const& filepath, int flags, mode_t mode)
  : fd(open_file_checked(filepath.c_str(), flags, mode)), _size{get_file_size(fd)}
{
}

file_wrapper::~file_wrapper() { close(fd); }

}  // namespace detail
}  // namespace io
}  // namespace cudf
