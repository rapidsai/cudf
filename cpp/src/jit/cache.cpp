/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <jit/cache.h>
#include <cudf/utilities/error.hpp>

#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include <cuda.h>

namespace cudf {
namespace jit {
// Get the directory in home to use for storing the cache
boost::filesystem::path get_user_home_cache_dir()
{
  auto home_dir = std::getenv("HOME");
  if (home_dir != nullptr) {
    return boost::filesystem::path(home_dir) / ".cudf";
  } else {
    return boost::filesystem::path();
  }
}

// Default `LIBCUDF_KERNEL_CACHE_PATH` to `$HOME/.cudf/$CUDF_VERSION`.
// This definition can be overridden at compile time by specifying a
// `-DLIBCUDF_KERNEL_CACHE_PATH=/kernel/cache/path` CMake argument.
// Use `boost::filesystem` for cross-platform path resolution and dir
// creation. This path is used in the `getCacheDir()` function below.
#if !defined(LIBCUDF_KERNEL_CACHE_PATH)
#define LIBCUDF_KERNEL_CACHE_PATH get_user_home_cache_dir()
#endif

/**
 * @brief Get the string path to the JITIFY kernel cache directory.
 *
 * This path can be overridden at runtime by defining an environment variable
 * named `LIBCUDF_KERNEL_CACHE_PATH`. The value of this variable must be a path
 * under which the process' user has read/write privileges.
 *
 * This function returns a path to the cache directory, creating it if it
 * doesn't exist.
 *
 * The default cache directory is `$HOME/.cudf/$CUDF_VERSION`. If no overrides
 * are used and if $HOME is not defined, returns an empty path and file
 * caching is not used.
 **/
boost::filesystem::path getCacheDir()
{
  // The environment variable always overrides the
  // default/compile-time value of `LIBCUDF_KERNEL_CACHE_PATH`
  auto kernel_cache_path_env = std::getenv("LIBCUDF_KERNEL_CACHE_PATH");
  auto kernel_cache_path     = boost::filesystem::path(
    kernel_cache_path_env != nullptr ? kernel_cache_path_env : LIBCUDF_KERNEL_CACHE_PATH);

  // Cache path could be empty when env HOME is unset or LIBCUDF_KERNEL_CACHE_PATH is defined to be
  // empty, to disallow use of file cache at runtime.
  if (not kernel_cache_path.empty()) {
    kernel_cache_path /= std::string{CUDF_STRINGIFY(CUDF_VERSION)};
    try {
      // `mkdir -p` the kernel cache path if it doesn't exist
      boost::filesystem::create_directories(kernel_cache_path);
    } catch (const std::exception& e) {
      // if directory creation fails for any reason, return empty path
      return boost::filesystem::path();
    }
  }
  return kernel_cache_path;
}

cudfJitCache::cudfJitCache() {}

cudfJitCache::~cudfJitCache() {}

std::mutex cudfJitCache::_kernel_cache_mutex;
std::mutex cudfJitCache::_program_cache_mutex;

named_prog<jitify::experimental::Program> cudfJitCache::getProgram(
  std::string const& prog_name,
  std::string const& cuda_source,
  std::vector<std::string> const& given_headers,
  std::vector<std::string> const& given_options,
  jitify::experimental::file_callback_type file_callback)
{
  // Lock for thread safety
  std::lock_guard<std::mutex> lock(_program_cache_mutex);

  return getCached(prog_name, program_map, [&]() {
    CUDF_EXPECTS(not cuda_source.empty(), "Program not found in cache, Needs source string.");
    return jitify::experimental::Program(cuda_source, given_headers, given_options, file_callback);
  });
}

named_prog<jitify::experimental::KernelInstantiation> cudfJitCache::getKernelInstantiation(
  std::string const& kern_name,
  named_prog<jitify::experimental::Program> const& named_program,
  std::vector<std::string> const& arguments)
{
  // Lock for thread safety
  std::lock_guard<std::mutex> lock(_kernel_cache_mutex);

  std::string prog_name                  = std::get<0>(named_program);
  jitify::experimental::Program& program = *std::get<1>(named_program);

  // Make instance name e.g. "prog_binop.kernel_v_v_int_int_long int_Add"
  std::string kern_inst_name = prog_name + '.' + kern_name;
  for (auto&& arg : arguments) kern_inst_name += '_' + arg;

  CUcontext c;
  cuCtxGetCurrent(&c);

  auto& kernel_inst_map = kernel_inst_context_map[c];

  return getCached(kern_inst_name, kernel_inst_map, [&]() {
    return program.kernel(kern_name).instantiate(arguments);
  });
}

// Another overload for getKernelInstantiation which might be useful to get
// kernel instantiations in one step
// ------------------------------------------------------------------------
/*
jitify::experimental::KernelInstantiation cudfJitCache::getKernelInstantiation(
    std::string const& kern_name,
    std::string const& prog_name,
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    file_callback_type file_callback = nullptr)
{
    auto program = getProgram(prog_name,
                              cuda_source,
                              given_headers,
                              given_options,
                              file_callback);
    return getKernelInstantiation(kern_name, program);
}
*/

cudfJitCache::cacheFile::cacheFile(std::string file_name) : _file_name{file_name} {}

cudfJitCache::cacheFile::~cacheFile() {}

std::string cudfJitCache::cacheFile::read()
{
  // Open file (duh)
  int fd = open(_file_name.c_str(), O_RDWR);
  if (fd == -1) {
    successful_read = false;
    return std::string();
  }

  // Create args for file locking
  flock fl{};
  fl.l_type   = F_RDLCK;  // Shared lock for reading
  fl.l_whence = SEEK_SET;

  // Lock the file descriptor. Only reading is allowed now
  if (fcntl(fd, F_SETLKW, &fl) == -1) {
    successful_read = false;
    return std::string();
  }

  // Get file descriptor from file pointer
  FILE* fp = fdopen(fd, "rb");

  // Get file length
  fseek(fp, 0L, SEEK_END);
  size_t file_size = ftell(fp);
  rewind(fp);

  // Allocate memory of file length size
  std::string content;
  content.resize(file_size);
  char* buffer = &content[0];

  // Copy file into buffer
  if (fread(buffer, file_size, 1, fp) != 1) {
    successful_read = false;
    fclose(fp);
    free(buffer);
    return std::string();
  }
  fclose(fp);
  successful_read = true;

  return content;
}

void cudfJitCache::cacheFile::write(std::string content)
{
  // Open file and create if it doesn't exist, with access 0600
  int fd = open(_file_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    successful_write = false;
    return;
  }

  // Create args for file locking
  flock fl{};
  fl.l_type   = F_WRLCK;  // Exclusive lock for writing
  fl.l_whence = SEEK_SET;

  // Lock the file descriptor. we the only ones now
  if (fcntl(fd, F_SETLKW, &fl) == -1) {
    successful_write = false;
    return;
  }

  // Get file descriptor from file pointer
  FILE* fp = fdopen(fd, "wb");

  // Copy string into file
  if (fwrite(content.c_str(), content.length(), 1, fp) != 1) {
    successful_write = false;
    fclose(fp);
    return;
  }
  fclose(fp);

  successful_write = true;
  return;
}

}  // namespace jit
}  // namespace cudf
