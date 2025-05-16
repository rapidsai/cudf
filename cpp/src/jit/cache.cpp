/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>

#include <jitify2.hpp>

#include <filesystem>

namespace cudf {
namespace jit {
namespace {

// Get the directory in home to use for storing the cache
std::filesystem::path get_user_home_cache_dir()
{
  auto home_dir = std::getenv("HOME");
  if (home_dir != nullptr) {
    return std::filesystem::path(home_dir) / ".cudf";
  } else {
    return {};
  }
}

// Default `LIBCUDF_KERNEL_CACHE_PATH` to `$HOME/.cudf/$CUDF_VERSION`.
// This definition can be overridden at compile time by specifying a
// `-DLIBCUDF_KERNEL_CACHE_PATH=/kernel/cache/path` CMake argument.
// Use `std::filesystem` for cross-platform path resolution and dir
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
 */
std::filesystem::path get_cache_dir()
{
  // The environment variable always overrides the
  // default/compile-time value of `LIBCUDF_KERNEL_CACHE_PATH`
  auto kernel_cache_path_env = std::getenv("LIBCUDF_KERNEL_CACHE_PATH");
  auto kernel_cache_path     = std::filesystem::path(
    kernel_cache_path_env != nullptr ? kernel_cache_path_env : LIBCUDF_KERNEL_CACHE_PATH);

  // Cache path could be empty when env HOME is unset or LIBCUDF_KERNEL_CACHE_PATH is defined to be
  // empty, to disallow use of file cache at runtime.
  if (not kernel_cache_path.empty()) {
    kernel_cache_path /= std::string{CUDF_STRINGIFY(CUDF_VERSION)};

    // Make per device cache based on compute capability. This is to avoid multiple devices of
    // different compute capability to access the same kernel cache.
    int device   = 0;
    int cc_major = 0;
    int cc_minor = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
    int const cc = cc_major * 10 + cc_minor;

    kernel_cache_path /= std::to_string(cc);

    try {
      // `mkdir -p` the kernel cache path if it doesn't exist
      std::filesystem::create_directories(kernel_cache_path);
    } catch (std::exception const& e) {
      // if directory creation fails for any reason, return empty path
      return {};
    }
  }
  return kernel_cache_path;
}

std::string get_program_cache_dir()
{
#if defined(JITIFY_USE_CACHE)
  return get_cache_dir().string();
#else
  return {};
#endif
}

std::size_t try_parse_numeric_env_var(char const* const env_name, std::size_t default_val)
{
  auto const value = std::getenv(env_name);
  return value != nullptr ? std::stoull(value) : default_val;
}
}  // namespace

jitify2::ProgramCache<>& get_program_cache(jitify2::PreprocessedProgramData preprog)
{
  static std::mutex caches_mutex{};
  static std::unordered_map<std::string, std::unique_ptr<jitify2::ProgramCache<>>> caches{};

  std::lock_guard<std::mutex> const caches_lock(caches_mutex);

  auto existing_cache = caches.find(preprog.name());

  if (existing_cache == caches.end()) {
    auto const kernel_limit_proc =
      try_parse_numeric_env_var("LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS", 10'000);
    auto const kernel_limit_disk =
      try_parse_numeric_env_var("LIBCUDF_KERNEL_CACHE_LIMIT_DISK", 100'000);

    // if kernel_limit_disk is zero, jitify will assign it the value of kernel_limit_proc.
    // to avoid this, we treat zero as "disable disk caching" by not providing the cache dir.
    auto const cache_dir = kernel_limit_disk == 0 ? std::string{} : get_program_cache_dir();

    auto const res =
      caches.insert({preprog.name(),
                     std::make_unique<jitify2::ProgramCache<>>(
                       kernel_limit_proc, preprog, nullptr, cache_dir, kernel_limit_disk)});
    existing_cache = res.first;
  }

  return *(existing_cache->second);
}

}  // namespace jit
}  // namespace cudf
