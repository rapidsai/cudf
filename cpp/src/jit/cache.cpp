/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/getenv_or.hpp"
#include "runtime/context.hpp"

#include <cudf/context.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <jit/cache.hpp>

#include <filesystem>

namespace cudf {
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

}  // namespace

jitify2::ProgramCache<>& jit::program_cache::get(jitify2::PreprocessedProgramData const& preprog)
{
  CUDF_FUNC_RANGE();
  std::lock_guard caches_lock(_caches_mutex);

  auto existing_cache = _caches.find(preprog.name());

  if (existing_cache == _caches.end() || _disabled.load(std::memory_order_seq_cst)) {
    auto res =
      _caches.emplace(preprog.name(),
                      std::make_unique<jitify2::ProgramCache<>>(
                        _kernel_limit_proc, preprog, nullptr, _cache_dir, _kernel_limit_disk));
    existing_cache = res.first;
  }

  return *(existing_cache->second);
}

void jit::program_cache::clear()
{
  CUDF_FUNC_RANGE();
  std::lock_guard caches_lock(_caches_mutex);

  _caches.clear();

  // non-atomic
  std::filesystem::remove_all(_cache_dir);
}

void jit::program_cache::enable(bool enable)
{
  _disabled.store(!enable, std::memory_order_seq_cst);
}

bool jit::program_cache::is_enabled() const { return !_disabled.load(std::memory_order_seq_cst); }

std::unique_ptr<jit::program_cache> jit::program_cache::create()
{
  auto kernel_limit_proc = getenv_or("LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS", 10'000);
  auto kernel_limit_disk = getenv_or("LIBCUDF_KERNEL_CACHE_LIMIT_DISK", 100'000);
  auto disabled          = get_bool_env_or("LIBCUDF_KERNEL_CACHE_DISABLED", false);

  // if kernel_limit_disk is zero, jitify will assign it the value of kernel_limit_proc.
  // to avoid this, we treat zero as "disable disk caching" by not providing the cache dir.
  auto cache_dir = kernel_limit_disk == 0 ? std::string{} : get_program_cache_dir();

  return std::make_unique<jit::program_cache>(
    kernel_limit_proc, kernel_limit_disk, cache_dir, disabled);
}

jitify2::ProgramCache<>& jit::get_program_cache(jitify2::PreprocessedProgramData const& preprog)
{
  return cudf::get_context().program_cache().get(preprog);
}
}  // namespace cudf
