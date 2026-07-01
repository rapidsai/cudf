/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/context.hpp"

#include "io/comp/nvcomp_adapter.hpp"
#include "jit/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/utilities/error.hpp>

#include <filesystem>
#include <memory>
#include <mutex>

namespace cudf {

context::context(context_config cfg, init_flags flags)
  : _config{std::move(cfg)}, _jit_cache_init_flag{}
{
  initialize_components(flags);
}

void context::ensure_nvcomp_loaded() { io::detail::nvcomp::load_nvcomp_library(); }

void context::ensure_jit_cache_initialized()
{
  std::call_once(_jit_cache_init_flag, [&]() {
    // make sure the required directories exist
    std::filesystem::create_directories(_config.rtcx_cache_dir);
    std::filesystem::create_directories(_config.jit_bundle_dir);
    std::filesystem::create_directories(_config.jit_pch_dir);
    std::filesystem::create_directories(_config.jit_tmp_dir);

    rtcx::initialize();

    auto limits = rtcx::cache_limits{.num_mem_blobs     = _config.kernel_cache_limit_process,
                                     .num_mem_libraries = _config.kernel_cache_limit_process};

    _rtcx_cache = std::make_unique<rtcx::cache_t>(_config.rtcx_cache_dir,
                                                  _config.jit_tmp_dir,
                                                  limits,
                                                  bool{_config.preload_jit_cache},
                                                  bool{_config.disable_jit_cache});

    if (_config.clear_jit_cache) {
      _rtcx_cache->clear_memory_store();
      _rtcx_cache->clear_disk_store();
    }

    // note that jit_bundle depends on rtcx_cache, so we ensure rtcx_cache is initialized first.
    _jit_bundle = std::make_unique<jit_bundle_t>(_config.jit_bundle_dir, *_rtcx_cache);
  });
}

context::~context()
{
  _jit_bundle.reset();
  _rtcx_cache.reset();
  rtcx::teardown();
}

rtcx::cache_t& context::rtcx_cache()
{
  ensure_jit_cache_initialized();
  return *_rtcx_cache;
}

jit_bundle_t& context::jit_bundle()
{
  ensure_jit_cache_initialized();
  return *_jit_bundle;
}

bool context::dump_codegen() const { return _config.dump_codegen; }

bool context::use_jit() const { return _config.use_jit; }

std::string const& context::get_jit_pch_dir() const { return _config.jit_pch_dir; }

void context::initialize_components(init_flags flags)
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) { ensure_jit_cache_initialized(); }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }
}

std::filesystem::path get_cudf_kernel_cache_dir()
{
  if (auto libcudf_kernel_cache_path =
        detail::getenv_optional<std::string>("LIBCUDF_KERNEL_CACHE_PATH");
      libcudf_kernel_cache_path.has_value()) {
    return std::filesystem::path(*libcudf_kernel_cache_path);
  }

  if (auto home = detail::getenv_optional<std::string>("HOME"); home.has_value()) {
    return std::filesystem::path(*home) / ".libcudf";
  }

  CUDF_FAIL(
    "Unable to determine the CUDF root directory. Please set the `LIBCUDF_KERNEL_CACHE_PATH` or "
    "`HOME` "
    "environment variables to allow automatic resolution of the root "
    "directory.",
    std::runtime_error);
}

static std::optional<context> _context{std::nullopt};
static std::mutex _context_mutex;

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  std::lock_guard guard{_context_mutex};

  if (!_context.has_value()) {
    auto const dump_codegen      = detail::get_bool_env_or("LIBCUDF_JIT_DUMP_CODEGEN", false);
    auto const use_jit           = detail::get_bool_env_or("LIBCUDF_JIT_ENABLED", false);
    auto const preload_jit_cache = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_PRELOAD", false);
    auto const disable_jit_cache = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_DISABLED", false);
    auto const clear_jit_cache   = detail::get_bool_env_or("LIBCUDF_KERNEL_CACHE_CLEAR", false);
    auto const disable_cuda_cache =
      detail::get_bool_env_or("LIBCUDF_JIT_DISABLE_CUDA_CACHE", false);
    auto const jit_verbose    = detail::get_bool_env_or("LIBCUDF_JIT_VERBOSE", false);
    auto const dump_jit_trace = detail::get_bool_env_or("LIBCUDF_JIT_DUMP_TRACE", false);
    auto const dump_jit_time_profile =
      detail::get_bool_env_or("LIBCUDF_JIT_DUMP_TIME_PROFILE", false);

    auto const kernel_cache_limit_process =
      detail::getenv_or("LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS", 16'384U);

    flags = flags | (use_jit ? init_flags::INIT_JIT_CACHE : init_flags::NONE);

    auto const cache_dir      = get_cudf_kernel_cache_dir();
    auto const jit_bundle_dir = cache_dir / "bundle";
    auto const rtcx_cache_dir = cache_dir / "rtcx_cache";
    auto const jit_pch_dir    = cache_dir / "pch";
    auto const jit_tmp_dir    = cache_dir / "tmp";

    context_config cfg{.dump_codegen               = dump_codegen,
                       .use_jit                    = use_jit,
                       .preload_jit_cache          = preload_jit_cache,
                       .disable_jit_cache          = disable_jit_cache,
                       .clear_jit_cache            = clear_jit_cache,
                       .disable_cuda_cache         = disable_cuda_cache,
                       .jit_verbose                = jit_verbose,
                       .dump_jit_trace             = dump_jit_trace,
                       .dump_jit_time_profile      = dump_jit_time_profile,
                       .rtcx_cache_dir             = rtcx_cache_dir,
                       .jit_bundle_dir             = jit_bundle_dir,
                       .jit_pch_dir                = jit_pch_dir,
                       .jit_tmp_dir                = jit_tmp_dir,
                       .kernel_cache_limit_process = kernel_cache_limit_process};

    _context.emplace(cfg, flags);
  }

  _context->initialize_components(flags);
}

void teardown()
{
  std::lock_guard guard{_context_mutex};
  // reset the context to destroy all global objects and release resources, allowing for clean
  // re-initialization in the future if desired.
  _context.reset();
}

void enable_jit_cache(bool enabled)
{
  auto& cache = get_context().rtcx_cache();
  cache.enable(enabled);
}

void clear_jit_cache()
{
  auto& cache = get_context().rtcx_cache();
  cache.clear_memory_store();
  cache.clear_disk_store();
}

context& get_context()
{
  cudf::initialize();
  return *_context;
}

}  // namespace CUDF_EXPORT cudf
