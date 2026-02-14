/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/context.hpp"

#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/getenv_or.hpp"
#include "jit/cache.hpp"
#include "jit/jit.hpp"
#include "jit/rtc/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

namespace cudf {

context::context(context_config cfg, init_flags flags)
  : _config{std::move(cfg)}, _program_cache_init_flag{}, _program_cache{nullptr}
{
  initialize_components(flags);
}

void context::ensure_nvcomp_loaded() { io::detail::nvcomp::load_nvcomp_library(); }

void context::ensure_jit_cache_initialized()
{
  std::call_once(_program_cache_init_flag,
                 [&]() { _program_cache = std::make_unique<jit::program_cache>(); });
}

void context::ensure_rtc_cache_initialized()
{
  std::call_once(_rtc_cache_init_flag, [&]() {
    // make sure the rtc cache directory exists
    std::filesystem::create_directories(_config.rtc_cache_dir);
    _rtc_cache =
      std::make_unique<rtc::cache_t>(_config.rtc_cache_dir, rtc::cache_limits::default_limits());
  });
}

void context::ensure_jit_bundle_initialized()
{
  // note that jit_bundle depends on rtc_cache, so we ensure rtc_cache is initialized first.
  ensure_rtc_cache_initialized();
  std::call_once(_jit_bundle_init_flag, [&]() {
    // make sure the jit bundle directory exists
    std::filesystem::create_directories(_config.jit_bundle_dir);
    _jit_bundle = std::make_unique<jit_bundle_t>(_config.jit_bundle_dir, *_rtc_cache);
  });
}

jit::program_cache& context::program_cache()
{
  ensure_jit_cache_initialized();
  return *_program_cache;
}

rtc::cache_t& context::rtc_cache()
{
  ensure_rtc_cache_initialized();
  return *_rtc_cache;
}

jit_bundle_t& context::jit_bundle()
{
  ensure_jit_bundle_initialized();
  return *_jit_bundle;
}

bool context::dump_codegen() const { return _config.dump_codegen; }

bool context::use_jit() const { return _config.use_jit; }

void context::initialize_components(init_flags flags)
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) { ensure_jit_cache_initialized(); }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }

  if (has_flag(flags, init_flags::INIT_RTC_CACHE)) { ensure_rtc_cache_initialized(); }

  if (has_flag(flags, init_flags::INIT_JIT_BUNDLE)) { ensure_jit_bundle_initialized(); }
}

static std::optional<context> _context{std::nullopt};
static std::optional<std::once_flag> _context_init_flag{std::in_place};
static std::optional<std::once_flag> _context_deinit_flag{std::in_place};

std::filesystem::path get_cudf_dir()
{
  auto cudf_dir_env = std::getenv("LIBCUDF_DIR");
  if (cudf_dir_env != nullptr) {
    return std::filesystem::path(cudf_dir_env);
  } else {
    auto home_dir = std::getenv("HOME");
    CUDF_EXPECTS(home_dir != nullptr, "HOME environment variable is not set", std::runtime_error);
    auto cudf_dir = std::filesystem::path(home_dir) / ".cudf";
    return cudf_dir;
  }
}

std::filesystem::path get_jit_bundle_dir()
{
  return getenv_or("LIBCUDF_JIT_BUNDLE_DIR", get_cudf_dir() / "jit_bundle");
}

std::filesystem::path get_rtc_cache_dir()
{
  return getenv_or("LIBCUDF_RTC_CACHE_DIR", get_cudf_dir() / "rtc_cache");
}

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  std::call_once(*_context_init_flag, [&]() {
    bool dump_codegen = get_bool_env_or("LIBCUDF_JIT_DUMP_CODEGEN", false);
    bool use_jit      = get_bool_env_or("LIBCUDF_JIT_ENABLED", false);

    flags = flags | (use_jit ? init_flags::INIT_JIT_CACHE : init_flags::NONE);

    auto jit_bundle_dir = get_jit_bundle_dir();
    auto rtc_cache_dir  = get_rtc_cache_dir();

    context_config cfg{
      .dump_codegen   = dump_codegen,
      .use_jit        = use_jit,
      .rtc_cache_dir  = rtc_cache_dir,
      .jit_bundle_dir = jit_bundle_dir,
    };

    _context.emplace(cfg, flags);
  });

  _context->initialize_components(flags);
}

void teardown()
{
  std::call_once(*_context_deinit_flag, [&]() {
    // reset the context to destroy all global objects and release resources, allowing for clean
    // re-initialization in the future if desired.
    _context.reset();
    _context_init_flag.emplace();
    _context_deinit_flag.emplace();
  });
}

context& get_context()
{
  cudf::initialize();
  return *_context;
}

}  // namespace CUDF_EXPORT cudf
