/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/context.hpp"

#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/getenv_or.hpp"
#include "jit/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

namespace cudf {

context::context(context_config const& cfg, init_flags flags)
  : _config{cfg}, _program_cache_init_flag{}, _program_cache{nullptr}
{
  initialize_components(flags);
}

void context::ensure_nvcomp_loaded() { io::detail::nvcomp::load_nvcomp_library(); }

void context::ensure_jit_cache_initialized()
{
  std::call_once(_program_cache_init_flag,
                 [&]() { _program_cache = std::make_unique<jit::program_cache>(); });
}

jit::program_cache& context::program_cache()
{
  ensure_jit_cache_initialized();
  return *_program_cache;
}

bool context::dump_codegen() const { return _config.dump_codegen; }

bool context::use_jit() const { return _config.use_jit; }

void context::initialize_components(init_flags flags)
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) { ensure_jit_cache_initialized(); }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }
}

static std::optional<context> _context{std::nullopt};
static std::optional<std::once_flag> _context_init_flag{std::in_place};
static std::optional<std::once_flag> _context_deinit_flag{std::in_place};

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  std::call_once(*_context_init_flag, [&]() {
    bool dump_codegen = get_bool_env_or("LIBCUDF_JIT_DUMP_CODEGEN", false);
    bool use_jit      = get_bool_env_or("LIBCUDF_JIT_ENABLED", false);

    flags = flags | (use_jit ? init_flags::INIT_JIT_CACHE : init_flags::NONE);

    context_config cfg{
      .dump_codegen = dump_codegen,
      .use_jit      = use_jit,
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
