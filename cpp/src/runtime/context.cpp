/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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

context::context(init_flags flags) : _program_cache{nullptr}
{
  auto dump_codegen_flag = getenv_or("LIBCUDF_JIT_DUMP_CODEGEN", std::string{"OFF"});
  _dump_codegen          = (dump_codegen_flag == "ON" || dump_codegen_flag == "1");

  auto use_jit_flag = getenv_or("LIBCUDF_JIT_ENABLED", std::string{"OFF"});
  _use_jit          = (use_jit_flag == "ON" || use_jit_flag == "1");

  initialize_components(flags);
}

jit::program_cache& context::program_cache()
{
  CUDF_EXPECTS(_program_cache != nullptr, "JIT cache not initialized", std::runtime_error);
  return *_program_cache;
}

bool context::dump_codegen() const { return _dump_codegen; }

void context::initialize_components(init_flags flags)
{
  // Only initialize components that haven't been initialized yet
  auto const new_flags = flags & ~_initialized_flags;

  if (has_flag(new_flags, init_flags::INIT_JIT_CACHE)) {
    _program_cache = std::make_unique<jit::program_cache>();
  }

  if (has_flag(new_flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }

  _initialized_flags = _initialized_flags | new_flags;
}

bool context::use_jit() const { return _use_jit; }

std::unique_ptr<context>& get_context_ptr_ref()
{
  static std::unique_ptr<context> context;
  return context;
}

context& get_context()
{
  auto& ctx = get_context_ptr_ref();
  if (ctx == nullptr) { cudf::initialize(); }
  return *ctx;
}

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  auto& ctx = get_context_ptr_ref();
  if (ctx == nullptr) {
    // First initialization - create the context
    ctx = std::make_unique<context>(flags);
  } else {
    // Context already exists - initialize additional components
    ctx->initialize_components(flags);
  }
}

void deinitialize() { get_context_ptr_ref().reset(); }
}  // namespace CUDF_EXPORT cudf
