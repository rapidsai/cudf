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
  CUDF_EXPECTS(_program_cache != nullptr, "JIT cache is not initialized", std::runtime_error);
  return *_program_cache;
}

bool context::dump_codegen() const { return _dump_codegen; }

void context::initialize_components(init_flags flags)
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) {
    _program_cache = std::make_unique<jit::program_cache>();
  }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }

  _initialized_flags = flags;
}

bool context::use_jit() const { return _use_jit; }

static std::unique_ptr<context> ctx;
static std::once_flag ctx_init_flag{};

}  // namespace cudf

namespace CUDF_EXPORT cudf {

void initialize(init_flags flags)
{
  std::call_once(ctx_init_flag, [&]() {
    auto c = std::make_unique<context>();
    c->initialize_components(flags);
    ctx = std::move(c);
  });

  CUDF_EXPECTS(has_flag(ctx->_initialized_flags, flags),
               "CUDF's context has already been initialized with incompatible flags",
               std::runtime_error);
}

context& get_context(init_flags expected)
{
  initialize(expected);
  return *ctx;
}

}  // namespace CUDF_EXPORT cudf
