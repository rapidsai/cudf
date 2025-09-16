/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "runtime/context.hpp"

#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/getenv_or.hpp"
#include "jit/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

namespace cudf {

context::context() : _program_cache{nullptr}
{
  if (has_flag(flags, init_flags::INIT_JIT_CACHE)) {
    _program_cache = std::make_unique<jit::program_cache>();
  }

  if (has_flag(flags, init_flags::LOAD_NVCOMP)) { io::detail::nvcomp::load_nvcomp_library(); }
  
  auto dump_codegen_flag = getenv_or("LIBCUDF_JIT_DUMP_CODEGEN", std::string{"OFF"});
  _dump_codegen          = (dump_codegen_flag == "ON" || dump_codegen_flag == "1");
}

jit::program_cache& context::program_cache()
{
  CUDF_EXPECTS(_program_cache != nullptr, "JIT cache not initialized", std::runtime_error);
  return *_program_cache;
}

bool context::dump_codegen() const { return _dump_codegen; }

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
  CUDF_EXPECTS(
    get_context_ptr_ref() == nullptr, "context is already initialized", std::runtime_error);
  get_context_ptr_ref() = std::make_unique<context>(flags);
}

void deinitialize()
{
  CUDF_EXPECTS(
    get_context_ptr_ref() != nullptr, "context has already been deinitialized", std::runtime_error);
  get_context_ptr_ref().reset();
}
}  // namespace CUDF_EXPORT cudf
