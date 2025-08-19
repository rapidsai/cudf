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

#include "jit/cache.hpp"

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

namespace cudf {

context::context() : _program_cache{std::make_unique<jit::program_cache>()} {}

jit::program_cache& context::program_cache() { return *_program_cache; }

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

void initialize()
{
  CUDF_EXPECTS(
    get_context_ptr_ref() == nullptr, "context is already initialized", std::runtime_error);
  get_context_ptr_ref() = std::make_unique<context>();
}

void deinitialize()
{
  CUDF_EXPECTS(
    get_context_ptr_ref() != nullptr, "context has already been deinitialized", std::runtime_error);
  get_context_ptr_ref().reset();
}
}  // namespace CUDF_EXPORT cudf
