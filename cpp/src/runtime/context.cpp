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

#include <cudf/context.hpp>
#include <cudf/utilities/error.hpp>

#include <jit/cache.hpp>

#include <memory>

namespace cudf {

namespace detail {
std::unique_ptr<context>& get_context_ptr()
{
  static std::unique_ptr<context> context;
  return context;
}
}  // namespace detail

context::context() : _program_cache{std::make_unique<jit::program_cache>()} {}

jit::program_cache& context::program_cache() { return *_program_cache; }

context& get_context()
{
  auto& ctx = detail::get_context_ptr();
  CUDF_EXPECTS(ctx != nullptr, "context is not initialized");
  return *ctx;
}

void initialize()
{
  CUDF_EXPECTS(detail::get_context_ptr() == nullptr, "context is already initialized");
  detail::get_context_ptr() = std::make_unique<context>();
}

void deinitialize()
{
  CUDF_EXPECTS(detail::get_context_ptr() != nullptr, "context has already been deinitialized");
  detail::get_context_ptr().reset();
}

}  // namespace cudf