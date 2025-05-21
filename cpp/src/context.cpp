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

#include <cudf/context.h>
#include <cudf/utilities/error.hpp>

#include <memory>

namespace cudf {

namespace detail {
std::unique_ptr<Context>& context_ptr()
{
  static std::unique_ptr<Context> context;
  return context;
}
}  // namespace detail

// [ ] todo
Context::Context(): _program_cache{}{}

Context& context()
{
  auto& ctx = detail::context_ptr();
  CUDF_EXPECTS(ctx != nullptr, "Context is not initialized");
  return *ctx;
}

void initialize()
{
  CUDF_EXPECTS(detail::context_ptr() == nullptr, "Context is already initialized");
  detail::context_ptr() = std::make_unique<Context>();
}

void deinitialize()
{
  CUDF_EXPECTS(detail::context_ptr() != nullptr, "Context has already been deinitialized");
  detail::context_ptr().reset();
}

}  // namespace cudf