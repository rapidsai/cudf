/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

namespace jit {
struct program_cache;
}

class context {
 private:
  std::unique_ptr<jit::program_cache> _program_cache;

 public:
  context();
  context(context const&)            = delete;
  context& operator=(context const&) = delete;
  context(context&&)                 = delete;
  context& operator=(context&&)      = delete;
  ~context()                         = default;

  jit::program_cache& program_cache();
};

namespace detail {
std::unique_ptr<context>& get_context_ptr();
}

context& get_context();

void initialize();

void deinitialize();

class context_guard {
 public:
  context_guard() {}
  context_guard(context_guard const&)            = delete;
  context_guard(context_guard&&)                 = delete;
  context_guard& operator=(context_guard&&)      = delete;
  context_guard& operator=(context_guard const&) = delete;
  ~context_guard() { deinitialize(); }
};

}  // namespace CUDF_EXPORT cudf
