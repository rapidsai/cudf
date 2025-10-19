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

#pragma once

#include <cudf/context.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>

namespace cudf {

namespace jit {
class program_cache;
}

/// @brief The context object contains global state internal to CUDF.
/// It helps to ensure structured and well-defined construction and destruction of global
/// objects/state across translation units.
class context {
 private:
  std::unique_ptr<jit::program_cache> _program_cache;
  init_flags _initialized_flags = init_flags::NONE;
  bool _dump_codegen            = false;
  bool _use_jit                 = false;

 public:
  context(init_flags flags = init_flags::INIT_JIT_CACHE);
  context(context const&)            = delete;
  context& operator=(context const&) = delete;
  context(context&&)                 = delete;
  context& operator=(context&&)      = delete;
  ~context()                         = default;

  jit::program_cache& program_cache();

  [[nodiscard]] bool dump_codegen() const;

  /// @brief Initialize additional components based on the provided flags
  /// @param flags The initialization flags to process
  void initialize_components(init_flags flags);

  [[nodiscard]] bool use_jit() const;
};

std::unique_ptr<context>& get_context_ptr_ref();

context& get_context();

}  // namespace cudf
