/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/context.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>
#include <mutex>

namespace cudf {

namespace jit {
class program_cache;
}

struct [[nodiscard]] context_config {
  bool dump_codegen = false;
  bool use_jit      = false;
};

/// @brief The context object contains global state internal to CUDF.
/// It helps to ensure structured and well-defined construction and destruction of global
/// objects/state across translation units.
class context {
 public:
 private:
  context_config _config;
  std::once_flag _program_cache_init_flag;
  std::unique_ptr<jit::program_cache> _program_cache;

 private:
  void ensure_nvcomp_loaded();

  void ensure_jit_cache_initialized();

 public:
  context(context_config const& cfg = {}, init_flags flags = init_flags::INIT_JIT_CACHE);
  context(context const&)            = delete;
  context& operator=(context const&) = delete;
  context(context&&)                 = delete;
  context& operator=(context&&)      = delete;
  ~context()                         = default;

  jit::program_cache& program_cache();

  [[nodiscard]] bool dump_codegen() const;

  [[nodiscard]] bool use_jit() const;

  /// @brief Initialize additional components based on the provided flags
  /// @param flags The initialization flags to process
  void initialize_components(init_flags flags);
};

/// @brief Get the cuDF global context
context& get_context();

}  // namespace cudf
