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

namespace rtc {
class cache_t;
class jit_bundle_t;
}  // namespace rtc

struct [[nodiscard]] context_config {
  bool dump_codegen          = false;
  bool use_jit               = false;
  std::string rtc_cache_dir  = {};
  std::string jit_bundle_dir = {};
};

/// @brief The context object contains global state internal to CUDF.
/// It helps to ensure structured and well-defined construction and destruction of global
/// objects/state across translation units.
class context {
 private:
  context_config _config;
  std::once_flag _program_cache_init_flag;
  std::unique_ptr<jit::program_cache> _program_cache;
  std::once_flag _rtc_cache_init_flag;
  std::unique_ptr<rtc::cache_t> _rtc_cache;
  std::once_flag _jit_bundle_init_flag;
  std::unique_ptr<rtc::jit_bundle_t> _jit_bundle;

 private:
  void ensure_nvcomp_loaded();

  void ensure_jit_cache_initialized();

  void ensure_rtc_cache_initialized();

  void ensure_jit_bundle_initialized();

 public:
  context(context_config cfg = {}, init_flags flags = init_flags::DEFAULT);
  context(context const&)            = delete;
  context& operator=(context const&) = delete;
  context(context&&)                 = delete;
  context& operator=(context&&)      = delete;
  ~context()                         = default;

  jit::program_cache& program_cache();

  rtc::cache_t& rtc_cache();

  rtc::jit_bundle_t& jit_bundle();

  [[nodiscard]] bool dump_codegen() const;

  [[nodiscard]] bool use_jit() const;

  /// @brief Initialize additional components based on the provided flags
  /// @param flags The initialization flags to process
  void initialize_components(init_flags flags);
};

/// @brief Get the cuDF global context
context& get_context();

}  // namespace cudf
