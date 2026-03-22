/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/context.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>
#include <mutex>

namespace rtcx {
class cache_t;
}  // namespace rtcx

namespace cudf {

namespace jit {
class program_cache;
}

class jit_bundle_t;

struct [[nodiscard]] context_config {
  bool dump_codegen      : 1          = false;
  bool use_jit           : 1          = false;
  bool preload_jit_cache : 1          = false;
  bool disable_jit_cache : 1          = false;
  bool clear_jit_cache   : 1          = false;
  std::string rtcx_cache_dir          = {};
  std::string jit_bundle_dir          = {};
  std::string jit_pch_dir             = {};
  std::string jit_tmp_dir             = {};
  uint32_t kernel_cache_limit_process = 0;
  uint32_t kernel_cache_limit_disk    = 0;
};

/// @brief The context object contains global state internal to CUDF.
/// It helps to ensure structured and well-defined construction and destruction of global
/// objects/state across translation units.
class context {
 private:
  context_config _config;
  std::once_flag _jit_cache_init_flag;
  std::unique_ptr<rtcx::cache_t> _rtcx_cache;
  std::unique_ptr<jit_bundle_t> _jit_bundle;

 private:
  void ensure_nvcomp_loaded();

  void ensure_jit_cache_initialized();

 public:
  context(context_config cfg = {}, init_flags flags = init_flags::DEFAULT);
  context(context const&)            = delete;
  context& operator=(context const&) = delete;
  context(context&&)                 = delete;
  context& operator=(context&&)      = delete;
  ~context();

  rtcx::cache_t& rtcx_cache();

  jit_bundle_t& jit_bundle();

  [[nodiscard]] bool dump_codegen() const;

  [[nodiscard]] bool use_jit() const;

  [[nodiscard]] std::string const& get_jit_pch_dir() const;

  /// @brief Initialize additional components based on the provided flags
  /// @param flags The initialization flags to process
  void initialize_components(init_flags flags);
};

/// @brief Get the cuDF global context
context& get_context();

}  // namespace cudf
