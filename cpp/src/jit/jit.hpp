/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <jit/rtc/cache.hpp>
#include <jit/rtc/rtc.hpp>

namespace CUDF_EXPORT cudf {

struct [[nodiscard]] jit_bundle_t {
 private:
  std::string install_dir_;
  rtc::fragment lto_library_;
  rtc::cache_t* cache_;

  void ensure_installed() const;

  void preload_lto_library();

 public:
  jit_bundle_t(std::string install_dir, rtc::cache_t& cache);

  [[nodiscard]] std::string get_hash() const;

  [[nodiscard]]
  std::string get_directory() const;

  [[nodiscard]] rtc::fragment get_lto_library() const;

  [[nodiscard]] std::vector<std::string> get_include_directories() const;

  [[nodiscard]] std::vector<std::string> get_compile_options() const;
};

[[nodiscard]] rtc::library compile_kernel(std::string const& name,
                                          std::string const& key,
                                          std::string const& cuda_udf,
                                          std::string const& kernel_symbol,
                                          bool use_cache = true,
                                          bool use_pch   = true,
                                          bool log_pch   = false);

[[nodiscard]] rtc::library compile_lto_ir_kernel(std::string const& name,
                                                 std::string const& key,
                                                 std::span<uint8_t const> lto_ir_binary,
                                                 std::string const& kernel_symbol,
                                                 bool use_cache = true);

}  // namespace CUDF_EXPORT cudf
