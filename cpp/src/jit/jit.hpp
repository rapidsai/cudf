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

struct [[nodiscard]] udf_compile_params {
  std::string_view name = {};

  std::span<char const> udf = {};

  std::string_view key = {};

  std::string_view kernel_symbol = {};

  std::span<std::string_view const> extra_compile_flags = {};

  std::span<std::string_view const> extra_link_flags = {};
};

[[nodiscard]] rtc::library compile_and_link_cuda_udf(udf_compile_params const& params);

struct [[nodiscard]] udf_link_params {
  std::string_view name = {};

  std::span<uint8_t const> udf_blob = {};

  rtc::binary_type type = rtc::binary_type::LTO_IR;

  std::string_view key = {};

  std::string_view kernel_symbol = {};

  std::span<std::string_view const> extra_link_flags = {};
};

[[nodiscard]] rtc::library link_udf(udf_link_params const& params);

}  // namespace CUDF_EXPORT cudf
