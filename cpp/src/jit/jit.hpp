/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <librtcx/rtcx.hpp>

namespace CUDF_EXPORT cudf {

struct [[nodiscard]] jit_bundle_t {
 private:
  std::string install_dir_;
  rtcx::cache_t* cache_;

  void ensure_installed() const;

 public:
  jit_bundle_t(std::string install_dir, rtcx::cache_t& cache);

  [[nodiscard]] std::string get_hash() const;

  [[nodiscard]] std::string get_directory() const;

  [[nodiscard]] std::vector<std::string> get_include_directories() const;
};

[[nodiscard]] rtcx::library get_library(std::string const& name,
                                        std::string const& key,
                                        std::string const& cuda_udf,
                                        bool use_cache = true,
                                        bool use_pch   = true,
                                        bool log_pch   = false);

}  // namespace CUDF_EXPORT cudf
