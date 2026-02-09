

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <jit/rtc/cache.hpp>
#include <jit/rtc/rtc.hpp>

namespace CUDF_EXPORT cudf {
namespace rtc {

class jit_bundle_t {
  std::string install_dir_;
  fragment lto_library_;
  cache_t* cache_;

 private:
  void ensure_installed() const;

  void preload_lto_library();

 public:
  jit_bundle_t(std::string install_dir, cache_t& cache);

  std::string get_hash() const;

  std::string get_directory() const;

  fragment get_lto_library() const;

  std::vector<std::string> get_include_directories() const;

  std::vector<std::string> get_compile_options() const;
};

library compile_and_link_udf(char const* name,
                             char const* udf_code,
                             char const* udf_key,
                             char const* kernel_symbol);

}  // namespace rtc
}  // namespace CUDF_EXPORT cudf
