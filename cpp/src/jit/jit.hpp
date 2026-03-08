/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

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

struct [[nodiscard]] kernel {
 private:
  rtcx::library _library;
  rtcx::kernel_ref _kernel;

 public:
  kernel(rtcx::library lib);
  kernel(kernel const&)            = default;
  kernel(kernel&&)                 = default;
  kernel& operator=(kernel const&) = default;
  kernel& operator=(kernel&&)      = default;
  ~kernel()                        = default;

  rtcx::kernel_ref get() const { return _kernel; }

  rtcx::kernel_occupancy_config max_occupancy_config(size_t dynamic_shared_memory_bytes,
                                                     int32_t block_size_limit) const
  {
    return _kernel.max_occupancy_config(dynamic_shared_memory_bytes, block_size_limit);
  }

  void launch(uint32_t grid_dim_x,
              uint32_t grid_dim_y,
              uint32_t grid_dim_z,
              uint32_t block_dim_x,
              uint32_t block_dim_y,
              uint32_t block_dim_z,
              uint32_t shared_mem_bytes,
              rmm::cuda_stream_view stream,
              std::span<void*> kernel_params) const
  {
    return _kernel.launch(grid_dim_x,
                          grid_dim_y,
                          grid_dim_z,
                          block_dim_x,
                          block_dim_y,
                          block_dim_z,
                          shared_mem_bytes,
                          stream.value(),
                          kernel_params.data());
  }
};

kernel get_kernel(std::string const& name,
                  std::string const& key,
                  std::string const& cuda_udf,
                  std::span<char const* const> header_include_names,
                  std::span<char const* const> headers,
                  char const* name_expression,
                  bool use_cache = true,
                  bool use_pch   = true,
                  bool log_pch   = false);

}  // namespace CUDF_EXPORT cudf
