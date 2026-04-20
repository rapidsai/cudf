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
  kernel(rtcx::library lib, rtcx::kernel_ref kernel) : _library(std::move(lib)), _kernel(kernel) {};
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

  void launch(rtcx::cuda_dim3 grid_dim,
              rtcx::cuda_dim3 block_dim,
              uint32_t shared_mem_bytes,
              rmm::cuda_stream_view stream,
              void** kernel_params) const
  {
    return _kernel.launch(grid_dim, block_dim, shared_mem_bytes, stream.value(), kernel_params);
  }

  template <typename... Args>
  void launch_with(rtcx::cuda_dim3 grid_dim,
                   rtcx::cuda_dim3 block_dim,
                   uint32_t shared_mem_bytes,
                   rmm::cuda_stream_view stream,
                   Args&&... args)
  {
    void const* params[] = {&args...};  // NOLINT(modernize-avoid-c-arrays)
    launch(grid_dim, block_dim, shared_mem_bytes, stream, const_cast<void**>(params));
  }
};

kernel get_kernel(std::string const& name,
                  std::string const& source_file,
                  std::span<char const* const> header_include_names,
                  std::span<char const* const> headers,
                  std::string const& kernel_instance,
                  bool use_cache                             = true,
                  bool use_pch                               = true,
                  bool use_minimal                           = true,
                  bool log_pch                               = false,
                  std::span<std::string const> extra_options = {});

}  // namespace CUDF_EXPORT cudf
