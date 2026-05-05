/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <rtcx.hpp>

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

/**
 * @brief Gets a kernel from an embedded CUDA source file
 * @param name Debug name for the kernel (used for caching and logging)
 * @param source_file_id Identifier for the embedded source file (used to locate the source and for
 * caching)
 * @param header_include_names Names of any additional embedded header files to include during
 * compilation
 * @param headers Contents of any additional embedded header files to include during compilation
 * @param kernel_instance String identifier for the specific kernel instance being requested (used
 * for caching)
 * @param use_cache Whether to use the caching mechanism to avoid redundant compilations
 * @param use_pch Whether to use precompiled headers for faster compilation (if supported)
 * @param use_minimal Whether to use NVRTC's minimal compilation mode for faster compilation (if
 * supported)
 * @param log_pch Whether to log precompiled header usage
 * @param extra_options Additional compilation options
 */
kernel get_kernel(std::string const& name,
                  std::string const& source_file_id,
                  std::span<char const* const> header_include_names,
                  std::span<char const* const> headers,
                  std::string const& kernel_instance,
                  bool use_cache                             = true,
                  bool use_pch                               = true,
                  bool use_minimal                           = true,
                  bool log_pch                               = false,
                  std::span<std::string const> extra_options = {});

/**
 * @brief Gets a kernel by linking together embedded binary fragments
 * @param name Debug name for the kernel (used for caching and logging)
 * @param fragment_file_id Identifier for the embedded fragment file (used to locate the fragments
 * and for caching)
 * @param file_fragments Embedded fragments to link
 * @param memory_fragments Memory fragments to link
 * @param use_cache Whether to use the cache system
 * @param extra_options Additional linking options
 */
kernel get_linked_kernel(std::string const& name,
                         std::string const& fragment_file_id,
                         std::span<rtcx::file_fragment const> file_fragments,
                         std::span<rtcx::memory_fragment const> memory_fragments,
                         bool use_cache                             = true,
                         std::span<std::string const> extra_options = {});

/**
 * @brief Gets a kernel by compiling a CUDA UDF and linking together binary fragments.
 * @param kernel_name Debug name for the kernel (used for caching and logging)
 * @param kernel_fragment_file_id Identifier for the embedded fragment file (used to locate the fragments
 * and for caching)
 * @param cuda_udf_name Name of the CUDA UDF
 * @param cuda_udf_source CUDA source code for the UDF
 * @param file_fragments Embedded fragments to link
 * @param memory_fragments Memory fragments to link
 * @param use_cache Whether to use the cache system
 * @param use_pch Whether to use precompiled headers for faster compilation (if supported)
 * @param use_minimal Whether to use NVRTC's minimal compilation mode for faster compilation (if
 * supported)
 * @param log_pch Whether to log precompiled header usage
 * @param extra_compile_options Additional compilation options
 * @param extra_link_options Additional linking options
 */
kernel get_cuda_linked_kernel(std::string const& kernel_name,
                              std::string const& kernel_fragment_file_id,
                              std::string const& cuda_udf_name,
                              std::string const& cuda_udf_source,
                              std::span<char const* const> header_include_names,
                              std::span<char const* const> headers,
                              std::span<rtcx::file_fragment const> file_fragments,
                              std::span<rtcx::memory_fragment const> memory_fragments,
                              bool use_cache                                     = true,
                              bool use_pch                                       = true,
                              bool use_minimal                                   = true,
                              bool log_pch                                       = false,
                              std::span<std::string const> extra_compile_options = {},
                              std::span<std::string const> extra_link_options    = {});

}  // namespace CUDF_EXPORT cudf
