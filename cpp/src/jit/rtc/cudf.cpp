
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <cudf_jit_embed.h>
#include <jit/rtc/cache.hpp>
#include <jit/rtc/cudf.hpp>
#include <jit/rtc/rtc.hpp>
#include <runtime/context.hpp>

#include <format>
#include <future>

namespace cudf {
namespace rtc {

int32_t get_driver_version()
{
  int32_t driver_version;
  CUDF_EXPECTS(cudaDriverGetVersion(&driver_version) == cudaSuccess,
               "Failed to get CUDA driver version");
  return driver_version;
}

int32_t get_runtime_version()
{
  int32_t runtime_version;
  CUDF_EXPECTS(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess,
               "Failed to get CUDA runtime version");
  return runtime_version;
}

int32_t get_current_device_physical_model()
{
  int32_t device;
  CUDF_EXPECTS(cudaGetDevice(&device) == cudaSuccess, "Failed to get current CUDA device");

  cudaDeviceProp props;
  CUDF_EXPECTS(cudaGetDeviceProperties(&props, device) == cudaSuccess,
               "Failed to get device properties");

  return props.major * 10 + props.minor;
}

void max_occupancy_config()
{
  // [ ] Same as configure_1d_max_occupancy
}

sha256_hash hash_string(std::span<char const> input)
{
  sha256_context ctx;
  ctx.update(std::span{reinterpret_cast<uint8_t const*>(input.data()), input.size()});
  return ctx.finalize();
}

cache_t& get_rtc_cache() { return cudf::get_context().rtc_cache(); }

fragment_t const& compile_fragment(char const* name, char const* source_code_cstr, char const* key)
{
  auto sm              = get_current_device_physical_model();
  auto const cache_key = std::format(R"***(
      fragment_type=LTO_IR,
      key={},
      cuda_runtime={},
      cuda_driver={},
      arch={})***",
                                     key,
                                     get_runtime_version(),
                                     get_driver_version(),
                                     sm);

  auto const cache_key_sha256 = hash_string(cache_key);

  auto& cache = get_rtc_cache();

  if (auto frag = cache.query_fragment(cache_key_sha256); frag.has_value()) {
    return *frag->get();
  } else if (auto disk_frag = cache.query_blob_from_disk(cache_key_sha256); disk_frag.has_value()) {
    std::promise<fragment> prom;
    auto fut = prom.get_future();
    {
      cache.store_fragment(cache_key_sha256, prom.get_future());
      fragment_t::load_params load_params{.binary = *disk_frag, .type = binary_type::LTO_IR};
      auto frag = fragment_t::load(load_params);
      prom.set_value(std::move(frag));
    }
    return *fut.get();
  }

  std::promise<fragment> prom;
  cache.store_fragment(cache_key_sha256, prom.get_future());
  auto fut = prom.get_future();

  auto const headers = header_map{
    .include_names =
      std::span{reinterpret_cast<const char* const*>(cudf_jit_embed_sources.include_names),
                cudf_jit_embed_sources.num_includes},
    .headers = std::span{reinterpret_cast<const char* const*>(cudf_jit_embed_sources.headers),
                         cudf_jit_embed_sources.num_includes},
    .header_sizes =
      std::span{cudf_jit_embed_sources.header_sizes, cudf_jit_embed_sources.num_includes}};

  std::vector<char const*> options;
  auto embed_options = reinterpret_cast<const char* const*>(cudf_jit_embed_options.elements);
  std::copy(
    embed_options, embed_options + cudf_jit_embed_options.size, std::back_inserter(options));
  auto arch_flag = std::format("-arch=sm_{}", sm);
  options.push_back(arch_flag.c_str());
  options.push_back("-dlto");
  options.push_back("-rdc=true");

  auto const params = fragment_t::compile_params{.name        = name,
                                                 .source      = source_code_cstr,
                                                 .headers     = headers,
                                                 .options     = options,
                                                 .target_type = binary_type::LTO_IR};

  auto frag = fragment_t::compile(params);

  prom.set_value(std::move(frag));
  return *fut.get();
}

fragment_t const& compile_library_fragment()
{
  return compile_fragment("cudf_lto_library",
                          R"***(
 #include "jit/lto/library.inl.cuh"
 )***",
                          "cudf_lto_library");
}

fragment_t const& compile_udf_fragment(char const* source_code_cstr, char const* key)
{
  return compile_fragment("cudf_udf_fragment", source_code_cstr, key);
}

kernel_ref compile_and_link_udf(char const* name,
                                char const* kernel_name,
                                char const* kernel_key,
                                char const* udf_code,
                                char const* udf_key)
{
  auto sm                       = get_current_device_physical_model();
  auto library_key              = std::format(R"***(
      fragment_types=LTO_IR,
      target=CUBIN,
      kernel={},
      udf={},
      cuda_runtime={},
      cuda_driver={},
      arch={})***",
                                 kernel_key,
                                 udf_key,
                                 get_runtime_version(),
                                 get_driver_version(),
                                 sm);
  auto const library_key_sha256 = hash_string(library_key);

  auto& cache = get_rtc_cache();

  // [ ] should probably use query_or_insert
  if (auto lib = cache.query_library(library_key_sha256); lib.has_value()) {
    return lib->get()->get_kernel(kernel_name);
  } else if (auto disk_lib = cache.query_blob_from_disk(library_key_sha256); disk_lib.has_value()) {
    std::promise<library> prom;
    auto fut = prom.get_future();

    {
      cache.store_library(library_key_sha256, prom.get_future());
      library_t::load_params load_params{.binary = (*disk_lib)->view(), .type = binary_type::CUBIN};
      auto lib = library_t::load(load_params);
      prom.set_value(std::move(lib));
    }

    return fut.get()->get_kernel(kernel_name);
  }

  auto const& library_frag = compile_library_fragment();
  auto const& udf_frag     = compile_udf_fragment(udf_code, udf_key);

  std::promise<library> prom;
  cache.store_library(library_key_sha256, prom.get_future());
  auto fut = prom.get_future();

  blob_view const link_fragments[]          = {library_frag.get_cubin()->view(),
                                               udf_frag.get_cubin()->view()};
  binary_type const fragment_binary_types[] = {binary_type::LTO_IR, binary_type::LTO_IR};

  char const* const fragment_names[] = {"cudf_lto_library", "cudf_udf_fragment"};

  auto arch_flag                   = std::format("-arch=sm_{}", sm);
  char const* const link_options[] = {"-lto", arch_flag.c_str()};

  auto const params = library_t::link_params{.name                  = name,
                                             .output_type           = binary_type::CUBIN,
                                             .fragments             = link_fragments,
                                             .fragment_binary_types = fragment_binary_types,
                                             .fragment_names        = fragment_names,
                                             .link_options          = link_options};

  auto lib = library_t::link(params);
  prom.set_value(std::move(lib));
  return fut.get()->get_kernel(kernel_name);
}

}  // namespace rtc
}  // namespace cudf
