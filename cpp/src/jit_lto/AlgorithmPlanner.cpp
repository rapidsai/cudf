/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cudf/detail/jit_lto/nvjitlink_checker.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <nvJitLink.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace cudf::detail::jit_lto {

namespace {

void emit_jit_lto_build_timing(double const build_ms)
{
  CUDF_LOG_INFO("jit_lto: AlgorithmPlanner::build %.6f ms", build_ms);
  if (std::getenv("CUDF_JIT_LTO_LINK_TIMING") != nullptr) {
    std::fprintf(stderr, "CUDF_JIT_LTO_LINK_TIMING build_ms=%.6f\n", build_ms);
    std::fflush(stderr);
  }
}

}  // namespace

std::string AlgorithmPlanner::get_fragments_key() const
{
  std::string key = "";
  for (const auto& fragment : this->fragments) {
    key += fragment->get_key();
  }
  return key;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::read_cache(std::string const& launch_key) const
{
  auto& launchers = jit_cache_.launchers;
  std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }
  return nullptr;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::get_launcher()
{
  auto& launchers = jit_cache_.launchers;
  auto launch_key = this->get_fragments_key();

  if (auto hit = read_cache(launch_key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }

  auto launcher         = this->build();
  launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::build()
{
  using clock       = std::chrono::steady_clock;
  using duration_ms = std::chrono::duration<double, std::milli>;

  auto const t_build_start = clock::now();

  int device = 0;
  int major  = 0;
  int minor  = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  nvJitLinkHandle handle;
  const char* lopts[] = {"-lto", archs.c_str()};
  auto result         = nvJitLinkCreate(&handle, 2, lopts);
  check_nvjitlink_result(handle, result);

  for (const auto& frag : this->fragments) {
    frag->add_to(handle);
  }

  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  CUDF_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  cudaLibrary_t library;
  CUDF_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  cudaKernel_t kernel;
  CUDF_CUDA_TRY(cudaLibraryGetKernel(&kernel, library, this->entrypoint.c_str()));

  double const build_ms = duration_ms(clock::now() - t_build_start).count();
  emit_jit_lto_build_timing(build_ms);

  return std::make_shared<AlgorithmLauncher>(kernel, library);
}

}  // namespace cudf::detail::jit_lto
