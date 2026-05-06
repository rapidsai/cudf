/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>

#include <driver_types.h>
#include <vector_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace cudf::detail::jit_lto {

struct AlgorithmLauncher {
  AlgorithmLauncher() : kernel{nullptr}, library{nullptr} {}

  AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib);

  ~AlgorithmLauncher();

  AlgorithmLauncher(const AlgorithmLauncher&)            = delete;
  AlgorithmLauncher& operator=(const AlgorithmLauncher&) = delete;

  AlgorithmLauncher(AlgorithmLauncher&& other) noexcept;
  AlgorithmLauncher& operator=(AlgorithmLauncher&& other) noexcept;

  template <typename FuncT, typename... Args>
  void dispatch(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    static_assert(std::is_same_v<FuncT, void(std::remove_reference_t<Args>...)>,
                  "dispatch() argument types do not match the kernel function signature FuncT");

    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaKernel_t kernel;
  cudaLibrary_t library;
};

}  // namespace cudf::detail::jit_lto
