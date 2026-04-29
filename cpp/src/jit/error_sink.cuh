/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/operators/error.hpp>
#include <cudf/operators/types.cuh>

#include <cuda/std/atomic>

namespace cudf {
namespace jit {

struct error_sink {
  ops::errc any_error_ = ops::errc::OK;

  constexpr error_sink() = default;

  template <ops::error_mode mode>
  __device__ void report(ops::errc error)
  {
    if constexpr (mode == ops::error_mode::IGNORE) {
      return;
    } else {
      if (error != ops::errc::OK) [[unlikely]] {
        cuda::std::atomic_ref any_error_ref{any_error_};
        any_error_ref.store(error, cuda::std::memory_order_relaxed);
      }
    }
  }

  [[nodiscard]] __host__ __device__ constexpr ops::errc any_error() const { return any_error_; }
};

}  // namespace jit
}  // namespace cudf
