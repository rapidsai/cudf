
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {
namespace lto {

/// @brief Type-erased parameters for LTO-JIT-compiled transform operations.
struct [[nodiscard]] transform_params {
  /// @brief Pointer to scope data (e.g. column views, scalars, etc.).
  void* __restrict__ const* __restrict__ scope = nullptr;

  /// @brief Total number of rows to process.
  size_type num_rows = 0;

  /// @brief Current row index.
  size_type row_index = 0;
};

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
