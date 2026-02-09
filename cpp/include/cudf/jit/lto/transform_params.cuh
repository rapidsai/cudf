
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
  void* __restrict__ const* __restrict__ scope =
    nullptr;                ///< Pointer to scope data (e.g. column views, scalars, etc.).
  size_type row_index = 0;  ///< Current row index.
};

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
