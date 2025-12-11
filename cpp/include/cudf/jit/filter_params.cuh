/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

/// @brief Type-erased parameters for LTO-JIT-compiled filter operations.
struct filter_params {
  void const* inputs  = nullptr;  ///< Pointer to inputs data.
  void* user_data     = nullptr;  ///< Pointer to user data / context.
  void const* outputs = nullptr;  ///< Pointer to outputs data.
  size_type row_index = 0;        ///< Current row index.
};

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
