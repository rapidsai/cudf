
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/jit/lto/types.cuh>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {

namespace lto {

struct transform_params {
  void const* inputs       = nullptr;
  void* user_data          = nullptr;
  void const* outputs      = nullptr;
  void const* span_outputs = nullptr;
  size_type row_index      = 0;
};

}  // namespace lto
}  // namespace CUDF_EXPORT cudf
