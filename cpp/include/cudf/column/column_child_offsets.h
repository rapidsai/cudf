/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {

static constexpr size_type offsets_column_index = 0;  ///< Child index of the offsets column

static constexpr size_type dictionary_indices_column_index =
  0;  ///< Child index of the dictionary offsets column

static constexpr size_type dictionary_keys_column_index =
  1;  ///< Child index of the dictionary key column

}  // namespace CUDF_EXPORT cudf
