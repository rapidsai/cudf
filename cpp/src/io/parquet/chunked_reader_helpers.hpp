/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstddef>

namespace CUDF_EXPORT cudf {
namespace io::parquet::detail {

[[nodiscard]] std::size_t derive_pass_read_limit(std::size_t chunk_read_limit);

}  // namespace io::parquet::detail
}  // namespace CUDF_EXPORT cudf
