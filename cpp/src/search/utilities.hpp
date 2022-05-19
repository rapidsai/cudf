/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf::detail {

/**
 * @brief Generate a column_view object having one row that views the content of the input scalar as
 * its (unique) row.
 *
 * This utility function is a helper for comparing rows of a column to a scalar using
 * `two_table_comparator`, which requires two input tables having the same structure. Since this
 * comparator is called only for nested types (non-nested types are compared by simpler
 * comparator(s) for better performance), only scalar of nested types will be processed.
 *
 * In addition, the input scalar is expected to be valid. However, no validity check will be
 * performed. Invalid scalar should be checked and handled at the caller site.
 *
 * @param input The input scalar to view.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return A pair of column_view and the auxiliary rmm::device_uvector allocated for storing column
 *         offsets (if applicable).
 */
std::pair<column_view, std::optional<rmm::device_uvector<offset_type>>>
nested_type_scalar_to_column_view(cudf::scalar const& input, rmm::cuda_stream_view stream);

}  // namespace cudf::detail
