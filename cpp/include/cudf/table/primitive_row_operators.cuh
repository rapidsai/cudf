/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * @file primitive_row_operators.cuh
 * @brief DEPRECATED: This header is deprecated as of 25.10 and will be removed in 25.12.
 *
 * Users should migrate to using `cudf::detail::row_operator::primitive_row_operators.cuh` directly
 * and access the primitive row operators in the `cudf::detail::row::primitive` namespace.
 *
 * @deprecated This header will be removed in cuDF 25.12. Use
 * `cudf::detail::row_operator::primitive_row_operators.cuh` instead.
 */

#include <cudf/detail/row_operator/primitive_row_operators.cuh>

namespace CUDF_EXPORT cudf {

/**
 * @brief DEPRECATED: Use `cudf::detail::is_primitive_row_op_compatible` instead.
 * @deprecated This function will be removed in cuDF 25.12. Use
 * `cudf::detail::is_primitive_row_op_compatible` instead.
 */
using detail::is_primitive_row_op_compatible;

}  // namespace CUDF_EXPORT cudf
