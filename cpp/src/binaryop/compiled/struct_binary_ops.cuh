/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

namespace cudf::binops::compiled::detail {
/**
 * @brief Generates comparison results for each row in the output column. Supports scalar columns
 * and negation of comparison results to mimic !=, <=, and >= operators.
 *
 * @tparam Comparator comparator type
 * @param out mutable column view of output column
 * @param compare initialized comparator function
 * @param is_lhs_scalar true if @p compare has a single element column representing a scalar on its
 * lhs
 * @param is_rhs_scalar true if @p compare has a single element column representing a scalar on its
 * rhs
 * @param flip_output true if the comparison results should be negated
 * @param stream CUDA stream used for device memory operations
 */
template <typename Comparator>
void struct_compare(mutable_column_view& out,
                    Comparator compare,
                    bool is_lhs_scalar,
                    bool is_rhs_scalar,
                    bool flip_output,
                    rmm::cuda_stream_view stream);
}  //  namespace cudf::binops::compiled::detail
