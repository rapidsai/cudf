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

#include "contains_table_impl.cuh"

#include <cudf/table/primitive_row_operators.cuh>

namespace cudf::detail {

// Explicit instantiation for perform_contains with primitive row operations
using primitive_hasher_adapter_type =
  hasher_adapter<cudf::row::primitive::row_hasher<>, cudf::row::primitive::row_hasher<>>;

using primitive_comparator_adapter_type =
  comparator_adapter<cudf::row::primitive::row_equality_comparator,
                     cudf::row::primitive::row_equality_comparator>;

template void perform_contains(
  table_view const& haystack,
  table_view const& needles,
  bool haystack_has_nulls,
  bool needles_has_nulls,
  null_equality compare_nulls,
  primitive_comparator_adapter_type const& d_equal,
  cuco::linear_probing<1, primitive_hasher_adapter_type> const& probing_scheme,
  rmm::device_uvector<bool>& contained,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
