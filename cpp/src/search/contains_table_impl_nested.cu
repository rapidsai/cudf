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

#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::detail {

// Explicit instantiations to reduce build time
using hasher_adapter_t = hasher_adapter<
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   nullate::DYNAMIC>,
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   nullate::DYNAMIC>>;

template void dispatch_nan_comparator<true, hasher_adapter_t>(
  table_view const& haystack,
  table_view const& needles,
  null_equality compare_nulls,
  nan_equality compare_nans,
  bool haystack_has_nulls,
  bool needles_has_nulls,
  bool has_any_nulls,
  cudf::experimental::row::equality::self_comparator self_equal,
  cudf::experimental::row::equality::two_table_comparator two_table_equal,
  hasher_adapter_t const& d_hasher,
  rmm::device_uvector<bool>& contained,
  rmm::cuda_stream_view stream);

// Explicit instantiations for perform_contains with nested types (experimental row operations)

// For HasNested=true (nested columns) with nan_equal_comparator
using nan_equal_self_comparator_nested = cudf::experimental::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;

using nan_equal_two_table_comparator_nested =
  cudf::experimental::row::equality::strong_index_comparator_adapter<
    nan_equal_self_comparator_nested>;

using nan_equal_comparator_adapter_nested =
  comparator_adapter<nan_equal_self_comparator_nested, nan_equal_two_table_comparator_nested>;

template void perform_contains(table_view const& haystack,
                               table_view const& needles,
                               bool haystack_has_nulls,
                               bool needles_has_nulls,
                               null_equality compare_nulls,
                               nan_equal_comparator_adapter_nested const& d_equal,
                               cuco::linear_probing<4, hasher_adapter_t> const& probing_scheme,
                               rmm::device_uvector<bool>& contained,
                               rmm::cuda_stream_view stream);

// For HasNested=true (nested columns) with nan_unequal_comparator
using nan_unequal_self_comparator_nested = cudf::experimental::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::experimental::row::equality::physical_equality_comparator>;

using nan_unequal_two_table_comparator_nested =
  cudf::experimental::row::equality::strong_index_comparator_adapter<
    nan_unequal_self_comparator_nested>;

using nan_unequal_comparator_adapter_nested =
  comparator_adapter<nan_unequal_self_comparator_nested, nan_unequal_two_table_comparator_nested>;

template void perform_contains(table_view const& haystack,
                               table_view const& needles,
                               bool haystack_has_nulls,
                               bool needles_has_nulls,
                               null_equality compare_nulls,
                               nan_unequal_comparator_adapter_nested const& d_equal,
                               cuco::linear_probing<4, hasher_adapter_t> const& probing_scheme,
                               rmm::device_uvector<bool>& contained,
                               rmm::cuda_stream_view stream);

}  // namespace cudf::detail
