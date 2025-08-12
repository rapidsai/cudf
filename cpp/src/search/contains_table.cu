/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cudf/detail/search.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <algorithm>

namespace cudf::detail {

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(haystack, needles), "Column types mismatch");

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  auto const preprocessed_needles =
    cudf::experimental::row::equality::preprocessed_table::create(needles, stream);
  auto const preprocessed_haystack =
    cudf::experimental::row::equality::preprocessed_table::create(haystack, stream);

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Only use primitive row operators for non-floating-point types since they don't handle NaN
  // equality
  auto const has_floating_point =
    std::any_of(haystack.begin(), haystack.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
    });
  if (cudf::is_primitive_row_op_compatible(haystack) && !has_floating_point) {
    auto const d_haystack_hasher =
      cudf::row::primitive::row_hasher{nullate::DYNAMIC{has_any_nulls}, preprocessed_haystack};
    auto const d_needle_hasher =
      cudf::row::primitive::row_hasher{nullate::DYNAMIC{has_any_nulls}, preprocessed_needles};
    auto const d_hasher     = hasher_adapter{d_haystack_hasher, d_needle_hasher};
    auto const d_self_equal = cudf::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_haystack, preprocessed_haystack, compare_nulls};
    auto const d_two_table_equal = cudf::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_needles, preprocessed_haystack, compare_nulls};
    auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};
    perform_contains(haystack,
                     needles,
                     haystack_has_nulls,
                     needles_has_nulls,
                     compare_nulls,
                     d_equal,
                     cuco::linear_probing<1, decltype(d_hasher)>{d_hasher},
                     contained,
                     stream);
  } else {
    auto const haystack_hasher   = cudf::experimental::row::hash::row_hasher(preprocessed_haystack);
    auto const d_haystack_hasher = haystack_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
    auto const needle_hasher     = cudf::experimental::row::hash::row_hasher(preprocessed_needles);
    auto const d_needle_hasher   = needle_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
    auto const d_hasher          = hasher_adapter{d_haystack_hasher, d_needle_hasher};

    auto const self_equal =
      cudf::experimental::row::equality::self_comparator(preprocessed_haystack);
    auto const two_table_equal = cudf::experimental::row::equality::two_table_comparator(
      preprocessed_needles, preprocessed_haystack);

    if (cudf::detail::has_nested_columns(haystack)) {
      dispatch_nan_comparator<true>(haystack,
                                    needles,
                                    compare_nulls,
                                    compare_nans,
                                    haystack_has_nulls,
                                    needles_has_nulls,
                                    has_any_nulls,
                                    self_equal,
                                    two_table_equal,
                                    d_hasher,
                                    contained,
                                    stream);
    } else {
      dispatch_nan_comparator<false>(haystack,
                                     needles,
                                     compare_nulls,
                                     compare_nans,
                                     haystack_has_nulls,
                                     needles_has_nulls,
                                     has_any_nulls,
                                     self_equal,
                                     two_table_equal,
                                     d_hasher,
                                     contained,
                                     stream);
    }
  }
  return contained;
}

}  // namespace cudf::detail
