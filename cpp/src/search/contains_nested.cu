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

#include <cudf/detail/iterator.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>

namespace cudf::detail {

bool contains_nested_element(column_view const& haystack,
                             column_view const& needle,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(needle.size() > 0, "Input needle column should have at least ONE row.");

  auto const haystack_tv = table_view{{haystack}};
  auto const needle_tv   = table_view{{needle}};
  auto const has_nulls   = has_nested_nulls(haystack_tv) || has_nested_nulls(needle_tv);

  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator(haystack_tv, needle_tv, stream);
  auto const d_comp = comparator.equal_to(nullate::DYNAMIC{has_nulls});

  auto const begin = cudf::experimental::row::lhs_iterator(0);
  auto const end   = begin + haystack.size();
  using cudf::experimental::row::rhs_index_type;

  if (haystack.has_nulls()) {
    auto const haystack_cdv_ptr  = column_device_view::create(haystack, stream);
    auto const haystack_valid_it = cudf::detail::make_validity_iterator<false>(*haystack_cdv_ptr);

    return thrust::count_if(rmm::exec_policy(stream),
                            begin,
                            end,
                            [d_comp, haystack_valid_it] __device__(auto const idx) {
                              if (!haystack_valid_it[static_cast<size_type>(idx)]) { return false; }
                              return d_comp(
                                idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
                            }) > 0;
  }

  return thrust::count_if(
           rmm::exec_policy(stream), begin, end, [d_comp] __device__(auto const idx) {
             return d_comp(idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
           }) > 0;
}

}  // namespace cudf::detail
