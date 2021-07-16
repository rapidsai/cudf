/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {
template <bool nested_nulls>
void generate_dense_rank_struct_comparisons(column_view const& order_by,
                                            mutable_column_view out,
                                            device_span<size_type const> group_labels,
                                            cudf::device_span<size_type const> group_offsets,
                                            rmm::cuda_stream_view stream)
{
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [d_order_by = *column_device_view::create(order_by, stream),
     has_nulls  = order_by.has_nulls(),
     labels     = group_labels.data(),
     offsets    = group_offsets.data()] __device__(size_type row_index) {
      if (row_index == offsets[labels[row_index]]) {
        return 1;
      } else if (has_nulls) {
        bool const lhs_is_null{d_order_by.is_null_nocheck(row_index)};
        bool const rhs_is_null{d_order_by.is_null_nocheck(row_index - 1)};
        if (lhs_is_null and rhs_is_null) {
          return 0;
        } else if (lhs_is_null != rhs_is_null) {
          return 1;
        }
      }

      return thrust::all_of(
               thrust::seq,
               thrust::make_counting_iterator<size_type>(0),
               thrust::make_counting_iterator<size_type>(d_order_by.num_child_columns()),
               [row_index, d_order_by] __device__(size_type child_index) {
                 column_device_view col = d_order_by.child(child_index);
                 element_equality_comparator<nested_nulls> element_comparator{col, col, true};
                 return cudf::type_dispatcher(
                   col.type(), element_comparator, row_index, row_index - 1);
               })
               ? 0
               : 1;
    });
}

template <bool has_nulls>
void generate_dense_rank_comparisons(column_view const& order_by,
                                     mutable_column_view out,
                                     device_span<size_type const> group_labels,
                                     cudf::device_span<size_type const> group_offsets,
                                     rmm::cuda_stream_view stream)
{
  auto d_order_by = column_device_view::create(order_by, stream);
  element_equality_comparator<has_nulls> element_comparator(*d_order_by, *d_order_by, true);
  thrust::tabulate(
    rmm::exec_policy(stream),
    out.begin<size_type>(),
    out.end<size_type>(),
    [type = d_order_by->type(),
     element_comparator,
     labels  = group_labels.data(),
     offsets = group_offsets.data()] __device__(size_type row_index) {
      return (row_index == offsets[labels[row_index]] ||
              !cudf::type_dispatcher(type, element_comparator, row_index, row_index - 1))
               ? 1
               : 0;
    });
}

bool has_nested_nulls(column_view const& struct_col)
{
  return struct_col.has_nulls() || std::any_of(struct_col.child_begin(),
                                               struct_col.child_end(),
                                               [](auto col) { return has_nested_nulls(col); });
}
}  // namespace
std::unique_ptr<column> dense_rank_scan(column_view const& order_by,
                                        cudf::device_span<size_type const> group_labels,
                                        cudf::device_span<size_type const> group_offsets,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto ranks         = make_fixed_width_column(cudf::data_type{cudf::type_to_id<size_type>()},
                                       order_by.size(),
                                       mask_state::ALL_VALID,
                                       stream,
                                       mr);
  auto mutable_ranks = ranks->mutable_view();
  if (order_by.type().id() == type_id::STRUCT) {
    bool nested_nulls = std::any_of(
      order_by.child_begin(), order_by.child_end(), [](auto col) { return has_nested_nulls(col); });
    bool is_nested = std::any_of(order_by.child_begin(), order_by.child_end(), [](auto col) {
      return col.type().id() == type_id::STRUCT || col.type().id() == type_id::LIST;
    });

    if (is_nested) {
      CUDF_FAIL("Nested struct and list types not supported");
    } else if (nested_nulls) {
      generate_dense_rank_struct_comparisons<true>(
        order_by, mutable_ranks, group_labels, group_offsets, stream);
    } else {
      generate_dense_rank_struct_comparisons<false>(
        order_by, mutable_ranks, group_labels, group_offsets, stream);
    }
  } else if (order_by.has_nulls()) {
    generate_dense_rank_comparisons<true>(
      order_by, mutable_ranks, group_labels, group_offsets, stream);
  } else {
    generate_dense_rank_comparisons<false>(
      order_by, mutable_ranks, group_labels, group_offsets, stream);
  }

  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_labels.begin(),
                                group_labels.end(),
                                mutable_ranks.begin<size_type>(),
                                mutable_ranks.begin<size_type>());
  return ranks;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
