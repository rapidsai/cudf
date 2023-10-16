/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace cudf {
namespace detail {
/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };

/**
 * @brief Tagged index type: `thrust::get<0>` indicates left/right side,
 * `thrust::get<1>` indicates the row index
 */
using index_type = thrust::pair<side, cudf::size_type>;

/**
 * @brief Vector of `index_type` values.
 */
using index_vector = rmm::device_uvector<index_type>;

template <bool has_nulls>
struct row_lexicographic_tagged_comparator {
  row_lexicographic_tagged_comparator(table_device_view lhs,
                                      table_device_view rhs,
                                      device_span<order const> column_order,
                                      device_span<null_order const> null_precedence)
    : _lhs{lhs}, _rhs{rhs}, _column_order{column_order}, _null_precedence{null_precedence}
  {
    // Add check for types to be the same.
    CUDF_EXPECTS(_lhs.num_columns() == _rhs.num_columns(), "Mismatched number of columns.");
  }

  __device__ bool operator()(index_type lhs_tagged_index,
                             index_type rhs_tagged_index) const noexcept
  {
    auto const [l_side, l_indx] = lhs_tagged_index;
    auto const [r_side, r_indx] = rhs_tagged_index;

    // Not sure why `const_cast` is needed here
    table_device_view* ptr_left_dview{l_side == side::LEFT
                                        ? const_cast<cudf::table_device_view*>(&_lhs)
                                        : const_cast<cudf::table_device_view*>(&_rhs)};
    table_device_view* ptr_right_dview{r_side == side::LEFT
                                         ? const_cast<cudf::table_device_view*>(&_lhs)
                                         : const_cast<cudf::table_device_view*>(&_rhs)};

    auto comparator = [&]() {
      if (has_nulls) {
        return cudf::experimental::row::lexicographic::device_row_comparator<false, bool>{
          has_nulls, *ptr_left_dview, *ptr_right_dview, _column_order, _null_precedence};
      } else {
        return cudf::experimental::row::lexicographic::device_row_comparator<false, bool>{
          has_nulls, *ptr_left_dview, *ptr_right_dview, _column_order};
      }
    }();

    auto weak_order = comparator(l_indx, r_indx);

    return weak_order == weak_ordering::LESS;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  device_span<null_order const> _null_precedence;
  device_span<order const> _column_order;
};

/**
 * @copydoc std::unique_ptr<cudf::table> merge(
 *            std::vector<table_view> const& tables_to_merge,
 *            std::vector<cudf::size_type> const& key_cols,
 *            std::vector<cudf::order> const& column_order,
 *            std::vector<cudf::null_order> const& null_precedence,
 *            rmm::mr::device_memory_resource* mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::table> merge(std::vector<table_view> const& tables_to_merge,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& column_order,
                                   std::vector<cudf::null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace cudf
