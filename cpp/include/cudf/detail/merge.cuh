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

/**
 * @brief The equivalent of `row_lexicographic_comparator` for tagged indices.
 */
template <typename LeftComparator, typename LeftRightComparator, typename RightComparator>
struct row_lexicographic_tagged_comparator {
  row_lexicographic_tagged_comparator(LeftComparator left_comp,
                                      LeftRightComparator left_right_comp,
                                      RightComparator right_comp)
    : _left_comp{left_comp}, _left_right_comp{left_right_comp}, _right_comp{right_comp}
  {
  }

  __device__ bool operator()(index_type lhs_tagged_index,
                             index_type rhs_tagged_index) const noexcept
  {
    using cudf::experimental::row::lhs_index_type;
    using cudf::experimental::row::rhs_index_type;

    auto const [l_side, l_indx] = lhs_tagged_index;
    auto const [r_side, r_indx] = rhs_tagged_index;

    if (l_side == side::LEFT && r_side == side::RIGHT) {
      return _left_right_comp(lhs_index_type{l_indx}, rhs_index_type{r_indx});
    } else if (l_side == side::RIGHT && r_side == side::LEFT) {
      return _left_right_comp(rhs_index_type{l_indx}, lhs_index_type{r_indx});
    } else if (l_side == side::LEFT && r_side == side::LEFT) {
      return _left_comp(l_indx, r_indx);
    } else if (l_side == side::RIGHT && r_side == side::RIGHT) {
      return _right_comp(l_indx, r_indx);
    }
    return false;
  }

 private:
  LeftComparator _left_comp;
  LeftRightComparator _left_right_comp;
  RightComparator _right_comp;
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
