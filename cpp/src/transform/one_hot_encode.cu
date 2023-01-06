/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type() == categories.type(), "Mismatch type between input and categories.");

  if (categories.is_empty()) { return std::pair(make_empty_column(type_id::BOOL8), table_view{}); }

  if (input.is_empty()) {
    auto empty_data = make_empty_column(type_id::BOOL8);
    std::vector<column_view> views(categories.size(), empty_data->view());
    return std::pair(std::move(empty_data), table_view{views});
  }

  auto const total_size = input.size() * categories.size();
  auto all_encodings =
    make_numeric_column(data_type{type_id::BOOL8}, total_size, mask_state::UNALLOCATED, stream, mr);

  auto const t_lhs = table_view{{input}};
  auto const t_rhs = table_view{{categories}};
  auto const comparator =
    cudf::experimental::row::equality::two_table_comparator{t_lhs, t_rhs, stream};
  auto const d_equal =
    comparator.equal_to(nullate::DYNAMIC{has_nested_nulls(t_lhs) || has_nested_nulls(t_rhs)});

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_size),
    all_encodings->mutable_view().begin<bool>(),
    [input_size = input.size(), d_equal] __device__(size_type i) {
      auto const element_index  = cudf::experimental::row::lhs_index_type{i % input_size};
      auto const category_index = cudf::experimental::row::rhs_index_type{i / input_size};
      return d_equal(element_index, category_index);
    });

  auto const split_iter =
    make_counting_transform_iterator(1, [width = input.size()](auto i) { return i * width; });
  std::vector<size_type> split_indices(split_iter, split_iter + categories.size() - 1);

  auto encodings_view = table_view{split(all_encodings->view(), split_indices, stream)};

  return std::pair(std::move(all_encodings), encodings_view);
}

}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::one_hot_encode(input, categories, cudf::get_default_stream(), mr);
}
}  // namespace cudf
