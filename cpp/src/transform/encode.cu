/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <numeric>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<table>, std::unique_ptr<column>> encode(
  table_view const& input_table, rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  std::vector<size_type> drop_keys(input_table.num_columns());
  std::iota(drop_keys.begin(), drop_keys.end(), 0);

  // side effects of this function we are now dependent on:
  // - resulting column elements are sorted ascending
  // - nulls are sorted to the beginning
  auto keys_table = cudf::detail::drop_duplicates(
    input_table, drop_keys, duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL, mr, stream);

  if (cudf::has_nulls(keys_table->view())) {
    // Rows with nulls appear at the top of `keys_table`, but we want them to appear at
    // the bottom. Below, we rearrange the rows so that nulls appear at the bottom:
    // TODO: we should be able to get rid of this logic once
    // https://github.com/rapidsai/cudf/issues/6144 is resolved

    auto num_rows = keys_table->num_rows();
    auto mask =
      cudf::detail::bitmask_and(keys_table->view(), stream, rmm::mr::get_current_device_resource());
    auto num_rows_with_nulls =
      cudf::count_unset_bits(reinterpret_cast<bitmask_type*>(mask.data()), 0, num_rows);

    rmm::device_vector<cudf::size_type> gather_map(num_rows);
    auto execpol = rmm::exec_policy(stream);
    thrust::transform(execpol->on(stream),
                      thrust::make_counting_iterator<cudf::size_type>(0),
                      thrust::make_counting_iterator<cudf::size_type>(num_rows),
                      gather_map.begin(),
                      [num_rows, num_rows_with_nulls] __device__(cudf::size_type i) {
                        if (i < (num_rows - num_rows_with_nulls)) {
                          return num_rows_with_nulls + i;
                        } else {
                          return num_rows - i - 1;
                        }
                      });

    cudf::column_view gather_map_column(
      cudf::data_type{type_id::INT32}, num_rows, thrust::raw_pointer_cast(gather_map.data()));

    keys_table = cudf::detail::gather(keys_table->view(),
                                      gather_map_column,
                                      cudf::detail::out_of_bounds_policy::FAIL,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED,
                                      mr,
                                      stream);
  }

  auto indices_column =
    cudf::detail::lower_bound(keys_table->view(),
                              input_table,
                              std::vector<order>(input_table.num_columns(), order::ASCENDING),
                              std::vector<null_order>(input_table.num_columns(), null_order::AFTER),
                              mr,
                              stream);

  return std::make_pair(std::move(keys_table), std::move(indices_column));
}

}  // namespace detail

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, rmm::mr::device_memory_resource* mr)
{
  return detail::encode(input, mr, 0);
}

}  // namespace cudf
