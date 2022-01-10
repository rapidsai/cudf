/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <numeric>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<table>, std::unique_ptr<column>> encode(
  table_view const& input_table, rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{
  std::vector<size_type> drop_keys(input_table.num_columns());
  std::iota(drop_keys.begin(), drop_keys.end(), 0);

  // side effects of this function we are now dependent on:
  // - resulting column elements are sorted ascending
  // - nulls are sorted to the beginning
  auto keys_table = cudf::detail::drop_duplicates(input_table,
                                                  drop_keys,
                                                  duplicate_keep_option::KEEP_FIRST,
                                                  null_equality::EQUAL,
                                                  null_order::AFTER,
                                                  stream,
                                                  mr);

  auto indices_column =
    cudf::detail::lower_bound(keys_table->view(),
                              input_table,
                              std::vector<order>(input_table.num_columns(), order::ASCENDING),
                              std::vector<null_order>(input_table.num_columns(), null_order::AFTER),
                              stream,
                              mr);

  return std::make_pair(std::move(keys_table), std::move(indices_column));
}

}  // namespace detail

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, rmm::mr::device_memory_resource* mr)
{
  return detail::encode(input, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
