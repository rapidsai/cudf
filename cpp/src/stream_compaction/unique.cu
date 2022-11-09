/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "stream_compaction_common.cuh"
#include "stream_compaction_common.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {
std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  // If keep is KEEP_ANY, just alias it to KEEP_FIRST.
  if (keep == duplicate_keep_option::KEEP_ANY) { keep = duplicate_keep_option::KEEP_FIRST; }

  auto const num_rows = input.num_rows();
  if (num_rows == 0 or input.num_columns() == 0 or keys.empty()) { return empty_like(input); }

  auto unique_indices = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_view     = mutable_column_device_view::create(*unique_indices, stream);
  auto keys_view        = input.select(keys);
  auto keys_device_view = cudf::table_device_view::create(keys_view, stream);
  auto row_equal        = row_equality_comparator(nullate::DYNAMIC{cudf::has_nulls(keys_view)},
                                           *keys_device_view,
                                           *keys_device_view,
                                           nulls_equal);

  // get indices of unique rows
  auto result_end = unique_copy(thrust::counting_iterator<size_type>(0),
                                thrust::counting_iterator<size_type>(num_rows),
                                mutable_view->begin<size_type>(),
                                row_equal,
                                keep,
                                stream);
  auto indices_view =
    cudf::detail::slice(column_view(*unique_indices),
                        0,
                        thrust::distance(mutable_view->begin<size_type>(), result_end));

  // gather unique rows and return
  return detail::gather(input,
                        indices_view,
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}
}  // namespace detail

std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option const keep,
                              null_equality nulls_equal,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::unique(input, keys, keep, nulls_equal, cudf::get_default_stream(), mr);
}

}  // namespace cudf
