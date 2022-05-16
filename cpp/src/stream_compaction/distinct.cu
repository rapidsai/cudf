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
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {
std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                null_equality nulls_equal,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);
  auto preprocessed_keys =
    cudf::experimental::row::hash::preprocessed_table::create(keys_view, stream);
  auto const has_null = nullate::DYNAMIC{cudf::has_nested_nulls(keys_view)};
  auto const num_rows{keys_view.num_rows()};

  hash_map_type key_map{compute_hash_table_size(num_rows),
                        COMPACTION_EMPTY_KEY_SENTINEL,
                        COMPACTION_EMPTY_VALUE_SENTINEL,
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};

  auto row_hash = cudf::experimental::row::hash::row_hasher(preprocessed_keys);
  experimental::compaction_hash hash_key(row_hash.device_hasher(has_null));

  cudf::experimental::row::equality::self_comparator row_equal(preprocessed_keys);
  auto key_equal = row_equal.device_comparator(has_null, nulls_equal);

  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });
  // insert distinct indices into the map.
  key_map.insert(iter, iter + num_rows, hash_key, key_equal, stream.value());

  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  rmm::device_uvector<bool> index_exists_in_map(num_rows, stream, mr);
  // enumerate all indices to check if they are present in the map.
  key_map.contains(counting_iter, counting_iter + num_rows, index_exists_in_map.begin(), hash_key);

  auto const output_size{key_map.get_size()};

  // write distinct indices to a numeric column
  auto distinct_indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr);
  auto mutable_view = mutable_column_device_view::create(*distinct_indices, stream);
  thrust::copy_if(rmm::exec_policy(stream),
                  counting_iter,
                  counting_iter + num_rows,
                  index_exists_in_map.begin(),
                  mutable_view->begin<size_type>(),
                  thrust::identity<bool>{});

  // run gather operation to establish new order
  return detail::gather(input,
                        distinct_indices->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                null_equality nulls_equal,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, keys, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
