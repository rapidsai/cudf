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
#include <cudf/detail/gather.cuh>
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
#include <thrust/iterator/discard_iterator.h>

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

#if 0
  auto const output_size{key_map.get_size()};
  auto distinct_indices = cudf::make_numeric_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr);
  // write distinct indices to a numeric column
  key_map.retrieve_all(distinct_indices->mutable_view().begin<cudf::size_type>(),
                       thrust::make_discard_iterator(),
                       stream.value());
  // run gather operation to establish new order
  return detail::gather(input,
                        distinct_indices->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
#else

  auto const key_size{keys_view.num_rows()};
  auto distinct_indices = rmm::device_uvector<size_type>(key_size, stream);

  // Fill `key_size` if keep_first
  // Fill `INT_MIN` if keep_last
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             distinct_indices.begin(),
                             distinct_indices.begin() + key_size,
                             key_size);

  auto const d_map = key_map.get_device_view();
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(key_size),
    [distinct_indices = distinct_indices.begin(), d_map, hash_key, key_equal] __device__(auto key) {
      // iter should always be valid, because all keys have been inserted.
      auto const idx =
        d_map.find(key, hash_key, key_equal)->second.load(cuda::std::memory_order_relaxed);
      atomicMin(&distinct_indices[idx], key);
    });

  auto gather_map = rmm::device_uvector<size_type>(key_size, stream);
  auto const gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    distinct_indices.begin(),
                    distinct_indices.end(),
                    gather_map.begin(),
                    [key_size] __device__(auto const idx) { return idx != key_size; });

  return cudf::detail::gather(
    input, gather_map.begin(), gather_map_end, out_of_bounds_policy::DONT_CHECK, stream, mr);
#endif
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
