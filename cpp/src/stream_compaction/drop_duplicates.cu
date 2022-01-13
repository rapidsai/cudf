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

#include <stream_compaction/stream_compaction_common.cuh>
#include <stream_compaction/stream_compaction_common.hpp>

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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <vector>

namespace cudf {
namespace detail {
std::unique_ptr<table> unordered_drop_duplicates(table_view const& input,
                                                 std::vector<size_type> const& keys,
                                                 null_equality nulls_equal,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  if (0 == input.num_rows() || 0 == input.num_columns() || 0 == keys.size()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);

  auto table_ptr = cudf::table_device_view::create(keys_view, stream);
  auto const num_rows{table_ptr->num_rows()};

  hash_map_type key_map{compute_hash_table_size(num_rows),
                        COMPACTION_EMPTY_KEY_SENTINEL,
                        COMPACTION_EMPTY_VALUE_SENTINEL,
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};

  compaction_hash hash_key{nullate::DYNAMIC{cudf::has_nulls(keys_view)}, *table_ptr};
  row_equality_comparator row_equal(
    nullate::DYNAMIC{cudf::has_nulls(keys_view)}, *table_ptr, *table_ptr, nulls_equal);

  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(std::move(i), std::move(i)); });
  key_map.insert(iter, iter + num_rows, hash_key, row_equal, stream.value());

  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  rmm::device_uvector<bool> existences(num_rows, stream, mr);
  key_map.contains(counting_iter, counting_iter + num_rows, existences.begin(), hash_key);

  auto const output_size{key_map.get_size()};

  rmm::device_uvector<size_type> unique_indices(output_size, stream, mr);
  thrust::copy_if(rmm::exec_policy(stream),
                  counting_iter,
                  counting_iter + num_rows,
                  existences.begin(),
                  unique_indices.begin(),
                  [] __device__(bool const b) { return b; });

  column_view unique_indices_view(data_type{type_id::INT32}, output_size, unique_indices.data());

  // run gather operation to establish new order
  return detail::gather(input,
                        unique_indices_view,
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail

std::unique_ptr<table> unordered_drop_duplicates(table_view const& input,
                                                 std::vector<size_type> const& keys,
                                                 null_equality nulls_equal,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::unordered_drop_duplicates(input, keys, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
