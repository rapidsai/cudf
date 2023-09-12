/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "distinct_reduce.cuh"

#include <reductions/hash_reduce_by_row.cuh>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

namespace {
/**
 * @brief
 *
 */
template <typename MapView, typename KeyHasher, typename KeyEqual>
struct distinct_reduce_fn : reduce_by_row_fn<MapView, KeyHasher, KeyEqual, size_type> {
  duplicate_keep_option const keep;

  distinct_reduce_fn(MapView const& d_map,
                     KeyHasher const& d_hasher,
                     KeyEqual const& d_equal,
                     duplicate_keep_option const keep,
                     size_type* const d_output)
    : reduce_by_row_fn<MapView, KeyHasher, KeyEqual, size_type>(d_map, d_hasher, d_equal, d_output),
      keep{keep}
  {
  }

  __device__ void operator()(size_type const idx) const
  {
    auto const out_ptr = this->get_output_ptr(idx);

    if (keep == duplicate_keep_option::KEEP_FIRST) {
      // Store the smallest index of all rows that are equal.
      atomicMin(out_ptr, idx);
    } else if (keep == duplicate_keep_option::KEEP_LAST) {
      // Store the greatest index of all rows that are equal.
      atomicMax(out_ptr, idx);
    } else {
      // Count the number of rows in each group of rows that are compared equal.
      atomicAdd(out_ptr, size_type{1});
    }
  }
};

template <duplicate_keep_option keep>
struct reduce_func_builder {
  template <typename MapView, typename KeyHasher, typename KeyEqual>
  static auto build(MapView const& d_map,
                    KeyHasher const& d_hasher,
                    KeyEqual const& d_equal,
                    size_type* const d_output)
  {
    return distinct_reduce_fn<MapView, KeyHasher, KeyEqual>{
      d_map, d_hasher, d_equal, keep, d_output};
  }
};

}  // namespace

rmm::device_uvector<size_type> distinct_reduce(
  hash_map_type const& map,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
  size_type num_rows,
  cudf::nullate::DYNAMIC has_nulls,
  bool has_nested_columns,
  duplicate_keep_option keep,
  null_equality nulls_equal,
  nan_equality nans_equal,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const hash_reduce = [&](auto const& func_builder) {
    return hash_reduce_by_row(map,
                              preprocessed_input,
                              num_rows,
                              has_nulls,
                              has_nested_columns,
                              nulls_equal,
                              nans_equal,
                              func_builder,
                              reduction_init_value(keep),
                              stream,
                              mr);
  };
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST:
      return hash_reduce(reduce_func_builder<duplicate_keep_option::KEEP_FIRST>{});
    case duplicate_keep_option::KEEP_LAST:
      return hash_reduce(reduce_func_builder<duplicate_keep_option::KEEP_LAST>{});
    case duplicate_keep_option::KEEP_NONE:
      return hash_reduce(reduce_func_builder<duplicate_keep_option::KEEP_NONE>{});
    default:  //  KEEP_ANY
      CUDF_FAIL("This function should not be called with KEEP_ANY");
  }
}

}  // namespace cudf::detail
