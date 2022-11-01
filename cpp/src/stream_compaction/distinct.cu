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

#include "distinct_reduce.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {

rmm::device_uvector<size_type> get_distinct_indices(table_view const& input,
                                                    duplicate_keep_option keep,
                                                    null_equality nulls_equal,
                                                    nan_equality nans_equal,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0) {
    return rmm::device_uvector<size_type>(0, stream, mr);
  }

  auto map = hash_map_type{compute_hash_table_size(input.num_rows()),
                           cuco::sentinel::empty_key{COMPACTION_EMPTY_KEY_SENTINEL},
                           cuco::sentinel::empty_value{COMPACTION_EMPTY_VALUE_SENTINEL},
                           detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                           stream.value()};

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = experimental::compaction_hash(row_hasher.device_hasher(has_nulls));

  auto const row_comp = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const pair_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

  auto const insert_keys = [&](auto const value_comp) {
    auto const key_equal = row_comp.equal_to(has_nulls, nulls_equal, value_comp);
    map.insert(pair_iter, pair_iter + input.num_rows(), key_hasher, key_equal, stream.value());
  };

  if (nans_equal == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    insert_keys(nan_equal_comparator{});
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    insert_keys(nan_unequal_comparator{});
  }

  auto output_indices = rmm::device_uvector<size_type>(map.get_size(), stream, mr);

  // If we don't care about order, just gather indices of distinct keys taken from map.
  if (keep == duplicate_keep_option::KEEP_ANY) {
    map.retrieve_all(output_indices.begin(), thrust::make_discard_iterator(), stream.value());
    return output_indices;
  }

  // For other keep options, reduce by row on rows that compare equal.
  auto const reduction_results = hash_reduce_by_row(map,
                                                    std::move(preprocessed_input),
                                                    input.num_rows(),
                                                    has_nulls,
                                                    keep,
                                                    nulls_equal,
                                                    nans_equal,
                                                    stream);

  // Extract the desired output indices from reduction results.
  auto const map_end = [&] {
    if (keep == duplicate_keep_option::KEEP_NONE) {
      // Reduction results with `KEEP_NONE` are either group sizes of equal rows, or `0`.
      // Thus, we only output index of the rows in the groups having group size of `1`.
      return thrust::copy_if(rmm::exec_policy(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(input.num_rows()),
                             output_indices.begin(),
                             [reduction_results = reduction_results.begin()] __device__(
                               auto const idx) { return reduction_results[idx] == size_type{1}; });
    }

    // Reduction results with `KEEP_FIRST` and `KEEP_LAST` are row indices of the first/last row in
    // each group of equal rows (which are the desired output indices), or the value given by
    // `reduction_init_value()`.
    return thrust::copy_if(rmm::exec_policy(stream),
                           reduction_results.begin(),
                           reduction_results.end(),
                           output_indices.begin(),
                           [init_value = reduction_init_value(keep)] __device__(auto const idx) {
                             return idx != init_value;
                           });
  }();

  output_indices.resize(thrust::distance(output_indices.begin(), map_end), stream);
  return output_indices;
}

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                nan_equality nans_equal,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto const gather_map =
    get_distinct_indices(input.select(keys), keep, nulls_equal, nans_equal, stream);
  return detail::gather(input,
                        gather_map,
                        out_of_bounds_policy::DONT_CHECK,
                        negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                nan_equality nans_equal,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(
    input, keys, keep, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

}  // namespace cudf
