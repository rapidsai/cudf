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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if.cuh>
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

namespace {
template <duplicate_keep_option keep, typename MapDeviceView, typename Hash, typename KeyEqual>
struct reduce_index_fn {
  __device__ void operator()(size_type const key) const
  {
    // iter should always be valid, because all keys have been inserted.
    auto const iter = d_map.find(key, d_hash, d_eqcomp);

    // Here idx is the index of the unique elements that has been inserted into the map.
    // As such, `find` calling for all duplicate keys will return the same idx value.
    auto const idx = iter->second.load(cuda::std::memory_order_relaxed);

    if constexpr (keep == duplicate_keep_option::KEEP_FIRST) {
      // Store the smallest index of all keys that are equal.
      atomicMin(&d_output[idx], key);
    } else if constexpr (keep == duplicate_keep_option::KEEP_LAST) {
      // Store the greatest index of all keys that are equal.
      atomicMax(&d_output[idx], key);
    } else {
      // Count the number of duplicates for key.
      atomicAdd(&d_output[idx], size_type{1});
    }
  }

  size_type* const d_output;
  MapDeviceView const d_map;
  Hash const d_hash;
  KeyEqual const d_eqcomp;
};

}  // namespace

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
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

  // If we don't care about order, just gather all rows having distinct keys taken from key_map.
  if (keep == duplicate_keep_option::KEEP_ANY) {
    auto const output_size = key_map.get_size();
    auto distinct_indices  = rmm::device_uvector<size_type>(output_size, stream);

    key_map.retrieve_all(distinct_indices.begin(), thrust::make_discard_iterator(), stream.value());
    return detail::gather(input,
                          distinct_indices.begin(),
                          distinct_indices.end(),
                          out_of_bounds_policy::DONT_CHECK,
                          stream,
                          mr);
  }

  auto const key_size  = keys_view.num_rows();
  auto reduced_indices = rmm::device_uvector<size_type>(key_size, stream);

  auto const init_value = [keep] {
    if (keep == duplicate_keep_option::KEEP_FIRST) {
      return std::numeric_limits<size_type>::max();
    } else if (keep == duplicate_keep_option::KEEP_LAST) {
      return std::numeric_limits<size_type>::min();
    }
    return size_type{0};  // keep == KEEP_NONE
  }();

  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             reduced_indices.begin(),
                             reduced_indices.begin() + key_size,
                             init_value);

  auto const do_reduce = [key_size, stream](auto const& fn) {
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::counting_iterator<size_type>(0),
                     thrust::counting_iterator<size_type>(key_size),
                     fn);
  };

  auto const d_map = key_map.get_device_view();
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST:
      do_reduce(
        reduce_index_fn<duplicate_keep_option::KEEP_FIRST,
                        decltype(d_map),
                        decltype(hash_key),
                        decltype(key_equal)>{reduced_indices.begin(), d_map, hash_key, key_equal});
      break;
    case duplicate_keep_option::KEEP_LAST:
      do_reduce(
        reduce_index_fn<duplicate_keep_option::KEEP_LAST,
                        decltype(d_map),
                        decltype(hash_key),
                        decltype(key_equal)>{reduced_indices.begin(), d_map, hash_key, key_equal});
      break;
    case duplicate_keep_option::KEEP_NONE:
      do_reduce(
        reduce_index_fn<duplicate_keep_option::KEEP_NONE,
                        decltype(d_map),
                        decltype(hash_key),
                        decltype(key_equal)>{reduced_indices.begin(), d_map, hash_key, key_equal});
      break;
    default:;  // KEEP_ANY has already been handled
  }

  // Filter out the invalid indices, which are indices of the duplicate keys
  // (the first duplicate key already has valid index being written in the previous step).
  if (keep == duplicate_keep_option::KEEP_FIRST || keep == duplicate_keep_option::KEEP_LAST) {
    return cudf::detail::copy_if(
      table_view{{input}},
      [init_value, reduced_indices = reduced_indices.begin()] __device__(auto const idx) {
        return reduced_indices[idx] != init_value;
      },
      stream,
      mr);
  }

  return cudf::detail::copy_if(
    table_view{{input}},
    [reduced_indices = reduced_indices.begin()] __device__(auto const idx) {
      return reduced_indices[idx] == size_type{1};
    },
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, keys, keep, nulls_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
