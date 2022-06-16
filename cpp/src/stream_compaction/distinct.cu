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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/uninitialized_fill.h>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief A functor to perform reduction on row indices.
 *
 * A reduction operator will be performed on each group of rows that are compared equal.
 */
template <typename MapDeviceView, typename KeyHasher, typename KeyComparator>
struct reduce_op_gen {
  MapDeviceView const d_map;
  KeyHasher const d_hash;
  KeyComparator const d_eqcomp;
  size_type* const d_output;

  reduce_op_gen(MapDeviceView const& d_map,
                KeyHasher const& d_hash,
                KeyComparator const& d_eqcomp,
                size_type* const d_output)
    : d_map{d_map}, d_hash{d_hash}, d_eqcomp{d_eqcomp}, d_output{d_output}
  {
  }

  template <duplicate_keep_option keep>
  auto reduce_op() const
  {
    return reduce_index_fn<keep>{*this};
  }

  /**
   * @brief The functor used on device for row index reduction.
   *
   * This inner functor has only one template argument. That reduces the amount of template
   * parameters required upon constructing this functor, which happens multiple times. Other
   * template arguments belong to its parent, which needs to be constructed just once.
   */
  template <duplicate_keep_option keep>
  struct reduce_index_fn {
    reduce_op_gen const parent;

    __device__ void operator()(size_type const idx) const
    {
      if constexpr (keep == duplicate_keep_option::KEEP_FIRST) {
        // Store the smallest index of all rows that are equal.
        atomicMin(&parent.get_output(idx), idx);
      } else if constexpr (keep == duplicate_keep_option::KEEP_LAST) {
        // Store the greatest index of all rows that are equal.
        atomicMax(&parent.get_output(idx), idx);
      } else {
        // Count the number of rows that are equal to the row having index inserted.
        atomicAdd(&parent.get_output(idx), size_type{1});
      }
    }
  };

 private:
  __device__ size_type& get_output(size_type const idx) const
  {
    // Here we don't check `iter` validity for performance reason, assuming that it is always
    // valid because all input `idx` values have been fed into `map.insert()` before.
    auto const iter = d_map.find(idx, d_hash, d_eqcomp);

    // Only one index value of the duplicate rows could be inserted into the map.
    // As such, looking up for all indices of duplicate rows always returns the same value.
    auto const inserted_idx = iter->second.load(cuda::std::memory_order_relaxed);

    // All duplicate rows will have concurrent access to the same output slot.
    return d_output[inserted_idx];
  }
};

}  // namespace

rmm::device_uvector<size_type> get_distinct_indices(table_view const& input,
                                                    std::vector<size_type> const& keys,
                                                    duplicate_keep_option keep,
                                                    null_equality nulls_equal,
                                                    nan_equality,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return rmm::device_uvector<size_type>(0, stream, mr);
  }

  auto const keys_tview = input.select(keys);
  auto const preprocessed_keys =
    cudf::experimental::row::hash::preprocessed_table::create(keys_tview, stream);
  auto const has_null  = nullate::DYNAMIC{cudf::has_nested_nulls(keys_tview)};
  auto const keys_size = keys_tview.num_rows();

  auto key_map = hash_map_type{compute_hash_table_size(keys_size),
                               cuco::sentinel::empty_key{COMPACTION_EMPTY_KEY_SENTINEL},
                               cuco::sentinel::empty_value{COMPACTION_EMPTY_VALUE_SENTINEL},
                               detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                               stream.value()};

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_keys);
  auto const key_hasher = experimental::compaction_hash(row_hasher.device_hasher(has_null));

  auto const row_comp  = cudf::experimental::row::equality::self_comparator(preprocessed_keys);
  auto const key_equal = row_comp.equal_to(has_null, nulls_equal);

  auto const kv_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });
  key_map.insert(kv_iter, kv_iter + keys_size, key_hasher, key_equal, stream.value());

  // The output distinct map.
  auto output_indices = rmm::device_uvector<size_type>(key_map.get_size(), stream, mr);

  // If we don't care about order, just gather indices of distinct keys taken from key_map.
  if (keep == duplicate_keep_option::KEEP_ANY) {
    key_map.retrieve_all(output_indices.begin(), thrust::make_discard_iterator(), stream.value());
    return output_indices;
  }

  // Perform reduction on each group of rows compared equal and the results are store
  // into this array. The reduction operator is:
  // - If KEEP_FIRST: min.
  // - If KEEP_LAST: max.
  // - If KEEP_NONE: count number of appearances.
  auto reduction_results = rmm::device_uvector<size_type>(keys_tview.num_rows(), stream);

  auto const init_value = [keep] {
    if (keep == duplicate_keep_option::KEEP_FIRST) {
      return std::numeric_limits<size_type>::max();
    } else if (keep == duplicate_keep_option::KEEP_LAST) {
      return std::numeric_limits<size_type>::min();
    }
    return size_type{0};  // keep == KEEP_NONE
  }();
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), reduction_results.begin(), reduction_results.end(), init_value);

  auto const d_map  = key_map.get_device_view();
  auto const fn_gen = reduce_op_gen{d_map, key_hasher, key_equal, reduction_results.begin()};

  auto const do_reduce = [keys_size, stream](auto const& fn) {
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(keys_size),
                     fn);
  };
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST:
      do_reduce(fn_gen.reduce_op<duplicate_keep_option::KEEP_FIRST>());
      break;
    case duplicate_keep_option::KEEP_LAST:
      do_reduce(fn_gen.reduce_op<duplicate_keep_option::KEEP_LAST>());
      break;
    case duplicate_keep_option::KEEP_NONE:
      do_reduce(fn_gen.reduce_op<duplicate_keep_option::KEEP_NONE>());
      break;
    default:;  // KEEP_ANY has already been handled above
  }

  // Filter out indices of the undesired duplicate keys.
  auto const map_end =
    keep == duplicate_keep_option::KEEP_NONE
      ? thrust::copy_if(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(keys_size),
                        output_indices.begin(),
                        [reduction_results = reduction_results.begin()] __device__(auto const idx) {
                          // Only output index of the rows that appeared once during reduction.
                          // Indices of duplicate rows will be either >1 or `0`.
                          return reduction_results[idx] == size_type{1};
                        })
      : thrust::copy_if(rmm::exec_policy(stream),
                        reduction_results.begin(),
                        reduction_results.end(),
                        output_indices.begin(),
                        [init_value] __device__(auto const idx) { return idx != init_value; });

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

  auto const gather_map = get_distinct_indices(input, keys, keep, nulls_equal, nans_equal, stream);
  return detail::gather(
    input, gather_map.begin(), gather_map.end(), out_of_bounds_policy::DONT_CHECK, stream, mr);
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
  return detail::distinct(input, keys, keep, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
