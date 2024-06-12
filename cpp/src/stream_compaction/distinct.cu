/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "distinct_helpers.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Invokes the given `func` with desired the row equality and probing method
 *
 * @tparam HasNested Flag indicating whether there are nested columns in the input
 * @tparam Hasher Type of device hash function
 * @tparam Func Type of the helper function doing `distinct` check
 *
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param has_nulls Flag indicating whether the input has nulls or not
 * @param equal Self table comparator
 * @param d_hash Device hash functor
 * @param func The input functor to invoke
 */
template <bool HasNested, typename Hasher, typename Func>
rmm::device_uvector<cudf::size_type> dispatch_hash_set(
  null_equality compare_nulls,
  nan_equality compare_nans,
  bool has_nulls,
  cudf::experimental::row::equality::self_comparator row_equal,
  Hasher const& d_hash,
  Func&& func)
{
  // Distinguish probing scheme CG sizes between nested and flat types for better performance
  auto const probing_scheme = [&]() {
    if constexpr (HasNested) {
      return cuco::linear_probing<4, Hasher>{d_hash};
    } else {
      return cuco::linear_probing<1, Hasher>{d_hash};
    }
  }();

  if (compare_nans == nan_equality::ALL_EQUAL) {
    auto const d_equal = row_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_nulls},
      compare_nulls,
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
    return func(d_equal, probing_scheme);
  } else {
    auto const d_equal = row_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_nulls},
      compare_nulls,
      cudf::experimental::row::equality::physical_equality_comparator{});
    return func(d_equal, probing_scheme);
  }
}
}  // namespace

template <typename Set>
rmm::device_uvector<size_type> process_keep_option(Set& set,
                                                   size_type set_size,
                                                   size_type num_rows,
                                                   duplicate_keep_option keep,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  auto output_indices = rmm::device_uvector<size_type>(set_size, stream, mr);

  // If we don't care about order, just gather indices of distinct keys taken from set.
  if (keep == duplicate_keep_option::KEEP_ANY) {
    set.retrieve_all(output_indices.begin(), stream.value());
    return output_indices;
  }

  auto reduction_results = rmm::device_uvector<size_type>(num_rows, stream, mr);
  thrust::uninitialized_fill(rmm::exec_policy(stream),
                             reduction_results.begin(),
                             reduction_results.end(),
                             reduction_init_value(keep));

  static auto constexpr cg_size = Set::cg_size;

  auto set_ref = set.ref(cuco::op::insert_and_find);

  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_rows * cg_size),
    [set_ref, keep, reduction_results = reduction_results.begin()] __device__(
      size_type const index) mutable {
      auto const idx = index / cg_size;
      auto const tile =
        cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());

      auto [out_ptr, inserted] = [&]() {
        if constexpr (cg_size == 1) {
          return set_ref.insert_and_find(idx);
        } else {
          return set_ref.insert_and_find(tile, idx);
        }
      }();

      if (tile.thread_rank() == 0) {
        auto ref =
          cuda::atomic_ref<size_type, cuda::thread_scope_device>{reduction_results[*out_ptr]};
        if (keep == duplicate_keep_option::KEEP_FIRST) {
          // Store the smallest index of all rows that are equal.
          ref.fetch_min(idx, cuda::memory_order_relaxed);
        }
        if (keep == duplicate_keep_option::KEEP_LAST) {
          // Store the greatest index of all rows that are equal.
          ref.fetch_max(idx, cuda::memory_order_relaxed);
        }
        if (keep == duplicate_keep_option::KEEP_NONE) {
          // Count the number of rows in each group of rows that are compared equal.
          ref.fetch_add(size_type{1}, cuda::memory_order_relaxed);
        }
      }
    });

  auto const map_end = [&] {
    if (keep == duplicate_keep_option::KEEP_NONE) {
      // Reduction results with `KEEP_NONE` are either group sizes of equal rows, or `0`.
      // Thus, we only output index of the rows in the groups having group size of `1`.
      return thrust::copy_if(rmm::exec_policy(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(num_rows),
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

rmm::device_uvector<size_type> distinct_indices(table_view const& input,
                                                duplicate_keep_option keep,
                                                null_equality nulls_equal,
                                                nan_equality nans_equal,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.num_rows();

  if (num_rows == 0 or input.num_columns() == 0) {
    return rmm::device_uvector<size_type>(0, stream, mr);
  }

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls          = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  auto const row_hash = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const d_hash   = row_hash.device_hasher(has_nulls);

  auto const row_equal = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const helper_func = [&](auto const& d_equal, auto const& probing_scheme) {
    auto set        = cuco::static_set{num_rows,
                                0.5,  // desired load factor
                                cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                                d_equal,
                                probing_scheme,
                                       {},
                                       {},
                                cudf::detail::cuco_allocator{stream},
                                stream.value()};
    auto const iter = thrust::counting_iterator<cudf::size_type>{0};
    auto const size = set.insert(iter, iter + num_rows, stream.value());
    return process_keep_option(set, size, num_rows, keep, stream, mr);
  };

  if (cudf::detail::has_nested_columns(input)) {
    return dispatch_hash_set<true>(
      nulls_equal, nans_equal, has_nulls, row_equal, d_hash, helper_func);
  } else {
    return dispatch_hash_set<false>(
      nulls_equal, nans_equal, has_nulls, row_equal, d_hash, helper_func);
  }
}

std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                nan_equality nans_equal,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto const gather_map = detail::distinct_indices(input.select(keys),
                                                   keep,
                                                   nulls_equal,
                                                   nans_equal,
                                                   stream,
                                                   rmm::mr::get_current_device_resource());
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
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(
    input, keys, keep, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> distinct_indices(table_view const& input,
                                         duplicate_keep_option keep,
                                         null_equality nulls_equal,
                                         nan_equality nans_equal,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto indices = detail::distinct_indices(input, keep, nulls_equal, nans_equal, stream, mr);
  return std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);
}

}  // namespace cudf
