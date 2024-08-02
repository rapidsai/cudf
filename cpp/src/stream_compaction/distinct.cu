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
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>
#include <thrust/iterator/discard_iterator.h>

#include <functional>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Invokes the given `func` with desired the row equality
 *
 * @tparam HasNested Flag indicating whether there are nested columns in the input
 * @tparam Func Type of the helper function doing `distinct` check
 *
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param has_nulls Flag indicating whether the input has nulls or not
 * @param row_equal Self table comparator
 * @param func The input functor to invoke
 */
template <bool HasNested, typename Func>
rmm::device_uvector<cudf::size_type> dipatch_row_equal(
  null_equality compare_nulls,
  nan_equality compare_nans,
  bool has_nulls,
  cudf::experimental::row::equality::self_comparator row_equal,
  Func&& func)
{
  if (compare_nans == nan_equality::ALL_EQUAL) {
    auto const d_equal = row_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_nulls},
      compare_nulls,
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
    return func(d_equal);
  } else {
    auto const d_equal = row_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_nulls},
      compare_nulls,
      cudf::experimental::row::equality::physical_equality_comparator{});
    return func(d_equal);
  }
}

struct plus_op {
  template <cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<size_type, Scope> ref, size_type const val)
  {
    ref.fetch_add(static_cast<size_type>(1), cuda::memory_order_relaxed);
  }
};

struct min_op {
  template <cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<size_type, Scope> ref, size_type const val)
  {
    ref.fetch_min(val, cuda::memory_order_relaxed);
  }
};

struct max_op {
  template <cuda::thread_scope Scope>
  __device__ void operator()(cuda::atomic_ref<size_type, Scope> ref, size_type const val)
  {
    ref.fetch_max(val, cuda::memory_order_relaxed);
  }
};

template <typename Map>
rmm::device_uvector<size_type> process_keep(Map& map,
                                            size_type num_rows,
                                            duplicate_keep_option keep,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  if ((keep == duplicate_keep_option::KEEP_FIRST) or (keep == duplicate_keep_option::KEEP_LAST)) {
    auto output_indices = rmm::device_uvector<size_type>(num_rows, stream, mr);

    auto pairs =
      thrust::make_transform_iterator(thrust::counting_iterator<size_type>(0),
                                      cuda::proclaim_return_type<cuco::pair<size_type, size_type>>(
                                        [] __device__(size_type const i) {
                                          return cuco::pair<size_type, size_type>{i, i};
                                        }));

    if (keep == duplicate_keep_option::KEEP_FIRST) {
      map.insert_or_apply_async(pairs, pairs + num_rows, min_op{}, stream.value());
    } else {
      map.insert_or_apply_async(pairs, pairs + num_rows, max_op{}, stream.value());
    }

    auto const [_, output_end] =
      map.retrieve_all(thrust::make_discard_iterator(), output_indices.begin(), stream.value());
    output_indices.resize(thrust::distance(output_indices.begin(), output_end), stream);
    return output_indices;
  }

  auto keys   = rmm::device_uvector<size_type>(num_rows, stream, mr);
  auto values = rmm::device_uvector<size_type>(num_rows, stream, mr);

  auto pairs = thrust::make_transform_iterator(
    thrust::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<size_type, size_type>>([] __device__(size_type const i) {
      return cuco::pair<size_type, size_type>{i, 1};
    }));

  map.insert_or_apply_async(pairs, pairs + num_rows, plus_op{}, stream.value());
  auto const [keys_end, _] = map.retrieve_all(keys.begin(), values.begin(), stream.value());

  auto num_distinct_keys = thrust::distance(keys.begin(), keys_end);
  keys.resize(num_distinct_keys, stream);
  values.resize(num_distinct_keys, stream);

  auto output_indices = rmm::device_uvector<size_type>(num_distinct_keys, stream, mr);

  auto const output_iter = cudf::detail::make_counting_transform_iterator(
    size_type(0),
    cuda::proclaim_return_type<size_type>(
      [keys = keys.begin(), values = values.begin()] __device__(auto const idx) {
        return values[idx] == size_type{1} ? keys[idx] : -1;
      }));

  auto const map_end = thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    output_iter,
    output_iter + num_distinct_keys,
    output_indices.begin(),
    cuda::proclaim_return_type<bool>([] __device__(auto const idx) { return idx != -1; }));

  output_indices.resize(thrust::distance(output_indices.begin(), map_end), stream);
  return output_indices;
}

}  // namespace

/**
 * @brief Return the reduction identity used to initialize results of `hash_reduce_by_row`.
 *
 * @param keep A value of `duplicate_keep_option` type, must not be `KEEP_ANY`.
 * @return The initial reduction value.
 */
auto constexpr reduction_init_value(duplicate_keep_option keep)
{
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST: return std::numeric_limits<size_type>::max();
    case duplicate_keep_option::KEEP_LAST: return std::numeric_limits<size_type>::min();
    case duplicate_keep_option::KEEP_NONE: return size_type{0};
    default: CUDF_UNREACHABLE("This function should not be called with KEEP_ANY");
  }
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

  auto const row_hash  = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const row_equal = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const probing_scheme = cuco::linear_probing<
    1,
    cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                     cudf::nullate::DYNAMIC>>{
    row_hash.device_hasher(has_nulls)};

  auto const helper_func = [&](auto const& d_equal) {
    // If we don't care about order, just gather indices of distinct keys taken from set.
    if (keep == duplicate_keep_option::KEEP_ANY) {
      auto set = cuco::static_set{num_rows,
                                  0.5,  // desired load factor
                                  cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                                  d_equal,
                                  probing_scheme,
                                  {},
                                  {},
                                  cudf::detail::cuco_allocator{stream},
                                  stream.value()};

      auto const iter = thrust::counting_iterator<cudf::size_type>{0};
      set.insert_async(iter, iter + num_rows, stream.value());
      auto output_indices   = rmm::device_uvector<size_type>(num_rows, stream, mr);
      auto const output_end = set.retrieve_all(output_indices.begin(), stream.value());
      output_indices.resize(thrust::distance(output_indices.begin(), output_end), stream);
      return output_indices;
    }

    auto map = cuco::static_map{num_rows,
                                0.5,  // desired load factor
                                cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                                cuco::empty_value{reduction_init_value(keep)},
                                d_equal,
                                probing_scheme,
                                {},
                                {},
                                cudf::detail::cuco_allocator{stream},
                                stream.value()};
    return process_keep(map, num_rows, keep, stream, mr);
  };

  if (cudf::detail::has_nested_columns(input)) {
    return dipatch_row_equal<true>(nulls_equal, nans_equal, has_nulls, row_equal, helper_func);
  } else {
    return dipatch_row_equal<false>(nulls_equal, nans_equal, has_nulls, row_equal, helper_func);
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
