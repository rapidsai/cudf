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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

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
rmm::device_uvector<cudf::size_type> dispatch_row_equal(
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
}  // namespace

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

  auto const helper_func = [&](auto const& d_equal) {
    using RowEqual = std::decay_t<decltype(d_equal)>;
    auto set       = distinct_set_t<RowEqual>{
      num_rows,
      0.5,  // desired load factor
      cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
      d_equal,
      {row_hash.device_hasher(has_nulls)},
      {},
      {},
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
      stream.value()};
    return detail::reduce_by_row(set, num_rows, keep, stream, mr);
  };

  if (cudf::detail::has_nested_columns(input)) {
    return dispatch_row_equal<true>(nulls_equal, nans_equal, has_nulls, row_equal, helper_func);
  } else {
    return dispatch_row_equal<false>(nulls_equal, nans_equal, has_nulls, row_equal, helper_func);
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
                                                   cudf::get_current_device_resource_ref());
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
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, keys, keep, nulls_equal, nans_equal, stream, mr);
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
