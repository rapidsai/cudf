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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Functor to check for `NaN` at an index in a `column_device_view`.
 *
 * @tparam T The type of `column_device_view`
 */
template <typename T>
struct check_for_nan {
  /*
   * @brief Construct from a column_device_view.
   *
   * @param[in] input The `column_device_view`
   */
  check_for_nan(cudf::column_device_view input) : _input{input} {}

  /**
   * @brief Operator to be called to check for `NaN` at `index` in `_input`
   *
   * @param[in] index The index at which the `NaN` needs to be checked in `input`
   *
   * @returns bool true if value at `index` is `NaN` and not null, else false
   */
  __device__ bool operator()(size_type index) const noexcept
  {
    return std::isnan(_input.data<T>()[index]) and _input.is_valid(index);
  }

  cudf::column_device_view _input;
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a
 * `column_view` has `NaN`.
 */
struct has_nans {
  /**
   * @brief Checks if `input` has `NaN`
   *
   * @note This will be applicable only for floating point type columns.
   *
   * @param[in] input The `column_view` which will be checked for `NaN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool true if `input` has `NaN` else false
   */
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream)
  {
    auto input_device_view = cudf::column_device_view::create(input, stream);
    auto device_view       = *input_device_view;
    return thrust::any_of(rmm::exec_policy(stream),
                          thrust::counting_iterator<cudf::size_type>(0),
                          thrust::counting_iterator<cudf::size_type>(input.size()),
                          check_for_nan<T>(device_view));
  }

  /**
   * @brief Checks if `input` has `NaN`
   *
   * @note This will be applicable only for non-floating point type columns. And
   * non-floating point columns can never have `NaN`, so it will always return
   * false
   *
   * @param[in] input The `column_view` which will be checked for `NaN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool Always false as non-floating point columns can't have `NaN`
   */
  template <typename T, std::enable_if_t<not std::is_floating_point_v<T>>* = nullptr>
  bool operator()(column_view const&, rmm::cuda_stream_view)
  {
    return false;
  }
};

/**
 * @brief A functor to be used along with device type_dispatcher to check if
 * the row `index` of `column_device_view` is `NaN`.
 */
struct check_nan {
  // Check if it's `NaN` for floating point type columns
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
  __device__ inline bool operator()(column_device_view const& input, size_type index)
  {
    return std::isnan(input.data<T>()[index]);
  }
  // Non-floating point type columns can never have `NaN`, so it will always return false.
  template <typename T, std::enable_if_t<not std::is_floating_point_v<T>>* = nullptr>
  __device__ inline bool operator()(column_device_view const&, size_type)
  {
    return false;
  }
};
}  // namespace

cudf::size_type distinct_count(table_view const& keys,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream)
{
  auto table_ptr = cudf::table_device_view::create(keys, stream);
  row_equality_comparator comp(
    nullate::DYNAMIC{cudf::has_nulls(keys)}, *table_ptr, *table_ptr, nulls_equal);
  return thrust::count_if(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
    [comp] __device__(cudf::size_type i) { return (i == 0 or not comp(i, i - 1)); });
}

cudf::size_type unordered_distinct_count(table_view const& keys,
                                         null_equality nulls_equal,
                                         rmm::cuda_stream_view stream)
{
  auto table_ptr      = cudf::table_device_view::create(keys, stream);
  auto const num_rows = table_ptr->num_rows();
  auto const has_null = nullate::DYNAMIC{cudf::has_nulls(keys)};

  hash_map_type key_map{compute_hash_table_size(num_rows),
                        COMPACTION_EMPTY_KEY_SENTINEL,
                        COMPACTION_EMPTY_VALUE_SENTINEL,
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value()};

  compaction_hash hash_key{has_null, *table_ptr};
  row_equality_comparator row_equal(has_null, *table_ptr, *table_ptr, nulls_equal);
  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(size_type i) { return cuco::make_pair(i, i); });

  // when nulls are equal, insert non-null rows only to improve efficiency
  if (nulls_equal == null_equality::EQUAL and has_null) {
    thrust::counting_iterator<size_type> stencil(0);
    auto const [row_bitmask, null_count] = cudf::detail::bitmask_or(keys, stream);
    row_validity pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    key_map.insert_if(iter, iter + num_rows, stencil, pred, hash_key, row_equal, stream.value());
    return key_map.get_size() + static_cast<std::size_t>((null_count > 0) ? 1 : 0);
  }
  // otherwise, insert all
  key_map.insert(iter, iter + num_rows, hash_key, row_equal, stream.value());
  return key_map.get_size();
}

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream)
{
  auto const num_rows = input.size();

  if (num_rows == 0 or num_rows == input.null_count()) { return 0; }

  auto const count_nulls      = null_handling == null_policy::INCLUDE;
  auto const nan_is_null      = nan_handling == nan_policy::NAN_IS_NULL;
  auto const should_check_nan = cudf::is_floating_point(input.type());
  auto input_device_view      = cudf::column_device_view::create(input, stream);
  auto device_view            = *input_device_view;
  auto input_table_view       = table_view{{input}};
  auto table_ptr              = cudf::table_device_view::create(input_table_view, stream);
  row_equality_comparator comp(nullate::DYNAMIC{cudf::has_nulls(input_table_view)},
                               *table_ptr,
                               *table_ptr,
                               null_equality::EQUAL);

  return thrust::count_if(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(num_rows),
    [count_nulls, nan_is_null, should_check_nan, device_view, comp] __device__(cudf::size_type i) {
      auto const is_null = device_view.is_null(i);
      auto const is_nan  = nan_is_null and should_check_nan and
                          cudf::type_dispatcher(device_view.type(), check_nan{}, device_view, i);
      if (not count_nulls and (is_null or (nan_is_null and is_nan))) { return false; }
      if (i == 0) { return true; }
      if (count_nulls and nan_is_null and (is_nan or is_null)) {
        auto const prev_is_nan =
          should_check_nan and
          cudf::type_dispatcher(device_view.type(), check_nan{}, device_view, i - 1);
        return not(prev_is_nan or device_view.is_null(i - 1));
      }
      return not comp(i, i - 1);
    });
}

cudf::size_type unordered_distinct_count(column_view const& input,
                                         null_policy null_handling,
                                         nan_policy nan_handling,
                                         rmm::cuda_stream_view stream)
{
  if (0 == input.size() or input.null_count() == input.size()) { return 0; }

  // Check for NaNs
  // Checking for nulls in input and flag nan_handling, as the count will
  // only get affected if these two conditions are true. NaN will only be
  // double-counted as a null if nan_handling was NAN_IS_NULL and input also
  // had null values. If so, we decrement the count.
  auto const has_nan_as_null = (nan_handling == nan_policy::NAN_IS_NULL) and
                               cudf::type_dispatcher(input.type(), has_nans{}, input, stream);
  auto const has_null = input.has_nulls();

  auto count = detail::unordered_distinct_count(table_view{{input}}, null_equality::EQUAL, stream);

  // if nan is considered null and there are already null values
  if (null_handling == null_policy::EXCLUDE and has_null) { --count; }
  if (has_nan_as_null and (has_null or null_handling == null_policy::EXCLUDE)) { --count; }
  return count;
}
}  // namespace detail

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling)
{
  CUDF_FUNC_RANGE();
  return detail::distinct_count(input, null_handling, nan_handling);
}

cudf::size_type distinct_count(table_view const& input, null_equality nulls_equal)
{
  CUDF_FUNC_RANGE();
  return detail::distinct_count(input, nulls_equal);
}

cudf::size_type unordered_distinct_count(column_view const& input,
                                         null_policy null_handling,
                                         nan_policy nan_handling)
{
  CUDF_FUNC_RANGE();
  return detail::unordered_distinct_count(input, null_handling, nan_handling);
}

cudf::size_type unordered_distinct_count(table_view const& input, null_equality nulls_equal)
{
  CUDF_FUNC_RANGE();
  return detail::unordered_distinct_count(input, nulls_equal);
}

}  // namespace cudf
