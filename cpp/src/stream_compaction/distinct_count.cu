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
#include <thrust/iterator/counting_iterator.h>
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
}  // namespace

cudf::size_type distinct_count(table_view const& keys,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream)
{
  auto table_ptr      = cudf::table_device_view::create(keys, stream);
  auto const num_rows = table_ptr->num_rows();
  auto const has_null = nullate::DYNAMIC{cudf::has_nulls(keys)};

  hash_map_type key_map{compute_hash_table_size(num_rows),
                        cuco::sentinel::empty_key{COMPACTION_EMPTY_KEY_SENTINEL},
                        cuco::sentinel::empty_value{COMPACTION_EMPTY_VALUE_SENTINEL},
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
  if (0 == input.size() or input.null_count() == input.size()) { return 0; }

  auto count = detail::distinct_count(table_view{{input}}, null_equality::EQUAL, stream);

  // Check for nulls. If the null policy is EXCLUDE and null values were found,
  // we decrement the count.
  auto const has_null = input.has_nulls();
  if (null_handling == null_policy::EXCLUDE and has_null) { --count; }

  // Check for NaNs. There are two cases that can lead to decrementing the
  // count. The first case is when the input has no nulls, but has NaN values
  // handled as a null via NAN_IS_NULL and has a policy to EXCLUDE null values
  // from the count. The second case is when the input has null values and NaN
  // values handled as nulls via NAN_IS_NULL. Regardless of whether the null
  // policy is set to EXCLUDE, we decrement the count to avoid double-counting
  // null and NaN as distinct entities.
  auto const has_nan_as_null = (nan_handling == nan_policy::NAN_IS_NULL) and
                               cudf::type_dispatcher(input.type(), has_nans{}, input, stream);
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
}  // namespace cudf
