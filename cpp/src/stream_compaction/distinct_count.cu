/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

namespace cudf {
namespace detail {

cudf::size_type distinct_count(table_view const& keys,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream)
{
  // sort only indices
  auto sorted_indices = sorted_order(keys,
                                     std::vector<order>{},
                                     std::vector<null_order>{},
                                     stream,
                                     rmm::mr::get_current_device_resource());

  // count unique elements
  auto sorted_row_index   = sorted_indices->view().data<cudf::size_type>();
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if (cudf::has_nulls(keys)) {
    row_equality_comparator<true> comp(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    return thrust::count_if(
      rmm::exec_policy(stream)->on(stream.value()),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
      [sorted_row_index, comp] __device__(cudf::size_type i) {
        return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i - 1]));
      });
  } else {
    row_equality_comparator<false> comp(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    return thrust::count_if(
      rmm::exec_policy(stream)->on(stream.value()),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
      [sorted_row_index, comp] __device__(cudf::size_type i) {
        return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i - 1]));
      });
  }
}

/**
 * @brief Functor to check for `NAN` at an index in a `column_device_view`.
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
   * @brief Operator to be called to check for `NAN` at `index` in `_input`
   *
   * @param[in] index The index at which the `NAN` needs to be checked in `input`
   *
   * @returns bool true if value at `index` is `NAN` and not null, else false
   */
  __device__ bool operator()(size_type index)
  {
    return std::isnan(_input.data<T>()[index]) and _input.is_valid(index);
  }

 protected:
  cudf::column_device_view _input;
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a
 * `column_view` has `NAN`.
 */
struct has_nans {
  /**
   * @brief Checks if `input` has `NAN`
   *
   * @note This will be applicable only for floating point type columns.
   *
   * @param[in] input The `column_view` which will be checked for `NAN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool true if `input` has `NAN` else false
   */
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream)
  {
    auto input_device_view = cudf::column_device_view::create(input, stream);
    auto device_view       = *input_device_view;
    auto count             = thrust::count_if(rmm::exec_policy(stream)->on(stream.value()),
                                  thrust::counting_iterator<cudf::size_type>(0),
                                  thrust::counting_iterator<cudf::size_type>(input.size()),
                                  check_for_nan<T>(device_view));
    return count > 0;
  }

  /**
   * @brief Checks if `input` has `NAN`
   *
   * @note This will be applicable only for non-floating point type columns. And
   * non-floating point columns can never have `NAN`, so it will always return
   * false
   *
   * @param[in] input The `column_view` which will be checked for `NAN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool Always false as non-floating point columns can't have `NAN`
   */
  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream)
  {
    return false;
  }
};

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream)
{
  if (0 == input.size() || input.null_count() == input.size()) { return 0; }

  cudf::size_type nrows = input.size();

  bool has_nan = false;
  // Check for Nans
  // Checking for nulls in input and flag nan_handling, as the count will
  // only get affected if these two conditions are true. NAN will only be
  // be an extra if nan_handling was NAN_IS_NULL and input also had null, which
  // will increase the count by 1.
  if (input.has_nulls() and nan_handling == nan_policy::NAN_IS_NULL) {
    has_nan = cudf::type_dispatcher(input.type(), has_nans{}, input, stream);
  }

  auto count = detail::distinct_count(table_view{{input}}, null_equality::EQUAL, stream);

  // if nan is considered null and there are already null values
  if (nan_handling == nan_policy::NAN_IS_NULL and has_nan and input.has_nulls()) --count;

  if (null_handling == null_policy::EXCLUDE and input.has_nulls())
    return --count;
  else
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
