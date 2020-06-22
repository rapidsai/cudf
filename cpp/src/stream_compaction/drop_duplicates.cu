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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <algorithm>
#include <cmath>

namespace cudf {
namespace detail {
/*
 * unique_copy copies elements from the range [first, last) to a range beginning
 * with output, except that in a consecutive group of duplicate elements only
 * depending on last argument keep, only the first one is copied, or the last
 * one is copied or neither is copied. The return value is the end of the range
 * to which the elements are copied.
 */
template <typename Exec, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(Exec&& exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate comp,
                           const duplicate_keep_option keep)
{
  size_type last_index = thrust::distance(first, last) - 1;
  if (keep == duplicate_keep_option::KEEP_NONE) {
    return thrust::copy_if(exec,
                           first,
                           last,
                           thrust::counting_iterator<size_type>(0),
                           output,
                           [first, comp, last_index] __device__(size_type i) {
                             return (i == 0 || !comp(first[i], first[i - 1])) &&
                                    (i == last_index || !comp(first[i], first[i + 1]));
                           });
  } else {
    size_type offset = 1;
    if (keep == duplicate_keep_option::KEEP_FIRST) {
      last_index = 0;
      offset     = -1;
    }
    return thrust::copy_if(exec,
                           first,
                           last,
                           thrust::counting_iterator<size_type>(0),
                           output,
                           [first, comp, last_index, offset] __device__(size_type i) {
                             return (i == last_index || !comp(first[i], first[i + offset]));
                           });
  }
}

/**
 * @brief Create a column_view of index values which represent the row values
 * without duplicates as per @p `keep`
 *
 * Given a `keys` table_view, each row index is copied to output `unique_indices`, if the
 * corresponding row of `keys` table_view is unique, where the definition of unique depends on the
 * value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: only unique rows are kept
 *
 * @param[in] keys            table_view to identify duplicate rows
 * @param[out] unique_indices Column to store the index with unique rows
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_equal     flag to denote nulls are equal if null_equality::EQUAL,
 *                            nulls are not equal if null_equality::UNEQUAL
 * @param[in] stream          CUDA stream used for device memory operations and kernel launches.
 *
 * @return column_view column_view of unique row index as per specified `keep`, this is actually
 * slice of `unique_indices`.
 */
column_view get_unique_ordered_indices(cudf::table_view const& keys,
                                       cudf::mutable_column_view& unique_indices,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       cudaStream_t stream = 0)
{
  // sort only indices
  auto sorted_indices = sorted_order(
    keys, std::vector<order>{}, std::vector<null_order>{}, rmm::mr::get_default_resource(), stream);

  // extract unique indices
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if (cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
                                  sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep);

    return cudf::detail::slice(
      column_view(unique_indices),
      0,
      thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  } else {
    auto comp = row_equality_comparator<false>(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    auto result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
                                  sorted_indices->view().begin<cudf::size_type>(),
                                  sorted_indices->view().end<cudf::size_type>(),
                                  unique_indices.begin<cudf::size_type>(),
                                  comp,
                                  keep);

    return cudf::detail::slice(
      column_view(unique_indices),
      0,
      thrust::distance(unique_indices.begin<cudf::size_type>(), result_end));
  }
}

cudf::size_type distinct_count(table_view const& keys,
                               null_equality nulls_equal,
                               cudaStream_t stream)
{
  // sort only indices
  auto sorted_indices = sorted_order(
    keys, std::vector<order>{}, std::vector<null_order>{}, rmm::mr::get_default_resource(), stream);

  // count unique elements
  auto sorted_row_index   = sorted_indices->view().data<cudf::size_type>();
  auto device_input_table = cudf::table_device_view::create(keys, stream);

  if (cudf::has_nulls(keys)) {
    row_equality_comparator<true> comp(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    return thrust::count_if(
      rmm::exec_policy(stream)->on(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
      [sorted_row_index, comp] __device__(cudf::size_type i) {
        return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i - 1]));
      });
  } else {
    row_equality_comparator<false> comp(
      *device_input_table, *device_input_table, nulls_equal == null_equality::EQUAL);
    return thrust::count_if(
      rmm::exec_policy(stream)->on(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
      [sorted_row_index, comp] __device__(cudf::size_type i) {
        return (i == 0 || not comp(sorted_row_index[i], sorted_row_index[i - 1]));
      });
  }
}

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream)
{
  if (0 == input.num_rows() || 0 == input.num_columns() || 0 == keys.size()) {
    return empty_like(input);
  }

  auto keys_view = input.select(keys);

  // The values will be filled into this column
  auto unique_indices = cudf::make_numeric_column(
    data_type{INT32}, keys_view.num_rows(), mask_state::UNALLOCATED, stream);
  auto mutable_unique_indices_view = unique_indices->mutable_view();
  // This is just slice of `unique_indices` but with different size as per the
  // keys_view has been processed in `get_unique_ordered_indices`
  auto unique_indices_view = detail::get_unique_ordered_indices(
    keys_view, mutable_unique_indices_view, keep, nulls_equal, stream);

  // run gather operation to establish new order
  return detail::gather(input,
                        unique_indices_view,
                        detail::out_of_bounds_policy::NULLIFY,
                        detail::negative_index_policy::NOT_ALLOWED,
                        mr,
                        stream);
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
  bool operator()(column_view const& input, cudaStream_t stream)
  {
    auto input_device_view = cudf::column_device_view::create(input, stream);
    auto device_view       = *input_device_view;
    auto count             = thrust::count_if(rmm::exec_policy(stream)->on(stream),
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
  bool operator()(column_view const& input, cudaStream_t stream)
  {
    return false;
  }
};

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               cudaStream_t stream)
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

std::unique_ptr<table> drop_duplicates(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option const keep,
                                       null_equality nulls_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::drop_duplicates(input, keys, keep, nulls_equal, mr);
}

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
