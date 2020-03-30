/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/types.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>

namespace cudf {
namespace experimental {
namespace detail {

/**
 * @brief Filters a table to remove null elements.
 *
 * Filters the rows of the `input` considering specified columns indicated in
 * `keys` for validity / null values.
 *
 * Given an input table_view, row `i` from the input columns is copied to
 * the output if the same row `i` of @p keys has at least @p keep_threshold
 * non-null fields.
 *
 * This operation is stable: the input order is preserved in the output.
 *
 * Any non-nullable column in the input is treated as all non-null.
 *
 * @example input   {col1: {1, 2,    3,    null},
 *                   col2: {4, 5,    null, null},
 *                   col3: {7, null, null, null}}
 *          keys = {0, 1, 2} // All columns
 *          keep_threshold = 2
 *
 *          output {col1: {1, 2}
 *                  col2: {4, 5}
 *                  col3: {7, null}}
 *
 * @note if @p input.num_rows() is zero, or @p keys is empty or has no nulls,
 * there is no error, and an empty `table` is returned
 *
 * @param[in] input The input `table_view` to filter.
 * @param[in] keys  vector of indices representing key columns from `input`
 * @param[in] keep_threshold The minimum number of non-null fields in a row
 *                           required to keep the row.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return unique_ptr<table> Table containing all rows of the `input` with at least @p keep_threshold non-null fields in @p keys.
 */
std::unique_ptr<experimental::table>
    drop_nulls(table_view const& input,
               std::vector<size_type> const& keys,
               cudf::size_type keep_threshold,
               rmm::mr::device_memory_resource *mr =
                   rmm::mr::get_default_resource(),
               cudaStream_t stream = 0);

/**
 * @brief Filters `input` using `boolean_mask` of boolean values as a mask.
 *
 * Given an input `table_view` and a mask `column_view`, an element `i` from
 * each column_view of the `input` is copied to the corresponding output column
 * if the corresponding element `i` in the mask is non-null and `true`.
 * This operation is stable: the input order is preserved.
 *
 * @note if @p input.num_rows() is zero, there is no error, and an empty table
 * is returned.
 *
 * @throws cudf::logic_error if The `input` size  and `boolean_mask` size mismatches.
 * @throws cudf::logic_error if `boolean_mask` is not `BOOL8` type.
 *
 * @param[in] input The input table_view to filter
 * @param[in] boolean_mask A nullable column_view of type BOOL8 used as
 * a mask to filter the `input`.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return unique_ptr<table> Table containing copy of all rows of @p input passing
 * the filter defined by @p boolean_mask.
 */
std::unique_ptr<experimental::table>
    apply_boolean_mask(table_view const& input,
                       column_view const& boolean_mask,
                       rmm::mr::device_memory_resource *mr =
                           rmm::mr::get_default_resource(),
                       cudaStream_t stream = 0);

/**
 * @brief Create a new table without duplicate rows
 *
 * Given an `input` table_view, each row is copied to output table if the corresponding
 * row of `keys` columns is unique, where the definition of unique depends on the value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate rows is copied
 * - KEEP_LAST: only the last of a sequence of duplicate rows is copied
 * - KEEP_NONE: only unique rows are kept
 *
 * @throws cudf::logic_error if The `input` row size mismatches with `keys`.
 *
 * @param[in] input           input table_view to copy only unique rows
 * @param[in] keys            vector of indices representing key columns from `input`
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_are_equal flag to denote nulls are equal if true,
 * nulls are not equal if false
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @return unique_ptr<table> Table with unique rows as per specified `keep`.
 */
std::unique_ptr<experimental::table>
    drop_duplicates(table_view const& input,
                    std::vector<size_type> const& keys,
                    duplicate_keep_option const& keep,
                    bool const& nulls_are_equal=true,
                    rmm::mr::device_memory_resource *mr =
                        rmm::mr::get_default_resource(),
                    cudaStream_t stream = 0);

/**
 * @brief Count the unique elements in the column_view
 *
 * Given an input column_view, number of unique elements in this column_view is returned
 *
 * If both `ignore_nulls` and `nan_as_null` are true, both `NaN` and `null`
 * values are ignored.
 * If `ignor_nulls` is true and `nan_as_null` is false, only `null` is
 * ignored, `NaN` is considered in unique count.
 *
 * @param[in] input         The column_view whose unique elements will be counted.
 * @param[in] ignore_nulls  flag to ignore `null` in unique count if true
 * @param[in] nan_as_null   flag to consider `NaN==null` if true.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @return number of unique elements
 */

cudf::size_type unique_count(column_view const& input,
                             bool const& ignore_nulls,
                             bool const& nan_as_null,
                             cudaStream_t stream = 0);

/**---------------------------------------------------------------------------*
 * @brief A structure to be used for checking `NAN` at an index in a 
 * `column_device_view`
 *
 * @tparam T The type of `column_device_view`
 *---------------------------------------------------------------------------**/
template <typename T>
struct check_for_nan
{
  /**---------------------------------------------------------------------------*
   * @brief Construct a strcuture
   *
   * @param[in] input The `column_device_view`
   *---------------------------------------------------------------------------**/
  check_for_nan(cudf::column_device_view input) :_input{input}{}


  /**---------------------------------------------------------------------------*
   * @brief Operator to be called to check for `NAN` at `index` in `_input`
   *
   * @param[in] index The index at which the `NAN` needs to be checked in `input`
   *
   * @returns bool true if value at `index` is `NAN` and not null, else false
   *---------------------------------------------------------------------------**/
  __device__
  bool operator()(size_type index)
  {
    return std::isnan(_input.data<T>()[index]) and _input.is_valid(index);
  }

protected:
  cudf::column_device_view _input;
};

/**---------------------------------------------------------------------------*
 * @brief A structure to be used along with type_dispatcher to check if a
 * `column_view` has `NAN`.
 *---------------------------------------------------------------------------**/
struct has_nans{

  /**---------------------------------------------------------------------------*
   * @brief Checks if `input` has `NAN`
   *
   * @note This will be applicable only for floating point type columns.
   *
   * @param[in] input The `column_view` which will be checked for `NAN`
   * @param[in] stream Optional CUDA stream on which to execute kernels
   *
   * @returns bool true if `input` has `NAN` else false
   *---------------------------------------------------------------------------**/
  template <typename T,
         std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  bool operator()(column_view const& input, cudaStream_t stream){
      auto input_device_view = cudf::column_device_view::create(input, stream);
      auto device_view = *input_device_view;
      auto count = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                    thrust::counting_iterator<cudf::size_type>(0),
                                    thrust::counting_iterator<cudf::size_type>(input.size()),
                                    check_for_nan<T>(device_view));
      return count > 0;
  }

  /**---------------------------------------------------------------------------*
   * @brief Checks if `input` has `NAN`
   *
   * @note This will be applicable only for non-floating point type columns. And
   * non-floating point columns can never have `NAN`, so it will always return 
   * false
   *
   * @param[in] input The `column_view` which will be checked for `NAN`
   * @param[in] stream Optional CUDA stream on which to execute kernels
   *
   * @returns bool Always false as non-floating point columns can't have `NAN`
   *---------------------------------------------------------------------------**/
  template <typename T,
          std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  bool operator()(column_view const& input, cudaStream_t stream){
      return false;
  }
};

} // namespace detail
} // namespace experimental
} // namespace cudf
