/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {

/**
 * @addtogroup column_copy
 * @{
 * @file
 * @brief Column APIs for gather, scatter, split, slice, etc.
 */

/**
 * @brief Policy to account for possible out-of-bounds indices
 *
 * `NULLIFY` means to nullify output values corresponding to out-of-bounds gather_map values.
 * `DONT_CHECK` means do not check whether the indices are out-of-bounds, for better performance.
 */

enum class out_of_bounds_policy : bool {
  NULLIFY,    /// Output values corresponding to out-of-bounds indices are null
  DONT_CHECK  /// No bounds checking is performed, better performance
};

/**
 * @brief Gathers the specified rows (including null values) of a set of columns.
 *
 * @ingroup copy_gather
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * For dictionary columns, the keys column component is copied and not trimmed
 * if the gather results in abandoned key elements.
 *
 * @throws cudf::logic_error if gather_map contains null values.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map View into a non-nullable column of integral indices that maps the
 * rows in the source columns to rows in the destination columns.
 * @param[in] bounds_policy Policy to apply to account for possible out-of-bounds indices
 * `DONT_CHECK` skips all bounds checking for gather map values. `NULLIFY` coerces rows that
 * corresponds to out-of-bounds indices in the gather map to be null elements. Callers should
 * use `DONT_CHECK` when they are certain that the gather_map contains only valid indices for
 * better performance. If `policy` is set to `DONT_CHECK` and there are out-of-bounds indices
 * in the gather map, the behavior is undefined. Defaults to `DONT_CHECK`.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 * @return std::unique_ptr<table> Result of the gather
 */
std::unique_ptr<table> gather(
  table_view const& source_table,
  column_view const& gather_map,
  out_of_bounds_policy bounds_policy  = out_of_bounds_policy::DONT_CHECK,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * @ingroup copy_scatter
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * A negative value `i` in the `scatter_map` is interpreted as `i+n`, where `n`
 * is the number of rows in the `target` table.
 *
 * @throws cudf::logic_error if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input columns containing values to be scattered into the
 * target columns
 * @param scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. The size must be equal
 * to or less than the number of elements in the source columns.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(
  table_view const& source,
  column_view const& scatter_map,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Scatters a row of scalar values into a copy of the target table
 * according to a scatter map.
 *
 * @ingroup copy_scatter
 *
 * Scatters values from the source row into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table is
 * replaced by the source row. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of elements in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws cudf::logic_error if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input scalars containing values to be scattered into the
 * target columns
 * @param indices A non-nullable column of integral indices that indicate
 * the rows in the target table to be replaced by source.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr Device memory resource used to allocate the returned table's device memory.
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(
  std::vector<std::reference_wrapper<const scalar>> const& source,
  column_view const& indices,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Indicates when to allocate a mask, based on an existing mask.
 */
enum class mask_allocation_policy {
  NEVER,   ///< Do not allocate a null mask, regardless of input
  RETAIN,  ///< Allocate a null mask if the input contains one
  ALWAYS   ///< Allocate a null mask, regardless of input
};

/**
 * @brief Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Immutable view of input column to emulate
 * @return std::unique_ptr<column> An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view const& input);

/**
 * @brief Creates an uninitialized new column of the same size and type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return A column with sufficient uninitialized capacity to hold the same
 * number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(
  column_view const& input,
  mask_allocation_policy mask_alloc   = mask_allocation_policy::RETAIN,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] size The desired number of elements that the new column should have capacity for
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 * @return A column with sufficient uninitialized capacity to hold the specified number of elements
 * as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(
  column_view const& input,
  size_type size,
  mask_allocation_policy mask_alloc   = mask_allocation_policy::RETAIN,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param[in] input_table Immutable view of input table to emulate
 * @return std::unique_ptr<table> A table of empty columns with the same types as the columns in
 * `input_table`
 */
std::unique_ptr<table> empty_like(table_view const& input_table);

/**
 * @brief Copies a range of elements in-place from one column to another.
 *
 * Overwrites the range of elements in @p target indicated by the indices
 * [@p target_begin, @p target_begin + N) with the elements from @p source
 * indicated by the indices [@p source_begin, @p source_end) (where N =
 * (@p source_end - @p source_begin)). Use the out-of-place copy function
 * returning std::unique_ptr<column> for uses cases requiring memory
 * reallocation. For example for strings columns and other variable-width types.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws cudf::logic_error if memory reallocation is required (e.g. for
 * variable width types).
 * @throws cudf::logic_error for invalid range (if
 * @p source_begin > @p source_end, @p source_begin < 0,
 * @p source_begin >= @p source.size(), @p source_end > @p source.size(),
 * @p target_begin < 0, target_begin >= @p target.size(), or
 * @p target_begin + (@p source_end - @p source_begin) > @p target.size()).
 * @throws cudf::logic_error if @p target and @p source have different types.
 * @throws cudf::logic_error if @p source has null values and @p target is not
 * nullable.
 *
 * @param source The column to copy from
 * @param target The preallocated column to copy into
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 */
void copy_range_in_place(column_view const& source,
                         mutable_column_view& target,
                         size_type source_begin,
                         size_type source_end,
                         size_type target_begin);

/**
 * @brief Copies a range of elements out-of-place from one column to another.
 *
 * Creates a new column as if an in-place copy was performed into @p target.
 * A copy of @p target is created first and then the elements indicated by the
 * indices [@p target_begin, @p target_begin + N) were copied from the elements
 * indicated by the indices [@p source_begin, @p source_end) of @p source
 * (where N = (@p source_end - @p source_begin)). Elements outside the range are
 * copied from @p target into the returned new column target.
 *
 * If @p source and @p target refer to the same elements and the ranges overlap,
 * the behavior is undefined.
 *
 * @throws cudf::logic_error for invalid range (if
 * @p source_begin > @p source_end, @p source_begin < 0,
 * @p source_begin >= @p source.size(), @p source_end > @p source.size(),
 * @p target_begin < 0, target_begin >= @p target.size(), or
 * @p target_begin + (@p source_end - @p source_begin) > @p target.size()).
 * @throws cudf::logic_error if @p target and @p source have different types.
 *
 * @param source The column to copy from inside the range.
 * @param target The column to copy from outside the range.
 * @param source_begin The starting index of the source range (inclusive)
 * @param source_end The index of the last element in the source range
 * (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return std::unique_ptr<column> The result target column
 */
std::unique_ptr<column> copy_range(
  column_view const& source,
  column_view const& target,
  size_type source_begin,
  size_type source_end,
  size_type target_begin,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a new column by shifting all values by an offset.
 *
 * @ingroup copy_shift
 *
 * Elements will be determined by `output[idx] = input[idx - offset]`.
 * Some elements in the output may be indeterminable from the input. For those
 * elements, the value will be determined by `fill_values`.
 *
 * @code{.pseudo}
 * Examples
 * -------------------------------------------------
 * input       = [0, 1, 2, 3, 4]
 * offset      = 3
 * fill_values = @
 * return      = [@, @, @, 0, 1]
 * -------------------------------------------------
 * input       = [5, 4, 3, 2, 1]
 * offset      = -2
 * fill_values = 7
 * return      = [3, 2, 1, 7, 7]
 * @endcode
 *
 * @note if the input is nullable, the output will be nullable.
 * @note if the fill value is null, the output will be nullable.
 *
 * @param input      Column to be shifted.
 * @param offset     The offset by which to shift the input.
 * @param fill_value Fill value for indeterminable outputs.
 * @param mr         Device memory resource used to allocate the returned result's device memory
 *
 * @throw cudf::logic_error if @p input dtype is not fixed-with.
 * @throw cudf::logic_error if @p fill_value dtype does not match @p input dtype.
 */
std::unique_ptr<column> shift(
  column_view const& input,
  size_type offset,
  scalar const& fill_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Slices a `column_view` into a set of `column_view`s according to a set of indices.
 *
 * @ingroup copy_slice
 *
 * The returned views of `input` are constructed from an even number indices where
 * the `i`th returned `column_view` views the elements in `input` indicated by the range
 * `[indices[2*i], indices[(2*i)+1])`.
 *
 * For all `i` it is expected `indices[i] <= input.size()`
 * For all `i%2==0`, it is expected that `indices[i] <= indices[i+1]`
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  {{12, 14}, {20, 22, 24, 26}, {14, 16}, {}}
 * @endcode
 *
 * @throws cudf::logic_error if `indices` size is not even.
 * @throws cudf::logic_error When the values in the pair are strictly decreasing.
 * @throws cudf::logic_error When any of the values in the pair don't belong to
 * the range [0, input.size()).
 *
 * @param input View of column to slice
 * @param indices A vector of indices used to take slices of `input`.
 * @return Vector of views of `input` indicated by the ranges in `indices`.
 */
std::vector<column_view> slice(column_view const& input, std::vector<size_type> const& indices);

/**
 * @brief Slices a `table_view` into a set of `table_view`s according to a set of indices.
 *
 * @ingroup copy_slice
 *
 * The returned views of `input` are constructed from an even number indices where
 * the `i`th returned `table_view` views the elements in `input` indicated by the range
 * `[indices[2*i], indices[(2*i)+1])`.
 *
 * For all `i` it is expected `indices[i] <= input.size()`
 * For all `i%2==0`, it is expected that `indices[i] <= indices[i+1]`
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * indices: {1, 3, 5, 9, 2, 4, 8, 8}
 * output:  [{{12, 14}, {20, 22, 24, 26}, {14, 16}, {}},
 *           {{52, 54}, {60, 22, 24, 26}, {14, 16}, {}}]
 * @endcode
 *
 * @throws cudf::logic_error if `indices` size is not even.
 * @throws cudf::logic_error When the values in the pair are strictly decreasing.
 * @throws cudf::logic_error When any of the values in the pair don't belong to
 * the range [0, input.size()).
 *
 * @param input View of table to slice
 * @param indices A vector of indices used to take slices of `input`.
 * @return Vector of views of `input` indicated by the ranges in `indices`.
 */
std::vector<table_view> slice(table_view const& input, std::vector<size_type> const& indices);

/**
 * @brief Splits a `column_view` into a set of `column_view`s according to a set of indices
 * derived from expected splits.
 *
 * @ingroup copy_split
 *
 * The returned view's of `input` are constructed from vector of splits, which indicates
 * where the split should occur. The `i`th returned `column_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * Example:
 * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
 * splits:  {2, 5, 9}
 * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
 * @endcode
 *
 * @throws cudf::logic_error if `splits` has end index > size of `input`.
 * @throws cudf::logic_error When the value in `splits` is not in the range [0, input.size()).
 * @throws cudf::logic_error When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of column to split
 * @param splits A vector of indices where the view will be split
 * @return The set of requested views of `input` indicated by the `splits`.
 */
std::vector<column_view> split(column_view const& input, std::vector<size_type> const& splits);

/**
 * @brief Splits a `table_view` into a set of `table_view`s according to a set of indices
 * derived from expected splits.
 *
 * @ingroup copy_split
 *
 * The returned views of `input` are constructed from vector of splits, which indicates
 * where the split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 * @throws cudf::logic_error if `splits` has end index > size of `input`.
 * @throws cudf::logic_error When the value in `splits` is not in the range [0, input.size()).
 * @throws cudf::logic_error When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits A vector of indices where the view will be split
 * @return The set of requested views of `input` indicated by the `splits`.
 */
std::vector<table_view> split(table_view const& input, std::vector<size_type> const& splits);

/**
 * @brief Column data in a serialized format
 *
 * @ingroup copy_split
 *
 * Contains data from an array of columns in two contiguous buffers: one on host, which contains
 * table metadata and one on device which contains the table data.
 */
struct packed_columns {
  /**
   * @brief Host-side metadata buffer used for reconstructing columns via unpack.
   *
   * @ingroup copy_split
   */
  struct metadata {
    metadata(std::vector<uint8_t>&& v) : data_(std::move(v)) {}
    uint8_t const* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

   private:
    std::vector<uint8_t> data_;
  };

  std::unique_ptr<metadata> metadata_;
  std::unique_ptr<rmm::device_buffer> gpu_data;
};

/**
 * @brief The result(s) of a `contiguous_split`
 *
 * @ingroup copy_split
 *
 * Each table_view resulting from a split operation performed by contiguous_split,
 * will be returned wrapped in a `packed_table`.  The table_view and internal
 * column_views in this struct are not owned by a top level cudf::table or cudf::column.
 * The backing memory and metadata is instead owned by the `data` field and is in one
 * contiguous block.
 *
 * The user is responsible for assuring that the `table` or any derived table_views do
 * not outlive the memory owned by `data`
 */
struct packed_table {
  cudf::table_view table;
  packed_columns data;
};

/**
 * @brief Performs a deep-copy split of a `table_view` into a set of `table_view`s into a single
 * contiguous block of memory.
 *
 * @ingroup copy_split
 *
 * The memory for the output views is allocated in a single contiguous `rmm::device_buffer` returned
 * in the `packed_table`. There is no top-level owning table.
 *
 * The returned views of `input` are constructed from a vector of indices, that indicate
 * where each split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory contained in the `all_data` field of the
 * returned packed_table.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 *
 * @throws cudf::logic_error if `splits` has end index > size of `input`.
 * @throws cudf::logic_error When the value in `splits` is not in the range [0, input.size()).
 * @throws cudf::logic_error When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits A vector of indices where the view will be split
 * @param[in] mr Device memory resource used to allocate the returned result's device memory
 * @return The set of requested views of `input` indicated by the `splits` and the viewed memory
 * buffer.
 */
std::vector<packed_table> contiguous_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Deep-copy a `table_view` into a serialized contiguous memory format
 *
 * The metadata from the `table_view` is copied into a host vector of bytes and the data from the
 * `table_view` is copied into a `device_buffer`. Pass the output of this function into
 * `cudf::unpack` to deserialize.
 *
 * @param input View of the table to pack
 * @param[in] mr Optional, The resource to use for all returned device allocations
 * @return packed_columns A struct containing the serialized metadata and data in contiguous host
 *         and device memory respectively
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Produce the metadata used for packing a table stored in a contiguous buffer.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes which can be used to
 * construct a `packed_columns` or `packed_table` structure.  The caller is responsible for
 * guaranteeing that that all of the columns in the table point into `contiguous_buffer`.
 *
 * @param input View of the table to pack
 * @param contgiuous_buffer A contiguous buffer of device memory which contains the data referenced
 * by the columns in `table`
 * @param buffer_size The size of `contiguous_buffer`.
 * @return Vector of bytes representing the metadata used to `unpack` a packed_columns struct.
 */
packed_columns::metadata pack_metadata(table_view const& table,
                                       uint8_t const* contiguous_buffer,
                                       size_t buffer_size);

/**
 * @brief Deserialize the result of `cudf::pack`
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `input`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param input The packed columns to unpack
 * @return The unpacked `table_view`
 */
table_view unpack(packed_columns const& input);

/**
 * @brief Deserialize the result of `cudf::pack`
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `gpu_data` using the metadata contained in the host
 * buffer `metadata`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param metadata The host-side metadata buffer resulting from the initial pack() call
 * @param gpu_data The device-side contiguous buffer storing the data that will be referenced by
 * the resulting `table_view`
 * @return The unpacked `table_view`
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data);

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs[i] : rhs[i]`
 *
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length
 * @throws cudf::logic_error if boolean mask is not of type bool
 * @throws cudf::logic_error if boolean mask is not of the same length as lhs and rhs
 * @param[in] lhs left-hand column_view
 * @param[in] rhs right-hand column_view
 * @param[in] boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs : rhs[i]`
 *
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if boolean mask is not of type bool
 * @throws cudf::logic_error if boolean mask is not of the same length as rhs
 * @param[in] lhs left-hand scalar
 * @param[in] rhs right-hand column_view
 * @param[in] boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  column_view const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs[i] : rhs`
 *
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if boolean mask is not of type bool
 * @throws cudf::logic_error if boolean mask is not of the same length as lhs
 * @param[in] lhs left-hand column_view
 * @param[in] rhs right-hand scalar
 * @param[in] boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. Null element represents false.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  column_view const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following
 * rule: `output[i] = (boolean_mask.valid(i) and boolean_mask[i]) ? lhs : rhs`
 *
 * @throws cudf::logic_error if boolean mask is not of type bool
 * @param[in] lhs left-hand scalar
 * @param[in] rhs right-hand scalar
 * @param[in] boolean_mask column of `type_id::BOOL8` representing "left (true) / right (false)"
 * boolean for each element. null element represents false.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else(
  scalar const& lhs,
  scalar const& rhs,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Scatters rows from the input table to rows of the output corresponding
 * to true values in a boolean mask.
 *
 * @ingroup copy_scatter
 *
 * The `i`th row of `input` will be written to the output table at the location
 * of the `i`th true value in `boolean_mask`. All other rows in the output will
 * equal the same row in `target`.
 *
 * `boolean_mask` should have number of `true`s <= number of rows in `input`.
 * If boolean mask is `true`, corresponding value in target is updated with
 * value from corresponding `input` column, else it is left untouched.
 *
 * @code{.pseudo}
 * Example:
 * input: {{1, 5, 6, 8, 9}}
 * boolean_mask: {true, false, false, false, true, true, false, true, true, false}
 * target:       {{   2,     2,     3,     4,    4,     7,    7,    7,    8,    10}}
 *
 * output:       {{   1,     2,     3,     4,    5,     6,    7,    8,    9,    10}}
 * @endcode
 *
 * @throw  cudf::logic_error if input.num_columns() != target.num_columns()
 * @throws cudf::logic_error if any `i`th input_column type != `i`th target_column type
 * @throws cudf::logic_error if boolean_mask.type() != bool
 * @throws cudf::logic_error if boolean_mask.size() != target.num_rows()
 * @throws cudf::logic_error if number of `true` in `boolean_mask` > input.num_rows()
 *
 * @param[in] input table_view (set of dense columns) to scatter
 * @param[in] target table_view to modify with scattered values from `input`
 * @param[in] boolean_mask column_view which acts as boolean mask.
 * @param[in] mr Device memory resource used to allocate device memory of the returned table.
 *
 * @returns Returns a table by scattering `input` into `target` as per `boolean_mask`.
 */
std::unique_ptr<table> boolean_mask_scatter(
  table_view const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Scatters scalar values to rows of the output corresponding
 * to true values in a boolean mask.
 *
 * @ingroup copy_scatter
 *
 * The `i`th scalar in `input` will be written to all columns of the output
 * table at the location of the `i`th true value in `boolean_mask`.
 * All other rows in the output will equal the same row in `target`.
 *
 * @code{.pseudo}
 * Example:
 * input: {11}
 * boolean_mask: {true, false, false, false, true, true, false, true, true, false}
 * target:      {{   2,     2,     3,     4,    4,     7,    7,    7,    8,    10}}
 *
 * output:       {{   11,    2,     3,     4,   11,    11,    7,   11,   11,    10}}
 * @endcode
 *
 * @throw  cudf::logic_error if input.size() != target.num_columns()
 * @throws cudf::logic_error if any `i`th input_scalar type != `i`th target_column type
 * @throws cudf::logic_error if boolean_mask.type() != bool
 * @throws cudf::logic_error if boolean_mask.size() != target.size()
 *
 * @param[in] input scalars to scatter
 * @param[in] target table_view to modify with scattered values from `input`
 * @param[in] boolean_mask column_view which acts as boolean mask.
 * @param[in] mr Device memory resource used to allocate device memory of the returned table.
 *
 * @returns Returns a table by scattering `input` into `target` as per `boolean_mask`.
 */
std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<const scalar>> const& input,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Get the element at specified index from a column
 *
 * @warning This function is expensive (invokes a kernel launch). So, it is not
 * recommended to be used in performance sensitive code or inside a loop.
 *
 * @throws cudf::logic_error if `index` is not within the range `[0, input.size())`
 *
 * @param input Column view to get the element from
 * @param index Index into `input` to get the element at
 * @param mr Device memory resource used to allocate the returned scalar's device memory.
 * @return std::unique_ptr<scalar> Scalar containing the single value
 */
std::unique_ptr<scalar> get_element(
  column_view const& input,
  size_type index,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Indicates whether a row can be sampled more than once.
 */
enum class sample_with_replacement : bool {
  FALSE,  // A row can be sampled only once
  TRUE    // A row can be sampled more than once
};

/**
 * @brief Gather `n` samples from given `input` randomly
 *
 * @code{.pseudo}
 * Example:
 * input: {col1: {1, 2, 3, 4, 5}, col2: {6, 7, 8, 9, 10}}
 * n: 3
 * replacement: false
 *
 * output:       {col1: {3, 1, 4}, col2: {8, 6, 9}}
 *
 * replacement: true
 *
 * output:       {col1: {3, 1, 1}, col2: {8, 6, 6}}
 * @endcode
 *
 * @throws cudf::logic_error if `n` > `input.num_rows()` and `replacement` == FALSE.
 * @throws cudf::logic_error if `n` < 0.
 *
 * @param input View of a table to sample.
 * @param n non-negative number of samples expected from `input`.
 * @param replacement Allow or disallow sampling of the same row more than once.
 * @param seed Seed value to initiate random number generator.
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return std::unique_ptr<table> Table containing samples from `input`
 */
std::unique_ptr<table> sample(
  table_view const& input,
  size_type const n,
  sample_with_replacement replacement = sample_with_replacement::FALSE,
  int64_t const seed                  = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */
}  // namespace cudf
