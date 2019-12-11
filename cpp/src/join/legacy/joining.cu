/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
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



#include <cudf/types.hpp>
#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <cudf/legacy/copying.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <nvstrings/NVCategory.h>
#include <copying/legacy/gather.hpp>
#include "joining.h"

#include <limits>
#include <set>
#include <vector>
#include <numeric>
#include <algorithm>

// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
using output_index_type = cudf::size_type;
constexpr output_index_type MAX_JOIN_SIZE{std::numeric_limits<output_index_type>::max()};

namespace cudf {
/* --------------------------------------------------------------------------*/
/**
 * @brief  Allocates a buffer and fills it with a repeated value
 *
 * @param[in,out] buffer Address of the buffer to be allocated
 * @param[in] buffer_length Amount of memory to be allocated
 * @param[in] value The value to be filled into the buffer
 * @tparam data_type The data type to be used for the buffer
 * 
 * @returns GDF_SUCCESS upon succesful completion
 */
/* ----------------------------------------------------------------------------*/
template <typename data_type>
gdf_error allocValueBuffer(data_type ** buffer,
                           const cudf::size_type buffer_length,
                           const data_type value) 
{
    RMM_TRY( RMM_ALLOC((void**)buffer, buffer_length*sizeof(data_type), 0) );
    thrust::fill(thrust::device, *buffer, *buffer + buffer_length, value);
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
 * @brief  Allocates a buffer and fills it with a sequence
 *
 * @param[in,out] buffer Address of the buffer to be allocated
 * @param[in] buffer_length Amount of memory to be allocated
 * @tparam data_type The data type to be used for the buffer
 * 
 * @returns GDF_SUCCESS upon succesful completion
 */
/* ----------------------------------------------------------------------------*/
template <typename data_type>
gdf_error allocSequenceBuffer(data_type ** buffer,
                              const cudf::size_type buffer_length) 
{
    RMM_TRY( RMM_ALLOC((void**)buffer, buffer_length*sizeof(data_type), 0) );
    thrust::sequence(thrust::device, *buffer, *buffer + buffer_length);
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Trivially computes full join of two tables if one of the tables
 * are empty
 *
 * @throws cudf::logic_error
 * "Dataset is empty" if both left_table and right_table are empty
 * 
 * @param[in] left_size The size of the left table
 * @param[in] right_size The size of the right table
 * @param[in] rightcol The right set of columns to join
 * @param[out] left_result The join computed indices of the left table
 * @param[out] right_result The join computed indices of the right table
 */
/* ----------------------------------------------------------------------------*/
void trivial_full_join(
        const cudf::size_type left_size,
        const cudf::size_type right_size,
        gdf_column *left_result,
        gdf_column *right_result) {
    // Deduce the type of the output gdf_columns
    gdf_dtype dtype;
    switch(sizeof(output_index_type))
    {
      case 1 : dtype = GDF_INT8;  break;
      case 2 : dtype = GDF_INT16; break;
      case 4 : dtype = GDF_INT32; break;
      case 8 : dtype = GDF_INT64; break;
    }

    output_index_type *l_ptr{nullptr};
    output_index_type *r_ptr{nullptr};
    cudf::size_type result_size{0};
    CUDF_EXPECTS((left_size != 0) || (right_size != 0), "Dataset is empty");
    if (left_size == 0) {
        allocValueBuffer(&l_ptr, right_size,
                         static_cast<output_index_type>(-1));
        allocSequenceBuffer(&r_ptr, right_size);
        result_size = right_size;
    } else if (right_size == 0) {
        allocValueBuffer(&r_ptr, left_size,
                         static_cast<output_index_type>(-1));
        allocSequenceBuffer(&l_ptr, left_size);
        result_size = left_size;
    }

    gdf_column_view_augmented(left_result,
                              l_ptr, nullptr,
                              result_size, dtype, 0,
                              left_result->dtype_info,
                              left_result->col_name);

    gdf_column_view_augmented(right_result,
                              r_ptr, nullptr,
                              result_size, dtype, 0,
                              right_result->dtype_info,
                              right_result->col_name);

    CHECK_CUDA(0);
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the join operation between two sets of columns
 *
 * @throws cudf::logic_error
 * If `left`/`right` table is empty
 * If number of rows in table is too big
 * If it has in-valid join context
 * If method is sort based and number of columns to join are more than `1`
 * 
 * @param[in] left  Table of left columns to join
 * @param[in] right Table of right  columns to join
 * @param[out] left_result The join computed indices of the `left` table
 * @param[out] right_result The join computed indices of the `right` table
 * @param[in] join_context A structure that determines various run parameters, such as
 *                         whether to perform a hash or sort based join
 * @tparam join_type The type of join to be performed
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
void join_call(cudf::table const& left, cudf::table const& right,
                     gdf_column *left_result, gdf_column *right_result,
                     gdf_context *join_context)
{
  CUDF_EXPECTS( 0 != left.num_columns(), "Left Dataset is empty");
  CUDF_EXPECTS( 0 != right.num_columns(), "Right Dataset is empty");
  CUDF_EXPECTS( nullptr != join_context, "Invalid join context");
  CUDF_EXPECTS( left.num_rows() < MAX_JOIN_SIZE, "left column size is too big");
  CUDF_EXPECTS( right.num_rows() < MAX_JOIN_SIZE, "right column size is too big");

  // If both frames are empty, return immediately
  if((0 == left.num_rows() ) && (0 == right.num_rows())) {
    return;
  }

  // If left join and the left table is empty, return immediately
  if( (JoinType::LEFT_JOIN == join_type) && (0 == left.num_rows())){
    return;
  }

  // If Inner Join and either table is empty, return immediately
  if( (JoinType::INNER_JOIN == join_type) && 
      ((0 == left.num_rows()) || (0 == right.num_rows())) ){
    return;
  }

  // If Full Join and either table is empty, compute trivial full join
  if( (JoinType::FULL_JOIN == join_type) && 
      ((0 == left.num_rows()) || (0 == right.num_rows())) ){
    return trivial_full_join(left.num_rows(), right.num_rows(), left_result, right_result);
  }

  // check that the columns data are not null, have matching types, 
  // and the same number of rows
  for (int i = 0; i < left.num_columns(); i++) {
    CUDF_EXPECTS (!((left.num_rows() > 0) && (nullptr == left.get_column(i)->data)), "Null column data in left set");
    CUDF_EXPECTS (!((right.num_rows() > 0) && (nullptr == right.get_column(i)->data)), "Null column data in right set");
    CUDF_EXPECTS (right.get_column(i)->dtype == left.get_column(i)->dtype, "DTYPE mismatch");

    // Ensure GDF_TIMESTAMP columns have the same resolution
    if (GDF_TIMESTAMP == right.get_column(i)->dtype) {
      CUDF_EXPECTS(
          right.get_column(i)->dtype_info.time_unit == left.get_column(i)->dtype_info.time_unit,
          "Timestamp resolution mismatch");
    }
  }

  gdf_method join_method = join_context->flag_method; 
  gdf_error gdf_error_code{GDF_SUCCESS};

  nvtx::range_push("CUDF_JOIN", nvtx::JOIN_COLOR);

  switch(join_method)
  {
    case GDF_HASH:
      {
        gdf_error_code = join_hash<join_type, output_index_type>(left, right, left_result, right_result);
        CUDF_EXPECTS(gdf_error_code == GDF_SUCCESS, "GDF Error");
        break;
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == left.num_columns())
        {
          gdf_error_code =  sort_join<join_type, output_index_type>(const_cast <gdf_column*> (left.get_column(0)), 
                            const_cast <gdf_column*> (right.get_column(0)), left_result, right_result);
          CUDF_EXPECTS(gdf_error_code != GDF_VALIDITY_UNSUPPORTED, "GDF Validity is unsupported by sort_join");
          CUDF_EXPECTS(gdf_error_code == GDF_SUCCESS, "GDF Error");
        }
        else
        {
          CUDF_FAIL("Sort-based join only supports a single \"on\" column.");
        }

        break;
      }
    default:
      CUDF_FAIL("Unsupported join Method");
  }

  nvtx::range_pop();
}

/**---------------------------------------------------------------------------*
 * @brief Returns a vector with non-common indices which is set difference
 * between `[0, num_columns)` and index values in common_column_indices
 *
 * @param num_columns The number of columns , which represents column indices
 * from `[0, num_columns)` in a table
 * @param common_column_indices A vector of common indices which needs to be
 * excluded from `[0, num_columns)`
 * @return vector A vector containing only the indices which are not present in
 * `common_column_indices`
 *---------------------------------------------------------------------------**/

auto non_common_column_indices(
    cudf::size_type num_columns,
    std::vector<cudf::size_type> const& common_column_indices) {
  CUDF_EXPECTS(common_column_indices.size() <= static_cast<unsigned long>(num_columns),
               "Too many columns in common");
  std::vector<cudf::size_type> all_column_indices(num_columns);
  std::iota(std::begin(all_column_indices), std::end(all_column_indices), 0);
  std::vector<cudf::size_type> sorted_common_column_indices{
      common_column_indices};
  std::sort(std::begin(sorted_common_column_indices),
            std::end(sorted_common_column_indices));
  std::vector<cudf::size_type> non_common_column_indices(num_columns -
                                                common_column_indices.size());
  std::set_difference(std::cbegin(all_column_indices),
                      std::cend(all_column_indices),
                      std::cbegin(sorted_common_column_indices),
                      std::cend(sorted_common_column_indices), std::begin(non_common_column_indices));
   return non_common_column_indices;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Gathers rows indicated by `left_indices` and `right_indices` from
 * tables `left` and `right`, respectively, into a single `table`.
 *
 * The row from `left` at `left_indices[i]` will be concatenated with the row i
 * from `right` at `right_indices[i]` to form a new row in the output `table`.
 * If either `left_indices[i]` or `right_indices[i]` is negative, then the i
 * contributions from `left` or `right` will be NULL.
 *
 * @throws cudf::logic_error
 * If call to nvcategory_gather_table fails
 * 
 * @param[in] left   The left table
 * @param[in] right  the right table
 * @param[in] columns_in_common is a vector of pairs of column indices
 * from tables `left` and `right` respectively, that are "in common".
 * For "common" columns, only a single output column will be produced.
 * For an inner or left join, the result will be gathered from the column in
 * `left`. For a full join, the result will be gathered from both common
 * columns in `left` and `right`.
 * @param[in] left_indices Row indices from `left` to gather. If any row index
 * is out of bounds, the contribution in the output `table` will be NULL.
 * @param[in] right_indicess  Row indices from `right` to gather. If any row
 * index is out of bounds, the contribution in the output `table` will be NULL.
 * 
 * @returns `table` containing the concatenation of rows from `left` and
 * `right` specified by `left_indices` and `right_indices`, respectively.
 * For any columns indicated by `columns_in_common`, only the corresponding
 * column in `left` will be included in the result. Final form would look like
 * `left(including common columns)+right(excluding common columns)`.
 */
/* ----------------------------------------------------------------------------*/

template <JoinType join_type, typename index_type>
cudf::table construct_join_output_df(
        cudf::table const& left, 
        cudf::table const& right,
        std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
        gdf_column * left_indices,
        gdf_column * right_indices) {

    nvtx::range_push("CUDF_JOIN_OUTPUT", nvtx::JOIN_COLOR);
    //create left and right input table with columns not joined on
    std::vector<cudf::size_type> left_columns_in_common (columns_in_common.size());
    std::vector<cudf::size_type> right_columns_in_common (columns_in_common.size());

    for (unsigned int i = 0; i < columns_in_common.size(); ++i)
    {
        left_columns_in_common[i] = columns_in_common[i].first;
        right_columns_in_common[i] = columns_in_common[i].second;
    }
  
    // Gathering the indices that are not in common
    std::vector<cudf::size_type> left_non_common_indices =
                                    non_common_column_indices(left.num_columns(), left_columns_in_common);
    std::vector<cudf::size_type> right_non_common_indices =
                                     non_common_column_indices(right.num_columns(), right_columns_in_common);

    cudf::size_type join_size = left_indices->size;
    // Update first set of type and type infos from left
    std::vector<gdf_dtype> result_dtypes = cudf::column_dtypes(left);
    std::vector<gdf_dtype_extra_info> result_dtype_infos = cudf::column_dtype_infos(left);

    // Gathering all the right table columns not in joining indices
    for (auto index: right_non_common_indices){
        result_dtypes.push_back(right.get_column(index)->dtype);
        result_dtype_infos.push_back(right.get_column(index)->dtype_info);
    }

    cudf::table result(join_size, result_dtypes, result_dtype_infos, true);
    bool const ignore_out_of_bounds{ join_type != JoinType::INNER_JOIN };

    // Construct the left columns
    if (not left_non_common_indices.empty()) {
      cudf::table left_source_table = left.select(left_non_common_indices);
      cudf::table left_destination_table = result.select(left_non_common_indices);

      cudf::detail::gather(&left_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &left_destination_table, false, ignore_out_of_bounds);
      gdf_error update_err = nvcategory_gather_table(left_source_table,left_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error");
    }

    // Construct the right columns
    if (not right_non_common_indices.empty()) {
      std::vector<cudf::size_type> result_right_non_common_indices (right_non_common_indices.size());
      std::iota(std::begin(result_right_non_common_indices),
                std::end(result_right_non_common_indices), left.num_columns());
      cudf::table right_source_table = right.select(right_non_common_indices);
      cudf::table right_destination_table = result.select(result_right_non_common_indices);

      cudf::detail::gather(&right_source_table,
                           static_cast<index_type const *>(right_indices->data),
                           &right_destination_table, false, ignore_out_of_bounds);
      gdf_error update_err = nvcategory_gather_table(right_source_table,right_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error");
    }

    // Construct the joined columns
    if (not columns_in_common.empty()) {

      // Gather the columns which join into single column from joined columns
      cudf::table join_source_table = left.select(left_columns_in_common);
      cudf::table join_destination_table = result.select(left_columns_in_common);

      // Gather valid rows from the right table
      // TODO: Revisit this, because it probably can be done more efficiently
      if (JoinType::FULL_JOIN == join_type) {
        cudf::table right_source_table = right.select(right_columns_in_common);

        cudf::detail::gather(
            &right_source_table,
            static_cast<index_type const *>(right_indices->data),
            &join_destination_table, false, ignore_out_of_bounds);
      }

      cudf::detail::gather(&join_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &join_destination_table, false, ignore_out_of_bounds);
      gdf_error update_err = nvcategory_gather_table(join_source_table,join_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error");
    }

    CHECK_CUDA(0);
    nvtx::range_pop();
    return result;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Performs join on the columns provided in `left` and `right` as per
 * the joining indices given in `left_on` and `right_on` and creates a single
 * table.
 *
 * @throws cudf::logic_error
 * if a sort-based join is requested and either `right_on` or `left_on` contains null values.
 *
 * @param[in] left The left table
 * @param[in] right The right table
 * @param[in] left_on The column's indices from `left` to join on.
 * Column `i` from `left_on` will be compared against column `i` of `right_on`.
 * @param[in] right_on The column's indices from `right` to join on.
 * Column `i` from `right_on` will be compared with column `i` of `left_on`. 
 * @param[in] columns_in_common is a vector of pairs of column indices into
 * `left_on` and `right_on`, respectively, that are "in common". For "common"
 * columns, only a single output column will be produced, which is gathered
 * from `left_on` if it is left join or from intersection of `left_on` and `right_on`
 * if it is inner join or gathered from both `left_on` and `right_on` if it is full join.
 * Else, for every column in `left_on` and `right_on`, an output column will be produced.
 *
 * @param[out] * @returns joined_indices Optional, if not `nullptr`, on return, will
 * contain two non-nullable, `GDF_INT32` columns containing the indices of
 * matching rows between `left_on` and `right_on`. The first column corresponds to
 * rows in `left_on`, and the second to `right_on`. A value of `-1` in the second column
 * indicates that the corresponding row in `left_on` has no match. And similarly `-1` in
 * first column indicates that the corresponding row in `right_on` has no match.
 * It is the caller's responsibility to free these columns.
 * @param[in] join_context The context to use to control how
 * the join is performed,e.g., sort vs hash based implementation
 *
 * @returns Result of joining `left` and `right` tables on the columns
 * specified by `left_on` and `right_on`. The resulting table will be joined columns of
 * `left(including common columns)+right(excluding common columns)`.
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type, typename index_type>
cudf::table join_call_compute_df(
                         cudf::table const& left, 
                         cudf::table const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudf::table *joined_indices,
                         gdf_context *join_context) {
 
  CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");
  CUDF_EXPECTS (nullptr != join_context, "Join context is invalid");
  CUDF_EXPECTS(std::none_of(std::cbegin(left), std::cend(left), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported left column dtype");
  CUDF_EXPECTS(std::none_of(std::cbegin(right), std::cend(right), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported right column dtype");

  std::vector<cudf::size_type> right_columns_in_common (columns_in_common.size());

  for (unsigned int i = 0; i < columns_in_common.size(); ++i)
  {
      right_columns_in_common [i] = columns_in_common[i].second;
  }

  cudf::table empty_left = cudf::empty_like(left);
  cudf::table empty_right = cudf::empty_like(right);
  std::vector <cudf::size_type> right_non_common_indices = non_common_column_indices(right.num_columns(),
                                                                         right_columns_in_common);;
  cudf::table tmp_right_table = empty_right.select(right_non_common_indices);
  cudf::table tmp_table = cudf::concat(empty_left, tmp_right_table);
  
  // If there is nothing to join, then send empty table with all columns
  if (left_on.empty() || right_on.empty() || left_on.size() != right_on.size())
  {
      return tmp_table;
  }

  // Even though the resulting table might be empty, but the column should match the expected dtypes and other necessary information
  // So, there is a possibility that there will be lesser number of right columns, so the tmp_table.
  // If the inputs are empty, immediately return
  if ((0 == left.num_rows()) && (0 == right.num_rows())) {
      return tmp_table;
  }

  // If left join and the left table is empty, return immediately
  if ((JoinType::LEFT_JOIN == join_type) && (0 == left.num_rows())) {
      return tmp_table;
  }

  // If Inner Join and either table is empty, return immediately
  if ((JoinType::INNER_JOIN == join_type) &&
      ((0 == left.num_rows()) || (0 == right.num_rows()))) {
      return cudf::empty_like(tmp_table);
  }

  //if the inputs are nvcategory we need to make the dictionaries comparable
  bool at_least_one_category_column = std::any_of (cbegin(left_on), cend(left_on),
                                                  [&](auto index) {  return (left.get_column(index)->dtype == GDF_STRING_CATEGORY);});
  
  std::vector<gdf_column*> new_left_cols(left.num_columns());
  std::vector<gdf_column*> new_right_cols(right.num_columns());

  std::transform (std::cbegin(left), std::cend(left),
                  std::begin(new_left_cols), [](auto col) { return const_cast<gdf_column*>(col); });
  std::transform (std::cbegin(right), std::cend(right),
                  std::begin(new_right_cols), [](auto col) { return const_cast<gdf_column*>(col); });

  std::vector<gdf_column *> temp_columns_to_free;
  if(at_least_one_category_column){
    for(unsigned int join_column_index = 0; join_column_index < left_on.size(); join_column_index++){
      if(left.get_column(left_on[join_column_index])->dtype == GDF_STRING_CATEGORY){
        CUDF_EXPECTS(right.get_column(right_on[join_column_index])->dtype == GDF_STRING_CATEGORY, "GDF type mismatch");

        gdf_column * left_original_column = new_left_cols[left_on[join_column_index]];
        gdf_column * right_original_column = new_right_cols[right_on[join_column_index]];

        gdf_column * new_left_column_ptr = new gdf_column{};
        gdf_column * new_right_column_ptr = new gdf_column{};

        temp_columns_to_free.push_back(new_left_column_ptr);
        temp_columns_to_free.push_back(new_right_column_ptr);

        gdf_column * input_join_columns_merge[2] = {left_original_column, right_original_column};
        gdf_column * new_join_columns[2] = {new_left_column_ptr,
            new_right_column_ptr};
        gdf_column_view(new_left_column_ptr, nullptr, nullptr, left_original_column->size, GDF_STRING_CATEGORY);
        gdf_column_view(new_right_column_ptr, nullptr, nullptr, right_original_column->size, GDF_STRING_CATEGORY);

        int col_width = cudf::byte_width(*new_left_column_ptr);
        RMM_TRY( RMM_ALLOC(&(new_left_column_ptr->data), col_width * left_original_column->size, 0) ); // TODO: non-default stream?
        if(left_original_column->valid != nullptr){
          RMM_TRY( RMM_ALLOC(&(new_left_column_ptr->valid), sizeof(cudf::valid_type)*gdf_valid_allocation_size(left_original_column->size), 0) );
          CUDA_TRY( cudaMemcpy(new_left_column_ptr->valid, left_original_column->valid, sizeof(cudf::valid_type)*gdf_num_bitmask_elements(left_original_column->size),cudaMemcpyDeviceToDevice) );
        }else{
          new_left_column_ptr->valid = nullptr;
        }
        new_left_column_ptr->null_count = left_original_column->null_count;


        RMM_TRY( RMM_ALLOC(&(new_right_column_ptr->data), col_width * right_original_column->size, 0) ); // TODO: non-default stream?
        if(right_original_column->valid != nullptr){
          RMM_TRY( RMM_ALLOC(&(new_right_column_ptr->valid), sizeof(cudf::valid_type)*gdf_valid_allocation_size(right_original_column->size), 0) );
          CUDA_TRY( cudaMemcpy(new_right_column_ptr->valid, right_original_column->valid, sizeof(cudf::valid_type)*gdf_num_bitmask_elements(right_original_column->size),cudaMemcpyDeviceToDevice) );
        }else{
          new_right_column_ptr->valid = nullptr;
        }
        new_right_column_ptr->null_count = right_original_column->null_count;
        gdf_error err = sync_column_categories(input_join_columns_merge,
            new_join_columns,
            2);

        CUDF_EXPECTS(GDF_SUCCESS == err, "GDF_ERROR");

        new_left_cols[left_on[join_column_index]] = new_join_columns[0];
        new_right_cols[right_on[join_column_index]] = new_join_columns[1];
      }
    }
  }

  cudf::table  updated_left_table(new_left_cols);
  cudf::table  updated_right_table(new_right_cols);
  using gdf_col_pointer =
      typename std::unique_ptr<gdf_column, std::function<void(gdf_column *)>>;
  auto gdf_col_deleter = [](gdf_column *col) {
    col->size = 0;
    if (col->data) {
      RMM_FREE(col->data, 0);
    }
    if (col->valid) {
      RMM_FREE(col->valid, 0);
    }
  };

  gdf_col_pointer left_index_temp, right_index_temp;
  gdf_column *left_index_out = nullptr;
  gdf_column *right_index_out = nullptr;

  if (nullptr != joined_indices && joined_indices->num_columns () > 0)
  {
      left_index_out = const_cast <gdf_column*>(joined_indices->get_column(0));
      right_index_out = const_cast <gdf_column*>(joined_indices->get_column(1));
  }
  else
  {
      left_index_temp = {new gdf_column{}, gdf_col_deleter};
      left_index_out = left_index_temp.get();

      right_index_temp = {new gdf_column{}, gdf_col_deleter};
      right_index_out = right_index_temp.get();
  }

  //get column pointers to join on
  join_call<join_type>(updated_left_table.select(left_on),
            updated_right_table.select(right_on),
            left_index_out, right_index_out,
            join_context);

  cudf::table result =
      construct_join_output_df<join_type, index_type>(
          updated_left_table, updated_right_table, 
          columns_in_common, 
          left_index_out, right_index_out);
  left_index_temp.reset(nullptr);
  right_index_temp.reset(nullptr);

  //freeing up the temp column used to synch categories between columns
  for(unsigned int column_to_free = 0; column_to_free < temp_columns_to_free.size(); column_to_free++){
      gdf_column_free(temp_columns_to_free[column_to_free]);
      delete temp_columns_to_free[column_to_free];
  }

  CHECK_CUDA(0);
    
  return result;
}

cudf::table left_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudf::table *joined_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::LEFT_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     columns_in_common,
                     joined_indices,
                     join_context);
}

cudf::table inner_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudf::table *joined_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::INNER_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     columns_in_common,
                     joined_indices,
                     join_context);
}

cudf::table full_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         std::vector<cudf::size_type> const& left_on,
                         std::vector<cudf::size_type> const& right_on,
                         std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
                         cudf::table *joined_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::FULL_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     columns_in_common,
                     joined_indices,
                     join_context);
}
}
