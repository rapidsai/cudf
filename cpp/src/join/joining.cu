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
#include <cudf/copying.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <utilities/nvtx/nvtx_utils.h>
#include <string/nvcategory_util.hpp>
#include <nvstrings/NVCategory.h>
#include <copying/gather.hpp>
#include "joining.h"

#include <limits>
#include <set>
#include <vector>

// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
using output_index_type = gdf_index_type;
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
                           const gdf_size_type buffer_length,
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
                              const gdf_size_type buffer_length) 
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
 * @param[in] left_size The size of the left table
 * @param[in] right_size The size of the right table
 * @param[in] rightcol The right set of columns to join
 * @param[out] left_result The join computed indices of the left table
 * @param[out] right_result The join computed indices of the right table
 */
/* ----------------------------------------------------------------------------*/
void trivial_full_join(
        const gdf_size_type left_size,
        const gdf_size_type right_size,
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
    gdf_size_type result_size{0};
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
    gdf_column_view( left_result, l_ptr, nullptr, result_size, dtype);
    gdf_column_view(right_result, r_ptr, nullptr, result_size, dtype);
    CUDA_CHECK_LAST();
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the join operation between two sets of columns
 * 
 * @param[in] num_cols The number of columns to join
 * @param[in] leftcols  cudf table of left columns to join
 * @param[in] rightcols cudf table of right  columns to join
 * @param[out] left_result The join computed indices of the left table
 * @param[out] right_result The join computed indices of the right table
 * @param[in] join_context A structure that determines various run parameters, such as
 *                         whether to perform a hash or sort based join
 * @tparam join_type The type of join to be performed
 * 
 * @returns void
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
void join_call( int num_cols, cudf::table leftcols, cudf::table rightcols,
                     gdf_column *left_result, gdf_column *right_result,
                     gdf_context *join_context)
{
  CUDF_EXPECTS( 0 != num_cols, "Dataset is empty");
  CUDF_EXPECTS( 0 != leftcols.num_columns(), "Left Dataset is empty");
  CUDF_EXPECTS( 0 != rightcols.num_columns(), "Right Dataset is empty");
  CUDF_EXPECTS( nullptr != join_context, "Invalid join context");

  const auto left_col_size = leftcols.get_column(0)->size;
  const auto right_col_size = rightcols.get_column(0)->size;
  
  CUDF_EXPECTS( left_col_size < MAX_JOIN_SIZE, "left column size is too big");
  CUDF_EXPECTS( right_col_size < MAX_JOIN_SIZE, "right column size is too big");


  // If both frames are empty, return immediately
  if((0 == left_col_size ) && (0 == right_col_size)) {
    return;
  }

  // If left join and the left table is empty, return immediately
  if( (JoinType::LEFT_JOIN == join_type) && (0 == left_col_size)){
    return;
  }

  // If Inner Join and either table is empty, return immediately
  if( (JoinType::INNER_JOIN == join_type) && 
      ((0 == left_col_size) || (0 == right_col_size)) ){
    return;
  }

  // If Full Join and either table is empty, compute trivial full join
  if( (JoinType::FULL_JOIN == join_type) && 
      ((0 == left_col_size) || (0 == right_col_size)) ){
    trivial_full_join(left_col_size, right_col_size, left_result, right_result);
    return;  
  }

  // check that the columns data are not null, have matching types, 
  // and the same number of rows
  for (int i = 0; i < num_cols; i++) {
    CUDF_EXPECTS (!((left_col_size > 0) && (nullptr == leftcols.get_column(i)->data)), "One of the column is null left column set");
    CUDF_EXPECTS (!((right_col_size > 0) && (nullptr == rightcols.get_column(i)->data)), "One of the column is null in right column set");
    CUDF_EXPECTS (rightcols.get_column(i)->dtype == leftcols.get_column(i)->dtype, "DTYPE mismatch");
    CUDF_EXPECTS (left_col_size == leftcols.get_column(i)->size, "left column size mismatch");
    CUDF_EXPECTS (right_col_size == rightcols.get_column(i)->size, "right column size mismatch");

    // Ensure GDF_TIMESTAMP columns have the same resolution
    if (GDF_TIMESTAMP == rightcols.get_column(i)->dtype) {
      CUDF_EXPECTS(
          rightcols.get_column(i)->dtype_info.time_unit == leftcols.get_column(i)->dtype_info.time_unit,
          "Timestamp resolution mismatch");
    }
  }

  gdf_method join_method = join_context->flag_method; 
  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_JOIN", JOIN_COLOR);

  switch(join_method)
  {
    case GDF_HASH:
      {
        gdf_error_code = join_hash<join_type, output_index_type>(leftcols, rightcols, left_result, right_result);
        CUDF_EXPECTS(gdf_error_code == GDF_SUCCESS, "GDF Error");
        break;
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == num_cols)
        {
          gdf_error_code =  sort_join<join_type, output_index_type>(leftcols.get_column(0), rightcols.get_column(0), left_result, right_result);
          CUDF_EXPECTS(gdf_error_code == GDF_SUCCESS, "GDF Error");
        }
        else
        {
          CUDF_EXPECTS(false, "Too many columns to join");
        }

        break;
      }
    default:
      CUDF_EXPECTS(false, "Unsupported Method");
  }

  POP_RANGE();
}

template <JoinType join_type, typename index_type>
std::pair<cudf::table, cudf::table> construct_join_output_df(
        cudf::table const& ljoincols,
        cudf::table const& rjoincols,
        cudf::table const& left_cols, 
        std::vector<int> & left_j_cols,
        cudf::table const& right_cols,
        gdf_column * left_indices,
        gdf_column * right_indices,
        std::vector<int> const& l_common_name_join_ind,
        std::vector<int> const& r_common_name_join_ind) {


    PUSH_RANGE("LIBGDF_JOIN_OUTPUT", JOIN_COLOR);
    //create left and right input table with columns not joined on
    int num_left_cols = left_cols.num_columns();
    int num_right_cols = right_cols.num_columns();
    int num_cols_joined_result = l_common_name_join_ind.size();
    gdf_size_type join_size = left_indices->size;
    std::vector <gdf_dtype> rdtypes;
    std::vector <gdf_dtype_extra_info> rdtype_infos;

    std::vector<gdf_column*> lnonjoincol;
    std::vector<gdf_column*> rnonjoincol;
    for (int i = 0; i < num_left_cols; ++i) {
        if (std::find(l_common_name_join_ind.begin(), l_common_name_join_ind.end(), i)
            == l_common_name_join_ind.end()) {
            lnonjoincol.push_back(const_cast<gdf_column*>(left_cols.get_column(i)));
        }
    }
    for (int i = 0; i < num_right_cols; ++i) {
        if (std::find(r_common_name_join_ind.begin(), r_common_name_join_ind.end(), i) 
            == r_common_name_join_ind.end()) {
            rnonjoincol.push_back(const_cast<gdf_column*>(right_cols.get_column(i)));
            rdtypes.push_back(right_cols.get_column(i)->dtype);
            rdtype_infos.push_back(right_cols.get_column(i)->dtype_info);
        }
    }

    cudf::table result_left(join_size, cudf::column_dtypes(left_cols), cudf::column_dtype_infos(left_cols), true);
    cudf::table result_right(join_size, rdtypes, rdtype_infos, true);
    
    std::vector<gdf_column*> result_lnonjoincol;
    std::vector<gdf_column*> result_rnonjoincol;
    std::vector<gdf_column*> result_joincol;

    for (int lindex = 0; lindex < num_left_cols; ++lindex)
    {
        // Accumalate the left non-join col
        if (std::find(l_common_name_join_ind.begin(), l_common_name_join_ind.end(), lindex)
            == l_common_name_join_ind.end()) {
            result_lnonjoincol.push_back(result_left.get_column(lindex));
        }
    }
        
    // Accumalate the join-col 
    for (int i=0; i < num_cols_joined_result; ++i)
    {
        result_joincol.push_back(result_left.get_column(l_common_name_join_ind[i]));
    }
    
    // Accumalate the right non-join col
    for (int rindex = 0; rindex < num_right_cols-num_cols_joined_result; ++rindex)
    {
        result_rnonjoincol.push_back(result_right.get_column(rindex));
    }
 
    bool const check_bounds{ join_type != JoinType::INNER_JOIN };

    // Construct the left columns
    if (0 != lnonjoincol.size()) {
      cudf::table left_source_table(lnonjoincol);
      cudf::table left_destination_table(result_lnonjoincol);

      cudf::detail::gather(&left_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &left_destination_table, check_bounds);

      CHECK_STREAM(0);
      gdf_error update_err = nvcategory_gather_table(left_source_table,left_destination_table);
      CHECK_STREAM(0);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table throwing a GDF error");
    }
    
    // Construct the right columns
    if (0 != rnonjoincol.size()) {
      cudf::table right_source_table(rnonjoincol);
      cudf::table right_destination_table(result_rnonjoincol);

      cudf::detail::gather(&right_source_table,
                           static_cast<index_type const *>(right_indices->data),
                           &right_destination_table, check_bounds);
      CHECK_STREAM(0);
      gdf_error update_err = nvcategory_gather_table(right_source_table,right_destination_table);
      CHECK_STREAM(0);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table throwing a GDF error");
    }

    // Construct the joined columns
    if (0 != ljoincols.num_columns() && num_cols_joined_result > 0) {

      std::vector <gdf_column *> l_join;
      std::vector <gdf_column *> r_join;
      for (int join_ind = 0; join_ind < num_cols_joined_result; ++join_ind)
      {
          std::vector<int>::iterator itr = std::find(left_j_cols.begin(), left_j_cols.end(),
               l_common_name_join_ind[join_ind]);

          int index = std::distance(left_j_cols.begin(), itr);

          l_join.push_back(const_cast<gdf_column*>(ljoincols.get_column(index)));

          if (JoinType::FULL_JOIN == join_type)
          {
              r_join.push_back(const_cast<gdf_column*>(rjoincols.get_column(index)));
          }
      }
      cudf::table join_source_table(l_join);
      cudf::table join_destination_table(result_joincol);

      // Gather valid rows from the right table
      // TODO: Revisit this, because it probably can be done more efficiently
      if (JoinType::FULL_JOIN == join_type) {
        cudf::table right_source_table(r_join);

        cudf::detail::gather(
            &right_source_table,
            static_cast<index_type const *>(right_indices->data),
            &join_destination_table, check_bounds);
        CHECK_STREAM(0);
      }

      cudf::detail::gather(&join_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &join_destination_table, check_bounds);
      CHECK_STREAM(0);
      gdf_error update_err = nvcategory_gather_table(join_source_table,join_destination_table);
      CHECK_STREAM(0);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table throwing a GDF error");
    }
     
    POP_RANGE();
    return std::pair<cudf::table, cudf::table>(result_left, result_right);
}

template <JoinType join_type, typename index_type>
std::pair<cudf::table, cudf::table> join_call_compute_df(
                         cudf::table const& left_cols, 
                         std::vector <int> left_join_cols,
                         cudf::table const& right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         std::vector <int> l_common_name_join_ind,
                         std::vector <int> r_common_name_join_ind,
                         gdf_context *join_context) {

  int num_left_cols = left_cols.num_columns();
  int num_right_cols = right_cols.num_columns();
  int num_cols_to_join = left_join_cols.size();

  std::vector <gdf_column*> tmp_left_cols;
  std::vector <gdf_column*> tmp_right_cols;

  for (int i=0; i < num_left_cols; i++)
  {
      tmp_left_cols.push_back(const_cast<gdf_column *> (left_cols.get_column(i)));
  }

  for (int i=0; i < num_right_cols; ++i)
  {
      if (std::find(r_common_name_join_ind.begin(), r_common_name_join_ind.end(), i)
            == r_common_name_join_ind.end()) {
          tmp_right_cols.push_back(const_cast<gdf_column *> (right_cols.get_column(i)));
      }
  }

  cudf::table tmp_left_table = (tmp_left_cols.size()>0)? cudf::table (tmp_left_cols) : cudf::table{};
  cudf::table tmp_right_table = (tmp_right_cols.size()>0)? cudf::table (tmp_right_cols) : cudf::table{};

  CUDF_EXPECTS (0 != num_left_cols, "Left table is empty");
  CUDF_EXPECTS (0 != num_right_cols, "Right table is empty");
  CUDF_EXPECTS (nullptr != join_context, "Join context is invalid");

  if (0 == num_cols_to_join)
  {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(tmp_left_table), cudf::empty_like(tmp_right_table));
  }

  for(int column_index = 0; column_index  < num_left_cols; column_index++){
    CUDF_EXPECTS(left_cols.get_column(column_index)->dtype != GDF_invalid, "Unsupported Dtype in Left column");
  }
  for(int column_index = 0; column_index  < num_right_cols; column_index++){
    CUDF_EXPECTS(right_cols.get_column(column_index)->dtype != GDF_invalid, "Unsupported Dtype in right column");
  }

  auto const left_col_size = left_cols.get_column(0)->size;
  auto const right_col_size = right_cols.get_column(0)->size;

  // If the inputs are empty, immediately return
  if ((0 == left_col_size) && (0 == right_col_size)) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(tmp_left_table), cudf::empty_like(tmp_right_table));
  }

  // If left join and the left table is empty, return immediately
  if ((JoinType::LEFT_JOIN == join_type) && (0 == left_col_size)) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(tmp_left_table), cudf::empty_like(tmp_right_table));
  }

  // If Inner Join and either table is empty, return immediately
  if ((JoinType::INNER_JOIN == join_type) &&
      ((0 == left_col_size) || (0 == right_col_size))) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(tmp_left_table), cudf::empty_like(tmp_right_table));
  }


  //if the inputs are nvcategory we need to make the dictionaries comparable
  bool at_least_one_category_column = false;
  for(int join_column_index = 0; join_column_index < num_cols_to_join; join_column_index++){
    at_least_one_category_column |= left_cols.get_column(left_join_cols[join_column_index])->dtype == GDF_STRING_CATEGORY;
  }

  std::vector<gdf_column*> new_left_cols;
  std::vector<gdf_column*> new_right_cols;

  for (int i = 0; i < num_left_cols; i++)
      new_left_cols.push_back (const_cast<gdf_column*>(left_cols.get_column(i)));
  for (int i = 0; i < num_right_cols; i++)
      new_right_cols.push_back (const_cast<gdf_column*>(right_cols.get_column(i)));

  std::vector<gdf_column *> temp_columns_to_free;
  if(at_least_one_category_column){
    for(int join_column_index = 0; join_column_index < num_cols_to_join; join_column_index++){
      if(left_cols.get_column(left_join_cols[join_column_index])->dtype == GDF_STRING_CATEGORY){
        CUDF_EXPECTS(right_cols.get_column(right_join_cols[join_column_index])->dtype == GDF_STRING_CATEGORY, "GDF type mismatch");

        gdf_column * left_original_column = new_left_cols[left_join_cols[join_column_index]];
        gdf_column * right_original_column = new_right_cols[right_join_cols[join_column_index]];




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
          RMM_TRY( RMM_ALLOC(&(new_left_column_ptr->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(left_original_column->size), 0) );
          CUDA_TRY( cudaMemcpy(new_left_column_ptr->valid, left_original_column->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(left_original_column->size),cudaMemcpyDeviceToDevice) );
        }else{
          new_left_column_ptr->valid = nullptr;
        }
        new_left_column_ptr->null_count = left_original_column->null_count;


        RMM_TRY( RMM_ALLOC(&(new_right_column_ptr->data), col_width * right_original_column->size, 0) ); // TODO: non-default stream?
        if(right_original_column->valid != nullptr){
          RMM_TRY( RMM_ALLOC(&(new_right_column_ptr->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(right_original_column->size), 0) );
          CUDA_TRY( cudaMemcpy(new_right_column_ptr->valid, right_original_column->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(right_original_column->size),cudaMemcpyDeviceToDevice) );
        }else{
          new_right_column_ptr->valid = nullptr;
        }
        new_right_column_ptr->null_count = right_original_column->null_count;
        gdf_error err = sync_column_categories(input_join_columns_merge,
            new_join_columns,
            2);

        CUDF_EXPECTS(GDF_SUCCESS == err, "GDF_ERROR");

        new_left_cols[left_join_cols[join_column_index]] = new_join_columns[0];
        new_right_cols[right_join_cols[join_column_index]] = new_join_columns[1];
        CHECK_STREAM(0);
      }
    }
  }

  cudf::table  updated_left_cols(new_left_cols);
  cudf::table  updated_right_cols(new_right_cols);
  // If index outputs are not requested, create columns to store them
  // for computing combined join output
  gdf_column *left_index_out = left_indices;
  gdf_column *right_index_out = right_indices;

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
  gdf_col_pointer l_index_temp, r_index_temp;

  if (nullptr == left_indices) {
    l_index_temp = {new gdf_column{}, gdf_col_deleter};
    left_index_out = l_index_temp.get();
    }

    if (nullptr == right_indices) {
        r_index_temp = {new gdf_column{}, gdf_col_deleter};
        right_index_out = r_index_temp.get();
    }

    //get column pointers to join on
    std::vector<gdf_column*> ljoincol;
    std::vector<gdf_column*> rjoincol;
    for (int i = 0; i < num_cols_to_join; ++i) {
        ljoincol.push_back(updated_left_cols.get_column(left_join_cols[i]));
        rjoincol.push_back(updated_right_cols.get_column(right_join_cols[i]));
    }


    cudf::table ljoin_cols(ljoincol);
    cudf::table rjoin_cols(rjoincol);
    join_call<join_type>(num_cols_to_join,
            ljoin_cols, rjoin_cols,
            left_index_out, right_index_out,
            join_context);
    CHECK_STREAM(0);

    std::pair<cudf::table, cudf::table> merged_result =
        construct_join_output_df<join_type, index_type>(
            ljoin_cols, rjoin_cols,
            updated_left_cols, left_join_cols,
            updated_right_cols, left_index_out, right_index_out, 
            l_common_name_join_ind, r_common_name_join_ind);
    CHECK_STREAM(0);
    l_index_temp.reset(nullptr);
    r_index_temp.reset(nullptr);





    //freeing up the temp column used to synch categories between columns
    for(unsigned int column_to_free = 0; column_to_free < temp_columns_to_free.size(); column_to_free++){
      gdf_column_free(temp_columns_to_free[column_to_free]);
      delete temp_columns_to_free[column_to_free];
    }

    CHECK_STREAM(0);
    
    return merged_result;
}

std::pair<cudf::table, cudf::table> left_join(
                         cudf::table const& left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table const& right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         std::vector <int> l_common_name_join_ind,
                         std::vector <int> r_common_name_join_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::LEFT_JOIN, output_index_type>(
                     left_cols, 
                     left_join_cols,
                     right_cols,
                     right_join_cols,
                     left_indices,
                     right_indices,
                     l_common_name_join_ind,
                     r_common_name_join_ind,
                     join_context);
}

std::pair<cudf::table, cudf::table> inner_join(
                         cudf::table const& left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table const& right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         std::vector <int> l_common_name_join_ind,
                         std::vector <int> r_common_name_join_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::INNER_JOIN, output_index_type>(
                     left_cols,
                     left_join_cols,
                     right_cols,
                     right_join_cols,
                     left_indices,
                     right_indices,
                     l_common_name_join_ind,
                     r_common_name_join_ind,
                     join_context);
}

std::pair<cudf::table, cudf::table> full_join(
                         cudf::table const& left_cols,
                         std::vector <int> left_join_cols,
                         cudf::table const& right_cols,
                         std::vector <int> right_join_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         std::vector <int> l_common_name_join_ind,
                         std::vector <int> r_common_name_join_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::FULL_JOIN, output_index_type>(
                     left_cols,
                     left_join_cols,
                     right_cols,
                     right_join_cols,
                     left_indices,
                     right_indices,
                     l_common_name_join_ind,
                     r_common_name_join_ind,
                     join_context);
}
}
