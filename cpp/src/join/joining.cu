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



#include <types.hpp>
#include <cudf.h>
#include <rmm/rmm.h>
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

/* --------------------------------------------------------------------------*/
/** 
 * @brief Computes the Join result between two tables using the hash-based implementation. 
 * 
 * @param[in] num_cols The number of columns to join
 * @param[in] leftcol The left set of columns to join
 * @param[in] rightcol The right set of columns to join
 * @param[out] l_result The join computed indices of the left table
 * @param[out] r_result The join computed indices of the right table
 * @tparam join_type The type of join to be performed
 * 
 * @returns Upon successful computation, returns GDF_SUCCESS. Otherwise returns appropriate error code 
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error hash_join(gdf_size_type num_cols, gdf_column **leftcol, gdf_column **rightcol,
                    gdf_column *l_result, gdf_column *r_result)
{
  cudf::table left_table{leftcol, num_cols};
  cudf::table right_table{rightcol, num_cols};

  return join_hash<join_type, output_index_type>(left_table, right_table,
                                                 l_result, r_result);
}

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
 * 
 * @returns GDF_SUCCESS upon succesfull compute, otherwise returns appropriate error code
 */
/* ----------------------------------------------------------------------------*/
gdf_error trivial_full_join(
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
    if ((left_size == 0) && (right_size == 0)) {
        return GDF_DATASET_EMPTY;
    }
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
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the join operation between two sets of columns
 * 
 * @param[in] num_cols The number of columns to join
 * @param[in] leftcol The left set of columns to join
 * @param[in] rightcol The right set of columns to join
 * @param[out] left_result The join computed indices of the left table
 * @param[out] right_result The join computed indices of the right table
 * @param[in] join_context A structure that determines various run parameters, such as
 *                         whether to perform a hash or sort based join
 * @tparam join_type The type of join to be performed
 * 
 * @returns GDF_SUCCESS upon succesfull compute, otherwise returns appropriate error code
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error join_call( int num_cols, gdf_column **leftcol, gdf_column **rightcol,
                     gdf_column *left_result, gdf_column *right_result,
                     gdf_context *join_context)
{
  GDF_REQUIRE( 0 != num_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE( nullptr != leftcol, GDF_DATASET_EMPTY);
  GDF_REQUIRE( nullptr != rightcol, GDF_DATASET_EMPTY);
  GDF_REQUIRE( nullptr != join_context, GDF_INVALID_API_CALL);

  const auto left_col_size = leftcol[0]->size;
  const auto right_col_size = rightcol[0]->size;
  
  GDF_REQUIRE( left_col_size < MAX_JOIN_SIZE, GDF_COLUMN_SIZE_TOO_BIG);
  GDF_REQUIRE( right_col_size < MAX_JOIN_SIZE, GDF_COLUMN_SIZE_TOO_BIG);


  // If both frames are empty, return immediately
  if((0 == left_col_size ) && (0 == right_col_size)) {
    return GDF_SUCCESS;
  }

  // If left join and the left table is empty, return immediately
  if( (JoinType::LEFT_JOIN == join_type) && (0 == left_col_size)){
    return GDF_SUCCESS;
  }

  // If Inner Join and either table is empty, return immediately
  if( (JoinType::INNER_JOIN == join_type) && 
      ((0 == left_col_size) || (0 == right_col_size)) ){
    return GDF_SUCCESS;
  }

  // If Full Join and either table is empty, compute trivial full join
  if( (JoinType::FULL_JOIN == join_type) && 
      ((0 == left_col_size) || (0 == right_col_size)) ){
    return trivial_full_join(left_col_size, right_col_size, left_result, right_result);
  }

  // check that the columns data are not null, have matching types, 
  // and the same number of rows
  for (int i = 0; i < num_cols; i++) {
    if((right_col_size > 0) && (nullptr == rightcol[i]->data)){
     return GDF_DATASET_EMPTY;
    } 
    if((left_col_size > 0) && (nullptr == leftcol[i]->data)){
     return GDF_DATASET_EMPTY;
    } 
    if(rightcol[i]->dtype != leftcol[i]->dtype) return GDF_DTYPE_MISMATCH;
    if(left_col_size != leftcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;
    if(right_col_size != rightcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;

    // Ensure GDF_TIMESTAMP columns have the same resolution
    if (GDF_TIMESTAMP == rightcol[i]->dtype) {
      GDF_REQUIRE(
          rightcol[i]->dtype_info.time_unit == leftcol[i]->dtype_info.time_unit,
          GDF_TIMESTAMP_RESOLUTION_MISMATCH);
    }
  }

  gdf_method join_method = join_context->flag_method; 

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_JOIN", JOIN_COLOR);

  switch(join_method)
  {
    case GDF_HASH:
      {
        gdf_error_code =  hash_join<join_type>(num_cols, leftcol, rightcol, left_result, right_result);
        break;
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == num_cols)
        {
          gdf_error_code =  sort_join<join_type, output_index_type>(leftcol[0], rightcol[0], left_result, right_result);
        }
        else
        {
          gdf_error_code =  GDF_JOIN_TOO_MANY_COLUMNS;
        }

        break;
      }
    default:
      gdf_error_code =  GDF_UNSUPPORTED_METHOD;
  }

  POP_RANGE();

  return gdf_error_code;
}



template <JoinType join_type, typename index_type>
gdf_error construct_join_output_df(
        std::vector<gdf_column*>& ljoincol,
        std::vector<gdf_column*>& rjoincol,
        gdf_column **left_cols, 
        int num_left_cols,
        int left_join_cols[],
        gdf_column **right_cols,
        int num_right_cols,
        int right_join_cols[],
        int num_cols_to_join,
        int result_num_cols,
        gdf_column ** result_cols,
        gdf_column * left_indices,
        gdf_column * right_indices) {


  PUSH_RANGE("LIBGDF_JOIN_OUTPUT", JOIN_COLOR);
    //create left and right input table with columns not joined on
    std::vector<gdf_column*> lnonjoincol;
    std::vector<gdf_column*> rnonjoincol;
    std::set<int> l_join_indices, r_join_indices;
    for (int i = 0; i < num_cols_to_join; ++i) {
        l_join_indices.insert(left_join_cols[i]);
        r_join_indices.insert(right_join_cols[i]);
    }
    for (int i = 0; i < num_left_cols; ++i) {
        if (l_join_indices.find(i) == l_join_indices.end()) {
            lnonjoincol.push_back(left_cols[i]);
        }
    }
    for (int i = 0; i < num_right_cols; ++i) {
        if (r_join_indices.find(i) == r_join_indices.end()) {
            rnonjoincol.push_back(right_cols[i]);
        }
    }
    //TODO : Invalid api

    gdf_size_type join_size = left_indices->size;
    int left_table_end = num_left_cols - num_cols_to_join;
    int right_table_begin = num_left_cols;

    //create left and right output column data buffers
    for (int i = 0; i < left_table_end; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, lnonjoincol[i]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size)) );
        CHECK_STREAM(0);
    }
    for (int i = right_table_begin; i < result_num_cols; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, rnonjoincol[i - right_table_begin]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size)) );
        CHECK_STREAM(0);
    }
    //create joined output column data buffers
    for (int join_index = 0; join_index < num_cols_to_join; ++join_index) {
        int i = left_table_end + join_index;
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, left_cols[left_join_cols[join_index]]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_valid_allocation_size(join_size)) );
        CHECK_STREAM(0);
    }


    // If the join_type is an outer join, then indices for non-matches will be
    // -1, requiring bounds checking when gathering the result table
    bool const check_bounds{ join_type != JoinType::INNER_JOIN };

    // Construct the left columns
    if (0 != lnonjoincol.size()) {
      cudf::table left_source_table(lnonjoincol.data(), lnonjoincol.size());
      cudf::table left_destination_table(result_cols,
                                         num_left_cols - num_cols_to_join);

      cudf::detail::gather(&left_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &left_destination_table, check_bounds);
      CHECK_STREAM(0);
      gdf_error update_err = nvcategory_gather_table(left_source_table,left_destination_table);
      CHECK_STREAM(0);
      GDF_REQUIRE(update_err == GDF_SUCCESS,update_err);
    }

    // Construct the right columns
    if (0 != rnonjoincol.size()) {
      cudf::table right_source_table(rnonjoincol.data(), rnonjoincol.size());
      cudf::table right_destination_table(result_cols + right_table_begin,
                                          num_right_cols - num_cols_to_join);

      cudf::detail::gather(&right_source_table,
                           static_cast<index_type const *>(right_indices->data),
                           &right_destination_table, check_bounds);
      CHECK_STREAM(0);
      gdf_error update_err = nvcategory_gather_table(right_source_table,right_destination_table);
      CHECK_STREAM(0);
      GDF_REQUIRE(update_err == GDF_SUCCESS,update_err);
    }

    // Construct the joined columns
    if (0 != ljoincol.size()) {
      cudf::table join_source_table(ljoincol.data(), ljoincol.size());
      cudf::table join_destination_table(result_cols + left_table_end,
                                         num_cols_to_join);

      // Gather valid rows from the right table
      // TODO: Revisit this, because it probably can be done more efficiently
      if (JoinType::FULL_JOIN == join_type) {
        cudf::table right_source_table(rjoincol.data(), rjoincol.size());

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
      GDF_REQUIRE(update_err == GDF_SUCCESS,update_err);
    }

    POP_RANGE();
    return GDF_SUCCESS;
}

template <JoinType join_type, typename index_type>
gdf_error join_call_compute_df(
                         gdf_column **left_cols, 
                         int num_left_cols,
                         int left_join_cols[],
                         gdf_column **right_cols,
                         int num_right_cols,
                         int right_join_cols[],
                         int num_cols_to_join,
                         int result_num_cols,
                         gdf_column **result_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context) {
  GDF_REQUIRE(nullptr != left_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != right_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(0 != num_cols_to_join, GDF_SUCCESS);
  GDF_REQUIRE(nullptr != left_join_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != right_join_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != join_context, GDF_INVALID_API_CALL);

  for(int column_index = 0; column_index  < num_left_cols; column_index++){
    GDF_REQUIRE(left_cols[column_index]->dtype != GDF_invalid,GDF_UNSUPPORTED_DTYPE);
  }
  for(int column_index = 0; column_index  < num_right_cols; column_index++){
    GDF_REQUIRE(right_cols[column_index]->dtype != GDF_invalid,GDF_UNSUPPORTED_DTYPE);
  }

  // Determine if requested output is the indices of matching rows, the fully
  // constructed output dataframe, or both
  bool const construct_output_dataframe{nullptr != result_cols};
  bool const return_output_indices{(nullptr != left_indices) and
                                   (nullptr != right_indices)};

  GDF_REQUIRE(construct_output_dataframe or return_output_indices,
              GDF_INVALID_API_CALL);

  auto const left_col_size = left_cols[0]->size;
  auto const right_col_size = right_cols[0]->size;

  // If the inputs are empty, immediately return
  if ((0 == left_col_size) && (0 == right_col_size)) {
    return GDF_SUCCESS;
  }

  // If left join and the left table is empty, return immediately
  if ((JoinType::LEFT_JOIN == join_type) && (0 == left_col_size)) {
    return GDF_SUCCESS;
  }

  // If Inner Join and either table is empty, return immediately
  if ((JoinType::INNER_JOIN == join_type) &&
      ((0 == left_col_size) || (0 == right_col_size))) {
    return GDF_SUCCESS;
  }


  //if the inputs are nvcategory we need to make the dictionaries comparable
  bool at_least_one_category_column = false;
  for(int join_column_index = 0; join_column_index < num_cols_to_join; join_column_index++){
    at_least_one_category_column |= left_cols[left_join_cols[join_column_index]]->dtype == GDF_STRING_CATEGORY;
  }

  std::vector<gdf_column*> new_left_cols(left_cols, left_cols + num_left_cols);
  std::vector<gdf_column*> new_right_cols(right_cols, right_cols + num_right_cols);
  std::vector<gdf_column *> temp_columns_to_free;
  if(at_least_one_category_column){
    for(int join_column_index = 0; join_column_index < num_cols_to_join; join_column_index++){
      if(left_cols[left_join_cols[join_column_index]]->dtype == GDF_STRING_CATEGORY){
        GDF_REQUIRE(right_cols[right_join_cols[join_column_index]]->dtype == GDF_STRING_CATEGORY, GDF_DTYPE_MISMATCH);

        gdf_column * left_original_column = new_left_cols[left_join_cols[join_column_index]];
        gdf_column * right_original_column = new_right_cols[right_join_cols[join_column_index]];




        gdf_column * new_left_column_ptr = new gdf_column;
        gdf_column * new_right_column_ptr = new gdf_column;

        temp_columns_to_free.push_back(new_left_column_ptr);
        temp_columns_to_free.push_back(new_right_column_ptr);


        gdf_column * input_join_columns_merge[2] = {left_original_column, right_original_column};
        gdf_column * new_join_columns[2] = {new_left_column_ptr,
            new_right_column_ptr};
        gdf_column_view(new_left_column_ptr, nullptr, nullptr, left_original_column->size, GDF_STRING_CATEGORY);
        gdf_column_view(new_right_column_ptr, nullptr, nullptr, right_original_column->size, GDF_STRING_CATEGORY);

        int col_width;
        get_column_byte_width(new_left_column_ptr, &col_width);
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

        GDF_REQUIRE(GDF_SUCCESS == err, err);

        new_left_cols[left_join_cols[join_column_index]] = new_join_columns[0];
        new_right_cols[right_join_cols[join_column_index]] = new_join_columns[1];
        CHECK_STREAM(0);
      }
    }


    left_cols = new_left_cols.data();
    right_cols = new_right_cols.data();
  }

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
    l_index_temp = {new gdf_column, gdf_col_deleter};
    left_index_out = l_index_temp.get();
    }

    if (nullptr == right_indices) {
        r_index_temp = {new gdf_column, gdf_col_deleter};
        right_index_out = r_index_temp.get();
    }

    //get column pointers to join on
    std::vector<gdf_column*> ljoincol;
    std::vector<gdf_column*> rjoincol;
    for (int i = 0; i < num_cols_to_join; ++i) {
        ljoincol.push_back(left_cols[ left_join_cols[i] ]);
        rjoincol.push_back(right_cols[ right_join_cols[i] ]);
    }


    gdf_error join_err = join_call<join_type>(num_cols_to_join,
            ljoincol.data(), rjoincol.data(),
            left_index_out, right_index_out,
            join_context);
    CHECK_STREAM(0);
    GDF_REQUIRE(GDF_SUCCESS == join_err, join_err);

    //If construct_output_dataframe is false then left_index_out or right_index_out
    //was not dynamically allocated.
    if (not construct_output_dataframe) {
        return join_err;
    }

    gdf_error df_err =
        construct_join_output_df<join_type, index_type>(
            ljoincol, rjoincol,
            left_cols, num_left_cols, left_join_cols,
            right_cols, num_right_cols, right_join_cols,
            num_cols_to_join, result_num_cols, result_cols,
            left_index_out, right_index_out);
    CHECK_STREAM(0);
    l_index_temp.reset(nullptr);
    r_index_temp.reset(nullptr);





    //freeing up the temp column used to synch categories between columns
    for(unsigned int column_to_free = 0; column_to_free < temp_columns_to_free.size(); column_to_free++){
      NVCategory::destroy(static_cast<NVCategory *>(temp_columns_to_free[column_to_free]->dtype_info.category));
      gdf_column_free(temp_columns_to_free[column_to_free]);
      delete temp_columns_to_free[column_to_free];
    }

    CHECK_STREAM(0);


    return df_err;
}

gdf_error gdf_left_join(
                         gdf_column **left_cols, 
                         int num_left_cols,
                         int left_join_cols[],
                         gdf_column **right_cols,
                         int num_right_cols,
                         int right_join_cols[],
                         int num_cols_to_join,
                         int result_num_cols,
                         gdf_column **result_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::LEFT_JOIN, output_index_type>(
                     left_cols, 
                     num_left_cols,
                     left_join_cols,
                     right_cols,
                     num_right_cols,
                     right_join_cols,
                     num_cols_to_join,
                     result_num_cols,
                     result_cols,
                     left_indices,
                     right_indices,
                     join_context);
}

gdf_error gdf_inner_join(
                         gdf_column **left_cols, 
                         int num_left_cols,
                         int left_join_cols[],
                         gdf_column **right_cols,
                         int num_right_cols,
                         int right_join_cols[],
                         int num_cols_to_join,
                         int result_num_cols,
                         gdf_column **result_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::INNER_JOIN, output_index_type>(
                     left_cols, 
                     num_left_cols,
                     left_join_cols,
                     right_cols,
                     num_right_cols,
                     right_join_cols,
                     num_cols_to_join,
                     result_num_cols,
                     result_cols,
                     left_indices,
                     right_indices,
                     join_context);
}

gdf_error gdf_full_join(
                         gdf_column **left_cols, 
                         int num_left_cols,
                         int left_join_cols[],
                         gdf_column **right_cols,
                         int num_right_cols,
                         int right_join_cols[],
                         int num_cols_to_join,
                         int result_num_cols,
                         gdf_column **result_cols,
                         gdf_column * left_indices,
                         gdf_column * right_indices,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::FULL_JOIN, output_index_type>(
                     left_cols, 
                     num_left_cols,
                     left_join_cols,
                     right_cols,
                     num_right_cols,
                     right_join_cols,
                     num_cols_to_join,
                     result_num_cols,
                     result_cols,
                     left_indices,
                     right_indices,
                     join_context);
}
