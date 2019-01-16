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


#include <limits>
#include <set>
#include <vector>

#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/error_utils.h"
#include "dataframe/cudf_table.cuh"
#include "utilities/nvtx/nvtx_utils.h"

#include "joining.h"

using namespace mgpu;

// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
using output_index_type = int;
constexpr output_index_type MAX_JOIN_SIZE{std::numeric_limits<output_index_type>::max()};

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Computes the Join result between two tables using the hash-based implementation. 
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param l_result The join computed indices of the left table
 * @Param r_result The join computed indices of the right table
 * @tparam join_type The type of join to be performed
 * @tparam size_type The data type used for size calculations
 * 
 * @Returns Upon successful computation, returns GDF_SUCCESS. Otherwise returns appropriate error code 
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type, 
          typename size_type>
gdf_error hash_join(size_type num_cols, gdf_column **leftcol, gdf_column **rightcol,
                    gdf_column *l_result, gdf_column *r_result)
{
  // Wrap the set of gdf_columns in a gdf_table class
  std::unique_ptr< gdf_table<size_type> > left_table(new gdf_table<size_type>(num_cols, leftcol));
  std::unique_ptr< gdf_table<size_type> > right_table(new gdf_table<size_type>(num_cols, rightcol));

  return join_hash<join_type, output_index_type>(*left_table, 
                                                        *right_table, 
                                                        l_result, 
                                                        r_result);
}

template <JoinType join_type>
struct SortJoin {
template<typename launch_arg_t = mgpu::empty_t,
  typename a_it, typename b_it, typename comp_t>
    std::pair<gdf_column, gdf_column>
    operator()(a_it a, int a_count, b_it b, int b_count,
               comp_t comp, context_t& context) {
        return std::pair<gdf_column, gdf_column>();
    }
};

template <>
struct SortJoin<JoinType::INNER_JOIN> {
template<typename launch_arg_t = mgpu::empty_t,
  typename a_it, typename b_it, typename comp_t>
    std::pair<gdf_column, gdf_column>
    operator()(a_it a, int a_count, b_it b, int b_count,
               comp_t comp, context_t& context) {
        return inner_join(a, a_count, b, b_count, comp, context);
    }
};

template <>
struct SortJoin<JoinType::LEFT_JOIN> {
  template<typename launch_arg_t = mgpu::empty_t,
    typename a_it, typename b_it, typename comp_t>
    std::pair<gdf_column, gdf_column>
    operator()(a_it a, int a_count, b_it b, int b_count,
               comp_t comp, context_t& context) {
        return left_join(a, a_count, b, b_count, comp, context);
      }
};

template <JoinType join_type, typename T>
gdf_error sort_join_typed(gdf_column *leftcol, gdf_column *rightcol,
                          gdf_column *left_result, gdf_column *right_result,
                          gdf_context *ctxt) 
{
  using namespace mgpu;
  gdf_error err = GDF_SUCCESS;
  GDF_REQUIRE(!leftcol->valid  || !leftcol->null_count , GDF_VALIDITY_UNSUPPORTED);
  GDF_REQUIRE(!rightcol->valid || !rightcol->null_count, GDF_VALIDITY_UNSUPPORTED);

  rmm_mgpu_context_t context(false);
  SortJoin<join_type> sort_based_join;
  auto output = sort_based_join(static_cast<T*>(leftcol->data), leftcol->size,
                                       static_cast<T*>(rightcol->data), rightcol->size,
                                       less_t<T>(), context);
  *left_result = output.first;
  *right_result = output.second;
  CUDA_CHECK_LAST();

  return err;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Computes the join operation between a single left and single right column
 using the sort based implementation.
 * 
 * @Param leftcol The left column to join
 * @Param rightcol The right column to join
 * @Param left_result The join computed indices of the left table
 * @Param right_result The join computed indices of the right table
 * @Param ctxt Structure that determines various run parameters, such as if the inputs
 are already sorted.
   @tparama join_type The type of join to perform
 * 
 * @Returns GDF_SUCCESS upon succesful completion of the join, otherwise returns 
 appropriate error code.
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error sort_join(gdf_column *leftcol, gdf_column *rightcol,
                    gdf_column *l_result, gdf_column *r_result,
                    gdf_context *ctxt)
{

  if(GDF_SORT != ctxt->flag_method) return GDF_INVALID_API_CALL;

  switch ( leftcol->dtype ){
    case GDF_INT8:      return sort_join_typed<join_type, int8_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_INT16:     return sort_join_typed<join_type,int16_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_INT32:     return sort_join_typed<join_type,int32_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_INT64:     return sort_join_typed<join_type,int64_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_FLOAT32:   return sort_join_typed<join_type,int32_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_FLOAT64:   return sort_join_typed<join_type,int64_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_DATE32:    return sort_join_typed<join_type,int32_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_DATE64:    return sort_join_typed<join_type,int64_t>(leftcol, rightcol, l_result, r_result, ctxt);
    case GDF_TIMESTAMP: return sort_join_typed<join_type,int64_t>(leftcol, rightcol, l_result, r_result, ctxt);
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

template
gdf_error sort_join<JoinType::INNER_JOIN>(gdf_column *leftcol, gdf_column *rightcol,
                                          gdf_column *l_result, gdf_column *r_result,
                                          gdf_context *ctxt);
template
gdf_error sort_join<JoinType::LEFT_JOIN>(gdf_column *leftcol, gdf_column *rightcol,
                                         gdf_column *l_result, gdf_column *r_result,
                                         gdf_context *ctxt);

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Allocates a buffer and fills it with a repeated value
*
* @Param buffer Address of the buffer to be allocated
* @Param buffer_length Amount of memory to be allocated
* @Param value The value to be filled into the buffer
* @tparam data_type The data type to be used for the buffer
* @tparam size_type The data type used for size calculations
*/
/* ----------------------------------------------------------------------------*/
template <typename data_type,
          typename size_type>
gdf_error allocValueBuffer(data_type ** buffer,
                           const size_type buffer_length,
                           const data_type value) 
{
    RMM_TRY( RMM_ALLOC((void**)buffer, buffer_length*sizeof(data_type), 0) );
    thrust::fill(thrust::device, *buffer, *buffer + buffer_length, value);
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Allocates a buffer and fills it with a sequence
*
* @Param buffer Address of the buffer to be allocated
* @Param buffer_length Amount of memory to be allocated
* @tparam data_type The data type to be used for the buffer
* @tparam size_type The data type used for size calculations
*/
/* ----------------------------------------------------------------------------*/
template <typename data_type,
          typename size_type>
gdf_error allocSequenceBuffer(data_type ** buffer,
                         const size_type buffer_length) 
{
    RMM_TRY( RMM_ALLOC((void**)buffer, buffer_length*sizeof(data_type), 0) );
    thrust::sequence(thrust::device, *buffer, *buffer + buffer_length);
    return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Trivially computes full join of two tables if one of the tables
 are empty
 * 
 * @Param left_size The size of the left table
 * @Param right_size The size of the right table
 * @Param rightcol The right set of columns to join
 * @Param left_result The join computed indices of the left table
 * @Param right_result The join computed indices of the right table
 * @tparam size_type The data type used for size calculations
 * 
 * @Returns GDF_SUCCESS upon succesfull compute, otherwise returns appropriate error code
 */
/* ----------------------------------------------------------------------------*/
template<typename size_type>
gdf_error trivial_full_join(
        const size_type left_size,
        const size_type right_size,
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
    size_type result_size{0};
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
 * @Synopsis  Computes the join operation between two sets of columns
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param left_result The join computed indices of the left table
 * @Param right_result The join computed indices of the right table
 * @Param join_context A structure that determines various run parameters, such as
   whether to perform a hash or sort based join
 * @tparam join_type The type of join to be performed
 * 
 * @Returns GDF_SUCCESS upon succesfull compute, otherwise returns appropriate error code
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type>
gdf_error join_call( int num_cols, gdf_column **leftcol, gdf_column **rightcol,
                     gdf_column *left_result, gdf_column *right_result,
                     gdf_context *join_context)
{

  using size_type = int64_t;

  if( (0 == num_cols) || (nullptr == leftcol) || (nullptr == rightcol))
    return GDF_DATASET_EMPTY;

  if(nullptr == join_context)
    return GDF_INVALID_API_CALL;

  const auto left_col_size = leftcol[0]->size;
  const auto right_col_size = rightcol[0]->size;
  
  // Check that the number of rows does not exceed the maximum
  if(left_col_size >= MAX_JOIN_SIZE) return GDF_COLUMN_SIZE_TOO_BIG;
  if(right_col_size >= MAX_JOIN_SIZE) return GDF_COLUMN_SIZE_TOO_BIG;


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

  // If Inner Join and either table is empty, compute trivial full join
  if( (JoinType::FULL_JOIN == join_type) && 
      ((0 == left_col_size) || (0 == right_col_size)) ){
    return trivial_full_join<size_type>(left_col_size, right_col_size, left_result, right_result);
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
        gdf_error_code =  hash_join<join_type, size_type>(num_cols, leftcol, rightcol, left_result, right_result);
        break;
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == num_cols)
        {
          gdf_error_code =  sort_join<join_type>(leftcol[0], rightcol[0], left_result, right_result, join_context);
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

template <JoinType join_type, typename size_type, typename index_type>
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

    size_type join_size = left_indices->size;
    int left_table_end = num_left_cols - num_cols_to_join;
    int right_table_begin = num_left_cols;

    //create left and right output column data buffers
    for (int i = 0; i < left_table_end; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, lnonjoincol[i]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }
    for (int i = right_table_begin; i < result_num_cols; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, rnonjoincol[i - right_table_begin]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }
    //create joined output column data buffers
    for (int join_index = 0; join_index < num_cols_to_join; ++join_index) {
        int i = left_table_end + join_index;
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, left_cols[left_join_cols[join_index]]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->data), col_width * join_size, 0) ); // TODO: non-default stream?
        RMM_TRY( RMM_ALLOC((void**)&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size), 0) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }

    gdf_error err{GDF_SUCCESS};

    //Construct the left columns
    if (0 != lnonjoincol.size()) {
        gdf_table<size_type> l_i_table(lnonjoincol.size(), lnonjoincol.data());
        gdf_table<size_type> l_table(num_left_cols - num_cols_to_join, result_cols);
        err = l_i_table.gather(static_cast<index_type*>(left_indices->data),
                l_table, join_type != JoinType::INNER_JOIN);
        if (err != GDF_SUCCESS) { return err; }
    }

    //Construct the right columns
    if (0 != rnonjoincol.size()) {
        gdf_table<size_type> r_i_table(rnonjoincol.size(), rnonjoincol.data());
        gdf_table<size_type> r_table(num_right_cols - num_cols_to_join, result_cols + right_table_begin);
        err = r_i_table.gather(static_cast<index_type*>(right_indices->data),
                r_table, join_type != JoinType::INNER_JOIN);
        if (err != GDF_SUCCESS) { return err; }
    }

    //Construct the joined columns
    if (0 != ljoincol.size()) {
        gdf_table<size_type> j_i_table(ljoincol.size(), ljoincol.data());
        gdf_table<size_type> j_table(num_cols_to_join, result_cols + left_table_end);
        //Gather valid rows from the right table
	// TODO: Revisit this, because it probably can be done more efficiently
        if (JoinType::FULL_JOIN == join_type) {
            gdf_table<size_type> j_i_r_table(rjoincol.size(), rjoincol.data());
            err = j_i_r_table.gather(static_cast<index_type*>(right_indices->data),
                    j_table, join_type != JoinType::INNER_JOIN);
            if (err != GDF_SUCCESS) { return err; }
        }
        err = j_i_table.gather(static_cast<index_type*>(left_indices->data),
                j_table, join_type != JoinType::INNER_JOIN);
    }

	POP_RANGE();
    return err;
}

template <JoinType join_type, typename size_type, typename index_type>
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

  // If the inputs are empty, immediately return
  if( (0 == left_cols[0]->size) && (0 == right_cols[0]->size) ){
    return GDF_SUCCESS;
  }

  GDF_REQUIRE(0 != num_cols_to_join, GDF_SUCCESS);

  GDF_REQUIRE(nullptr != left_join_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != right_join_cols, GDF_DATASET_EMPTY);
  GDF_REQUIRE(nullptr != join_context, GDF_INVALID_API_CALL);

  // Determine if requested output is the indices of matching rows, the fully
  // constructed output dataframe, or both
  bool const construct_output_dataframe{nullptr != result_cols};
  bool const return_output_indices{(nullptr != left_indices) and
                                   (nullptr != right_indices)};

  GDF_REQUIRE(construct_output_dataframe or return_output_indices,
              GDF_INVALID_API_CALL);

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
    //If construct_output_dataframe is false then left_index_out or right_index_out
    //was not dynamically allocated.
    if ((!construct_output_dataframe) || (GDF_SUCCESS != join_err)) {
        return join_err;
    }

    gdf_error df_err =
        construct_join_output_df<join_type, size_type, index_type>(
            ljoincol, rjoincol,
            left_cols, num_left_cols, left_join_cols,
            right_cols, num_right_cols, right_join_cols,
            num_cols_to_join, result_num_cols, result_cols,
            left_index_out, right_index_out);

    l_index_temp.reset(nullptr);
    r_index_temp.reset(nullptr);

    CUDA_CHECK_LAST();

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
    return join_call_compute_df<JoinType::LEFT_JOIN, int64_t, output_index_type>(
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
    return join_call_compute_df<JoinType::INNER_JOIN, int64_t, output_index_type>(
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
    return join_call_compute_df<JoinType::FULL_JOIN, int64_t, output_index_type>(
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
