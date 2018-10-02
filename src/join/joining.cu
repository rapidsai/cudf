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

#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <limits>
#include <set>
#include <vector>

#include "joining.h"
#include "../gdf_table.cuh"

using namespace mgpu;

template <typename T>
void dump_mem(const char name[], const mem_t<T> & mem) {

    auto data = from_mem(mem);
    std::cout << name << " = " ;
    for (int i=0; i < data.size(); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "\n";
}


// Size limit due to use of int32 as join output.
// FIXME: upgrade to 64-bit
using output_index_type = int;
constexpr output_index_type MAX_JOIN_SIZE{std::numeric_limits<output_index_type>::max()};

// TODO This macro stuff will go away once Outer join is implemented
#define DEF_JOIN(Fn, T, Joiner)                                             \
gdf_error gdf_##Fn(gdf_column *leftcol, gdf_column *rightcol,               \
                   gdf_column *left_result, gdf_column *right_result) {     \
    using namespace mgpu;                                                   \
    if ( leftcol->dtype != rightcol->dtype) return GDF_UNSUPPORTED_DTYPE;   \
    if ( leftcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;   \
    if ( rightcol->size >= MAX_JOIN_SIZE ) return GDF_COLUMN_SIZE_TOO_BIG;  \
    standard_context_t context;                                             \
    auto output = Joiner((T*)leftcol->data, leftcol->size,                  \
                                (T*)rightcol->data, rightcol->size,         \
                                less_t<T>(), context);                      \
    *left_result = output.first;                                            \
    *right_result = output.second;                                          \
    CUDA_CHECK_LAST();                                                      \
    return GDF_SUCCESS;                                                     \
}

#define DEF_JOIN_GENERIC(Fn)                                                            \
gdf_error gdf_##Fn##_generic(gdf_column *leftcol, gdf_column * rightcol,                \
                             gdf_column *l_result, gdf_column *r_result) {              \
    switch ( leftcol->dtype ){                                                          \
    case GDF_INT8:      return gdf_##Fn##_i8 (leftcol, rightcol, l_result, r_result);   \
    case GDF_INT16:     return gdf_##Fn##_i16(leftcol, rightcol, l_result, r_result);   \
    case GDF_INT32:     return gdf_##Fn##_i32(leftcol, rightcol, l_result, r_result);   \
    case GDF_INT64:     return gdf_##Fn##_i64(leftcol, rightcol, l_result, r_result);   \
    case GDF_FLOAT32:   return gdf_##Fn##_f32(leftcol, rightcol, l_result, r_result);   \
    case GDF_FLOAT64:   return gdf_##Fn##_f64(leftcol, rightcol, l_result, r_result);   \
    case GDF_DATE32:    return gdf_##Fn##_i32(leftcol, rightcol, l_result, r_result);   \
    case GDF_DATE64:    return gdf_##Fn##_i64(leftcol, rightcol, l_result, r_result);   \
    case GDF_TIMESTAMP: return gdf_##Fn##_i64(leftcol, rightcol, l_result, r_result);   \
    default: return GDF_UNSUPPORTED_DTYPE;                                              \
    }                                                                                   \
}

#define DEF_OUTER_JOIN(Fn, T) DEF_JOIN(outer_join_ ## Fn, T, outer_join)
DEF_JOIN_GENERIC(outer_join)
DEF_OUTER_JOIN(i8,  int8_t)
DEF_OUTER_JOIN(i16, int16_t)
DEF_OUTER_JOIN(i32, int32_t)
DEF_OUTER_JOIN(i64, int64_t)
DEF_OUTER_JOIN(f32, int32_t)
DEF_OUTER_JOIN(f64, int64_t)

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Computes the Join result between two tables using the hash-based implementation. 
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param out_result The result of the join operation. The first n/2 elements of the
   output are the left indices, the last n/2 elements of the output are the right indices.
   @tparam join_type The type of join to be performed
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

  standard_context_t context(false);
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
 * @Param out_result The output of the join operation
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
 * @Synopsis  Computes the join operation between two sets of columns
 * 
 * @Param num_cols The number of columns to join
 * @Param leftcol The left set of columns to join
 * @Param rightcol The right set of columns to join
 * @Param out_result The result of the join operation. The output is structured such that
 * the pair (i, i + output_size/2) is the (left, right) index of matching rows.
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

  // check that the columns data are not null, have matching types, 
  // and the same number of rows
  for (int i = 0; i < num_cols; i++) {
    if((right_col_size > 0) && (nullptr == rightcol[i]->data)){
     return GDF_DATASET_EMPTY;
    } 
    if((left_col_size > 0) && (nullptr == leftcol[i]->data)){
     return GDF_DATASET_EMPTY;
    } 
    if(rightcol[i]->dtype != leftcol[i]->dtype) return GDF_JOIN_DTYPE_MISMATCH;
    if(left_col_size != leftcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;
    if(right_col_size != rightcol[i]->size) return GDF_COLUMN_SIZE_MISMATCH;
  }

  gdf_method join_method = join_context->flag_method; 

  switch(join_method)
  {
    case GDF_HASH:
      {
        return hash_join<join_type, int64_t>(num_cols, leftcol, rightcol, left_result, right_result);
      }
    case GDF_SORT:
      {
        // Sort based joins only support single column joins
        if(1 == num_cols)
        {
          return sort_join<join_type>(leftcol[0], rightcol[0], left_result, right_result, join_context);
        }
        else
        {
          return GDF_JOIN_TOO_MANY_COLUMNS;
        }
      }
    default:
      return GDF_UNSUPPORTED_METHOD;
  }

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

    size_t join_size = left_indices->size;
    int left_table_end = num_left_cols - num_cols_to_join;
    int right_table_begin = num_left_cols;

    //create left and right output column data buffers
    for (int i = 0; i < left_table_end; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, lnonjoincol[i]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        CUDA_TRY( cudaMalloc(&(result_cols[i]->data), col_width * join_size) );
        CUDA_TRY( cudaMalloc(&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }
    for (int i = right_table_begin; i < result_num_cols; ++i) {
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, rnonjoincol[i - right_table_begin]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        CUDA_TRY( cudaMalloc(&(result_cols[i]->data), col_width * join_size) );
        CUDA_TRY( cudaMalloc(&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }
    //create joined output column data buffers
    for (int join_index = 0; join_index < num_cols_to_join; ++join_index) {
        int i = left_table_end + join_index;
        gdf_column_view(result_cols[i], nullptr, nullptr, join_size, left_cols[left_join_cols[join_index]]->dtype);
        int col_width; get_column_byte_width(result_cols[i], &col_width);
        CUDA_TRY( cudaMalloc(&(result_cols[i]->data), col_width * join_size) );
        CUDA_TRY( cudaMalloc(&(result_cols[i]->valid), sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
        CUDA_TRY( cudaMemset(result_cols[i]->valid, 0, sizeof(gdf_valid_type)*gdf_get_num_chars_bitmask(join_size)) );
    }

    gdf_table<size_type> l_i_table(lnonjoincol.size(), lnonjoincol.data());
    gdf_table<size_type> r_i_table(rnonjoincol.size(), rnonjoincol.data());
    gdf_table<size_type> j_i_table(ljoincol.size(), ljoincol.data());

    gdf_table<size_type> l_table(num_left_cols - num_cols_to_join, result_cols);
    gdf_table<size_type> r_table(num_right_cols - num_cols_to_join, result_cols + right_table_begin);
    gdf_table<size_type> j_table(num_cols_to_join, result_cols + left_table_end);

    gdf_error err{GDF_SUCCESS};
    err = l_i_table.gather(static_cast<index_type*>(left_indices->data),
            l_table, join_type != JoinType::INNER_JOIN);
    if (err != GDF_SUCCESS) { return err; }
    err = r_i_table.gather(static_cast<index_type*>(right_indices->data),
            r_table, join_type != JoinType::INNER_JOIN);
    if (err != GDF_SUCCESS) { return err; }
    err = j_i_table.gather(static_cast<index_type*>(left_indices->data),
            j_table, join_type != JoinType::INNER_JOIN);
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
    //return error if the inputs are invalid
    if ((left_cols == nullptr)  ||
        (right_cols == nullptr)) { return GDF_DATASET_EMPTY; }

    //check if combined join output is expected
    bool compute_df = (result_cols != nullptr);

    //return error if no output pointers are valid
    if ( ((left_indices == nullptr)||(right_indices == nullptr)) &&
         (!compute_df) ) { return GDF_DATASET_EMPTY; }

    //If index outputs are not requested, create columns to store them
    //for computing combined join output
    gdf_column * left_index_out = left_indices;
    gdf_column * right_index_out = right_indices;

    using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;
    auto gdf_col_deleter = [](gdf_column* col){
        col->size = 0;
        if (col->data)  { cudaFree(col->data);  }
        if (col->valid) { cudaFree(col->valid); }
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
    //If compute_df is false then left_index_out or right_index_out
    //was not dynamically allocated.
    if ((!compute_df) || (GDF_SUCCESS != join_err)) {
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
