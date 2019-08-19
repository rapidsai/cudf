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
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/nvtx/nvtx_utils.h>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <nvstrings/NVCategory.h>
#include <copying/gather.hpp>
#include "joining.h"

#include <limits>
#include <set>
#include <vector>
#include <numeric>
#include <algorithm>

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
 * @throws cudf::logic_error
 * "Dataset is empty" if both left_dataframe and right_dataframe is empty
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

    CUDA_CHECK_LAST();
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the join operation between two sets of columns
 *
 * @throws cudf::logic_error
 * 
 * @param[in] left  Table of left columns to join
 * @param[in] right Table of right  columns to join
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

  PUSH_RANGE("LIBGDF_JOIN", JOIN_COLOR);

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
          CUDF_EXPECTS(false, "Too many columns to join");
        }

        break;
      }
    default:
      CUDF_EXPECTS(false, "Unsupported Method");
  }

  POP_RANGE();
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Computes the resulting joined table
 *
 * @throws cudf::logic_error
 * 
 * @param[in] ljoin  Table of left join columns 
 * @param[in] rjoin  Table of right join columns
 * @param[in] left   Updated left dataframe
 * @param[in] right  Updated right dataframe
 * @param[in] left_on The table containing two columns,
 * first column - indices of left table provided for join
 * second column - indices of right table provided for join
 * @param[in] joining_ind is a vector of pairs of left and right
 * join indcies derived from left_on and right_on. This contains
 * the indices with the same name which evetually result into a 
 * single column.
 * @param[in] left_indices 
 * @param[in] right_indicess  Table contatining right join columns
 * @tparam join_type The type of join to be performed
 * 
 * @returns void
 */
/* ----------------------------------------------------------------------------*/

template <JoinType join_type, typename index_type>
std::pair<cudf::table, cudf::table> construct_join_output_df(
        cudf::table const& ljoin,
        cudf::table const& rjoin,
        cudf::table const& left, 
        cudf::table const& right,
        cudf::table const& left_on,
        std::vector<std::pair<int, int>> const& joining_ind,
        gdf_column * left_indices,
        gdf_column * right_indices) {

    PUSH_RANGE("LIBGDF_JOIN_OUTPUT", JOIN_COLOR);
    //create left and right input table with columns not joined on
    std::vector<int> l_col_ind(left.num_columns());
    std::vector<int> r_col_ind(right.num_columns());
    std::vector<int> left_j_cols (left_on.num_rows());
    std::vector<int> l_joining_ind (joining_ind.size());
    std::vector<int> r_joining_ind (joining_ind.size());
    std::vector<int> l_nonjoin_ind (left.num_columns() - l_joining_ind.size());
    std::vector<int> r_nonjoin_ind (right.num_columns() - r_joining_ind.size());

    if (left_on.num_rows() > 0)
    {
        CUDA_TRY (cudaMemcpy(left_j_cols.data(), left_on.get_column(0)->data, 
                             sizeof(int)*left_on.num_rows() , cudaMemcpyDeviceToHost));
    }

    for (unsigned int i = 0; i < joining_ind.size(); ++i)
    {
        l_joining_ind[i] = joining_ind[i].first;
        r_joining_ind[i] = joining_ind[i].second;
    }
    
    std::vector <int> tmp_l_join_ind = l_joining_ind;
    std::vector <int> tmp_r_join_ind = r_joining_ind;
  
    std::iota(std::begin(l_col_ind), std::end(l_col_ind), 0);
    std::iota(std::begin(r_col_ind), std::end(r_col_ind), 0);
    std::sort(std::begin(tmp_l_join_ind), std::end(tmp_l_join_ind));
    std::sort(std::begin(tmp_r_join_ind), std::end(tmp_r_join_ind));

    // Gathering the indices that are not in join
    if (l_nonjoin_ind.size() > 0)
        std::set_difference(std::cbegin(l_col_ind), std::cend(l_col_ind),
                                 std::cbegin(tmp_l_join_ind), std::cend(tmp_l_join_ind),
                                 std::begin(l_nonjoin_ind));
    
    if (r_nonjoin_ind.size() > 0)
        std::set_difference(std::cbegin(r_col_ind), std::cend(r_col_ind),
                                 std::cbegin(tmp_r_join_ind), std::cend(tmp_r_join_ind),
                                 std::begin(r_nonjoin_ind));

    gdf_size_type join_size = left_indices->size;
    std::vector <gdf_dtype> rdtypes;
    std::vector <gdf_dtype_extra_info> rdtype_infos;

    std::vector<gdf_column*> lnonjoincol;
    std::vector<gdf_column*> rnonjoincol;

    // Gathering all the left table columns not in joining indices 
    for (std::vector<int>::iterator it = l_nonjoin_ind.begin() ; it != l_nonjoin_ind.end(); ++it)
    {
        lnonjoincol.push_back(const_cast<gdf_column*>(left.get_column(*it)));
    }
    // Gathering all the right table columns not in joining indices
    for (std::vector<int>::iterator it = r_nonjoin_ind.begin() ; it != r_nonjoin_ind.end(); ++it){
        rnonjoincol.push_back(const_cast<gdf_column*>(right.get_column(*it)));
        rdtypes.push_back(right.get_column(*it)->dtype);
        rdtype_infos.push_back(right.get_column(*it)->dtype_info);
    }
  
    cudf::table result_left(join_size, cudf::column_dtypes(left), cudf::column_dtype_infos(left), true);
    cudf::table result_right(join_size, rdtypes, rdtype_infos, true);
    
    std::vector<gdf_column*> result_lnonjoincol;
    std::vector<gdf_column*> result_rnonjoincol;
    std::vector<gdf_column*> result_joincol;

    // Gather the left non-join col of result
    for (std::vector<int>::iterator it = l_nonjoin_ind.begin(); it != l_nonjoin_ind.end(); ++it)
    {
        result_lnonjoincol.push_back(result_left.get_column(*it));
    }
        
    // Gather join-col of result 
    for (unsigned int i=0; i < joining_ind.size(); ++i)
    {
        result_joincol.push_back(result_left.get_column(l_joining_ind[i]));
    }
    
    // Gather the right non-join col of result
    for (int i=0; i < result_right.num_columns(); ++i)
    {
        result_rnonjoincol.push_back(result_right.get_column(i));
    }
 
    bool const check_bounds{ join_type != JoinType::INNER_JOIN };

    // Construct the left columns
    if (0 != lnonjoincol.size()) {
      cudf::table left_source_table(lnonjoincol);
      cudf::table left_destination_table(result_lnonjoincol);

      cudf::detail::gather(&left_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &left_destination_table, check_bounds);

      gdf_error update_err = nvcategory_gather_table(left_source_table,left_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table throwing a GDF error");
    }
    
    // Construct the right columns
    if (0 != rnonjoincol.size()) {
      cudf::table right_source_table(rnonjoincol);
      cudf::table right_destination_table(result_rnonjoincol);

      cudf::detail::gather(&right_source_table,
                           static_cast<index_type const *>(right_indices->data),
                           &right_destination_table, check_bounds);
      gdf_error update_err = nvcategory_gather_table(right_source_table,right_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table throwing a GDF error");
    }

    // Construct the joined columns
    if (0 != ljoin.num_columns() && joining_ind.size() > 0) {

      std::vector <gdf_column *> l_join;
      std::vector <gdf_column *> r_join;
      // Gather the columns which join into single column from joined columns
      for (unsigned int join_ind = 0; join_ind < joining_ind.size(); ++join_ind)
      {
          std::vector<int>::iterator itr = std::find(left_j_cols.begin(), left_j_cols.end(),
               l_joining_ind[join_ind]);

          int index = std::distance(left_j_cols.begin(), itr);

          l_join.push_back(const_cast<gdf_column*>(ljoin.get_column(index)));

          if (JoinType::FULL_JOIN == join_type)
          {
              r_join.push_back(const_cast<gdf_column*>(rjoin.get_column(index)));
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
      }

      cudf::detail::gather(&join_source_table,
                           static_cast<index_type const *>(left_indices->data),
                           &join_destination_table, check_bounds);
      gdf_error update_err = nvcategory_gather_table(join_source_table,join_destination_table);
      CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error");
    }
     
    CHECK_STREAM(0);
    POP_RANGE();
    return std::pair<cudf::table, cudf::table>(result_left, result_right);
}

template <JoinType join_type, typename index_type>
std::pair<cudf::table, cudf::table> join_call_compute_df(
                         cudf::table const& left, 
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind, 
                         gdf_context *join_context) {
 
  if (0 == left_on.num_rows() || 0 == right_on.num_rows() || left_on.num_rows() != right_on.num_rows())
  {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(left), cudf::empty_like(right));
  }
  CUDF_EXPECTS (0 != left.num_columns(), "Left table is empty");
  CUDF_EXPECTS (0 != right.num_columns(), "Right table is empty");
  CUDF_EXPECTS (nullptr != join_context, "Join context is invalid");

  std::vector<int> left_on_ind (left_on.num_rows());
  std::vector<int> right_on_ind (right_on.num_rows());
  std::vector<int> r_joining_ind (joining_ind.size());

  if (left_on.num_rows() > 0)
  {
      CUDA_TRY (cudaMemcpy((void *)left_on_ind.data(), (void *)left_on.get_column(0)->data, 
                           sizeof(int)*left_on.num_rows(), cudaMemcpyDeviceToHost));
  }
  if (right_on.num_rows() > 0)
  {
      CUDA_TRY (cudaMemcpy((void *)right_on_ind.data(), (void *)right_on.get_column(0)->data, 
                           sizeof(int)*right_on.num_rows(), cudaMemcpyDeviceToHost));
  }

  for (unsigned int i = 0; i < joining_ind.size(); ++i)
  {
      r_joining_ind [i] = joining_ind[i].second;
  }

  std::vector <gdf_column*> tmp_right_cols;
  std::vector<int> r_col_ind(right.num_columns());
  std::iota(std::begin(r_col_ind), std::end(r_col_ind), 0);
  std::sort(std::begin(r_joining_ind), std::end(r_joining_ind));
  std::vector <int> r_nonjoin_ind (right.num_columns() - r_joining_ind.size());

  // Gathering the indices that are not in join 
  std::set_difference(std::cbegin(r_col_ind), std::cend(r_col_ind),
                      std::cbegin(r_joining_ind), std::cend(r_joining_ind),
                      std::begin(r_nonjoin_ind));

  for (std::vector<int>::iterator it = r_nonjoin_ind.begin() ; it != r_nonjoin_ind.end(); ++it){
      tmp_right_cols.push_back(const_cast<gdf_column *> (right.get_column(*it)));
  }

  cudf::table tmp_right_table = (tmp_right_cols.size()>0)? cudf::table (tmp_right_cols) : cudf::table{};

  CUDF_EXPECTS(std::none_of(std::cbegin(left), std::cend(left), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported left column dtype");
  CUDF_EXPECTS(std::none_of(std::cbegin(right), std::cend(right), [](auto col) { return col->dtype == GDF_invalid; }), "Unsupported right column dtype");

  // Even though the resulting table might be empty, but the column should match the expected dtypes and other necessary information
  // So, there is a possibility that there will be lesser number of right columns, so the tmp_right_table.
  // If the inputs are empty, immediately return
  if ((0 == left.num_rows()) && (0 == right.num_rows())) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(left), cudf::empty_like(tmp_right_table));
  }

  // If left join and the left table is empty, return immediately
  if ((JoinType::LEFT_JOIN == join_type) && (0 == left.num_rows())) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(left), cudf::empty_like(tmp_right_table));
  }

  // If Inner Join and either table is empty, return immediately
  if ((JoinType::INNER_JOIN == join_type) &&
      ((0 == left.num_rows()) || (0 == right.num_rows()))) {
      return std::pair <cudf::table, cudf::table>(cudf::empty_like(left), cudf::empty_like(tmp_right_table));
  }

  //if the inputs are nvcategory we need to make the dictionaries comparable
  bool at_least_one_category_column = false;
  for(int join_column_index = 0; join_column_index < left_on.num_rows(); join_column_index++){
    at_least_one_category_column |= left.get_column(left_on_ind[join_column_index])->dtype == GDF_STRING_CATEGORY;
  }
  
  std::vector<gdf_column*> new_left_cols;
  std::vector<gdf_column*> new_right_cols;

  for (int i = 0; i < left.num_columns(); i++)
      new_left_cols.push_back (const_cast<gdf_column*>(left.get_column(i)));
  for (int i = 0; i < right.num_columns(); i++)
      new_right_cols.push_back (const_cast<gdf_column*>(right.get_column(i)));

  std::vector<gdf_column *> temp_columns_to_free;
  if(at_least_one_category_column){
    for(int join_column_index = 0; join_column_index < left_on.num_rows(); join_column_index++){
      if(left.get_column(left_on_ind[join_column_index])->dtype == GDF_STRING_CATEGORY){
        CUDF_EXPECTS(right.get_column(right_on_ind[join_column_index])->dtype == GDF_STRING_CATEGORY, "GDF type mismatch");

        gdf_column * left_original_column = new_left_cols[left_on_ind[join_column_index]];
        gdf_column * right_original_column = new_right_cols[right_on_ind[join_column_index]];

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

        new_left_cols[left_on_ind[join_column_index]] = new_join_columns[0];
        new_right_cols[right_on_ind[join_column_index]] = new_join_columns[1];
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

  gdf_col_pointer l_index_temp, r_index_temp;
  gdf_column *left_index_out = nullptr;
  gdf_column *right_index_out = nullptr;

  if (nullptr != out_ind && out_ind->num_columns () > 0)
  {
      left_index_out = const_cast <gdf_column*>(out_ind->get_column(0));
      right_index_out = const_cast <gdf_column*>(out_ind->get_column(1));
  }
  else
  {
      l_index_temp = {new gdf_column{}, gdf_col_deleter};
      left_index_out = l_index_temp.get();

      r_index_temp = {new gdf_column{}, gdf_col_deleter};
      right_index_out = r_index_temp.get();
  }

  //get column pointers to join on
  std::vector<gdf_column*> ljoincol;
  std::vector<gdf_column*> rjoincol;
  for (int i = 0; i < left_on.num_rows(); ++i) {
      ljoincol.push_back(updated_left_table.get_column(left_on_ind[i]));
      rjoincol.push_back(updated_right_table.get_column(right_on_ind[i]));
  }
  cudf::table ljoin_ind_table(ljoincol);
  cudf::table rjoin_ind_table(rjoincol);
  join_call<join_type>(ljoin_ind_table, rjoin_ind_table,
            left_index_out, right_index_out,
            join_context);

  std::pair<cudf::table, cudf::table> result =
      construct_join_output_df<join_type, index_type>(
          ljoin_ind_table, rjoin_ind_table,
          updated_left_table, updated_right_table, 
          left_on, joining_ind, 
          left_index_out, right_index_out);
  l_index_temp.reset(nullptr);
  r_index_temp.reset(nullptr);

  //freeing up the temp column used to synch categories between columns
  for(unsigned int column_to_free = 0; column_to_free < temp_columns_to_free.size(); column_to_free++){
      gdf_column_free(temp_columns_to_free[column_to_free]);
      delete temp_columns_to_free[column_to_free];
  }

  CHECK_STREAM(0);
    
  return result;
}

std::pair<cudf::table, cudf::table> left_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::LEFT_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     joining_ind,
                     out_ind,
                     join_context);
}

std::pair<cudf::table, cudf::table> inner_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::INNER_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     joining_ind,
                     out_ind,
                     join_context);
}

std::pair<cudf::table, cudf::table> full_join(
                         cudf::table const& left,
                         cudf::table const& right,
                         cudf::table const& left_on,
                         cudf::table const& right_on,
                         std::vector<std::pair<int, int>> const& joining_ind,
                         cudf::table *out_ind,
                         gdf_context *join_context) {
    return join_call_compute_df<JoinType::FULL_JOIN, output_index_type>(
                     left,
                     right,
                     left_on,
                     right_on,
                     joining_ind,
                     out_ind,
                     join_context);
}
}
