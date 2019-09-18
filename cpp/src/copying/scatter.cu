/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "gather.cuh"
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cudf_utils.h>
#include <cudf/legacy/table.hpp>

#include <copying/gather.hpp>

#include <cudf/types.h>
#include <utilities/bit_util.cuh>
#include <utilities/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <utilities/column_utils.hpp>
#include <bitmask/legacy/bit_mask.cuh>

using bit_mask::bit_mask_t;

namespace cudf {
namespace detail {


template <typename value_type, typename InputIterator>
struct scatter_to_gather {
  scatter_to_gather(InputIterator first, InputIterator last, value_type default_value):
    first(first), last(last), default_value(default_value) {}

  value_type operator()(value_type in) {
    InputIterator found = thrust::find(first, last, in);
    if (found == last) {
      return default_value;
    }
    else {
      return thrust::distance(first, found);
    }
  }

  InputIterator first, last;
  value_type default_value;
};

struct dispatch_map_type {
template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
void operator()(table const* source_table,
		const gdf_column scatter_map,
		table *destination_table) {
  map_type const * typed_scatter_map = static_cast<map_type const*>(scatter_map.data);

  // Turn the scatter_map[] into a gather_map[] and then call gather(...).
  auto gather_map = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      scatter_to_gather<map_type, map_type const*>(
	  typed_scatter_map,
	  typed_scatter_map+source_table->num_rows(),
	  -1));;

  detail::gather<decltype(gather_map)>(source_table, gather_map, destination_table, false, true, true, false);

}

template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
void operator()(table const* source_table,
    const gdf_column scatter_map,
    table *destination_table) {
  CUDF_FAIL("Scatter map must be an integral type.");
}

};

void scatter(table const* source_table, gdf_column const scatter_map, table* destination_table) {
  const gdf_size_type num_source_rows = source_table->num_rows();
  const gdf_size_type num_destination_rows = destination_table->num_rows();
  
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");
  
  if (0 == source_table->num_rows()) {
    return;
  }

  type_dispatcher(scatter_map.dtype, dispatch_map_type{},
		  source_table,
		  scatter_map,
		  destination_table);
}

void scatter(table const* source_table, gdf_index_type const scatter_map[],
            table* destination_table) {
  gdf_column scatter_map_column{};
  gdf_column_view(&scatter_map_column,
		  const_cast<gdf_index_type*>(scatter_map),
		  nullptr,
		  source_table->num_rows(),
		  gdf_dtype_of<gdf_index_type>());
  scatter(source_table, scatter_map_column, destination_table);
}

template<bool mark_true>
__global__ void marking_bitmask_kernel(
    bit_mask_t* destination_mask,
    gdf_size_type num_destination_rows,
    const gdf_index_type scatter_map[],
    gdf_size_type num_scatter_rows
){
  
  gdf_index_type row = threadIdx.x + blockIdx.x * blockDim.x;
 
  while (row < num_scatter_rows) {

    const gdf_index_type output_row = scatter_map[row];

    if(mark_true){
      bit_mask::set_bit_safe(destination_mask, output_row);
    }else{
      bit_mask::clear_bit_safe(destination_mask, output_row);
    }

    row += blockDim.x * gridDim.x;
  }
}

struct scalar_scatterer {
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to scatter from one scalar to a table based
   * on a `scatter_map`.
   *
   * @tparam ColumnType Dispatched type for the column being scattered 
   * @param source The scalar to scatter to
   * @param scatter_map Array of indices that maps the source element to destination
   * elements
   * @param destination_column The column to gather into
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename ColumnType>
  void operator()(gdf_scalar const& source,
                  gdf_index_type const scatter_map[], const gdf_size_type num_scatter_rows,
                  gdf_column* destination_column, cudaStream_t stream = 0) {
    
    const ColumnType source_data {
        *reinterpret_cast<ColumnType const*>(&source.data) };
    ColumnType* destination_data {
        reinterpret_cast<ColumnType*>(destination_column->data) };

    thrust::constant_iterator<ColumnType> const_iter(source_data);
    thrust::scatter(rmm::exec_policy(stream)->on(stream), const_iter,
                     const_iter + num_scatter_rows, scatter_map, 
                     destination_data);
    
    CHECK_STREAM(stream);
  
  }
};

void scalar_scatter(const std::vector<gdf_scalar>& source, 
                    gdf_index_type const scatter_map[],
                    gdf_size_type num_scatter_rows, table* destination_table){
 
  CUDF_EXPECTS(source.size() == (size_t)destination_table->num_columns(),
    "scalar vector and destination table size mismatch.");

  const int n_cols = source.size();

  std::vector<cudf::util::cuda::scoped_stream> v_streams(2*n_cols);

  // data part
  for(int i = 0; i < n_cols; i++){
    CUDF_EXPECTS(source[i].dtype == destination_table->get_column(i)->dtype,
        "source/destination data type mismatch.");
    CUDF_EXPECTS(source[i].dtype != GDF_STRING_CATEGORY,
        "Scalar scatter currently does not support GDF_STRING_CATEGORY.");
    type_dispatcher(source[i].dtype, scalar_scatterer{}, source[i], 
        scatter_map, num_scatter_rows, destination_table->get_column(i), v_streams[i]);
  }

  constexpr int block_size = 256;  
  const int grid_size = cudf::util::cuda::grid_config_1d(num_scatter_rows, block_size).num_blocks;
  
  // bitmask part
  for(int i = 0; i < n_cols; i++){
    gdf_column* dest_col = destination_table->get_column(i);
    if(dest_col->valid){
      bit_mask_t* dest_valid = reinterpret_cast<bit_mask_t*>(dest_col->valid);
      auto bitmask_kernel = source[i].is_valid ?
        marking_bitmask_kernel<true> : marking_bitmask_kernel<false>;
      bitmask_kernel<<<grid_size, block_size, 0, v_streams[i+n_cols]>>>
        (dest_valid, dest_col->size, scatter_map, num_scatter_rows);
      set_null_count(*dest_col);
    }
  }

}

}  // namespace detail

table scatter(table const& source, gdf_column const scatter_map,
	      table const& target) {

  const gdf_size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(cudf::has_nulls(*source.get_column(i)) && !is_nullable(*target.get_column(i))){

      gdf_size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));

      gdf_size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));

    }
  }

  detail::scatter(&source, scatter_map, &output);
  nvcategory_gather_table(output, output);

  return output;

}


table scatter(table const& source, gdf_index_type const scatter_map[], 
    table const& target) {
  
  const gdf_size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(cudf::has_nulls(*source.get_column(i)) && !is_nullable(*target.get_column(i))){
      
      gdf_size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));
      
      gdf_size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));
    
    }
  }

  detail::scatter(&source, scatter_map, &output);
  nvcategory_gather_table(output, output);

  return output;

}

table scatter(std::vector<gdf_scalar> const& source, 
              gdf_index_type const scatter_map[],
              gdf_size_type num_scatter_rows, table const& target){

  const gdf_size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(source[i].is_valid == false && !is_nullable(*target.get_column(i))){
      
      gdf_size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));
    	
      gdf_size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));
    
    }
  }

  detail::scalar_scatter(source, scatter_map, num_scatter_rows, &output);
  
  return output;
}


}  // namespace cudf
