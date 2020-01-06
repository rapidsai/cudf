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
#include "scatter.hpp"
#include "gather.cuh"
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/filling.hpp>
#include <cudf/cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/legacy/cudf_utils.h>
#include <cudf/legacy/table.hpp>

#include <copying/legacy/gather.hpp>

#include <cudf/types.h>
#include <utilities/legacy/bit_util.cuh>
#include <utilities/legacy/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <utilities/legacy/column_utils.hpp>
#include <bitmask/legacy/bit_mask.cuh>
#include <reductions/legacy/reduction_functions.cuh>
#include <stream_compaction/legacy/copy_if.cuh>



using bit_mask::bit_mask_t;


namespace cudf {
namespace detail {

template <typename index_type, typename scatter_map_type>
__global__ void invert_map(index_type gather_map[], const cudf::size_type destination_rows,
    scatter_map_type const scatter_map, const cudf::size_type source_rows){
  index_type tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < source_rows){
    index_type destination_row = *(scatter_map + tid);
    if(destination_row < destination_rows){
      gather_map[destination_row] = tid;
    }
  }
}

struct dispatch_map_type {
template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value>* = nullptr>
void operator()(table const *source_table, gdf_column const& scatter_map,
    table *destination_table, bool check_bounds, bool allow_negative_indices) {

  map_type const * typed_scatter_map = static_cast<map_type const*>(scatter_map.data);

  if (check_bounds) {
    cudf::size_type begin = (allow_negative_indices) ? -destination_table->num_rows() : 0;
    CUDF_EXPECTS(
	source_table->num_rows() == thrust::count_if(
	    rmm::exec_policy()->on(0),
	    typed_scatter_map,
	    typed_scatter_map + source_table->num_rows(),
	    bounds_checker<map_type>{begin, destination_table->num_rows()}),
	"Index out of bounds.");
  }

  gdf_column gather_map = cudf::allocate_like(scatter_map, destination_table->num_rows());

  map_type default_value = -1;
  gdf_scalar fill_value;
  fill_value.dtype = gdf_dtype_of<map_type>();
  fill_value.is_valid = true;

  switch (gdf_dtype_of<map_type>()) {
  case GDF_INT8:
    fill_value.data.si08 = default_value;
    break;
  case GDF_INT16:
    fill_value.data.si16 = default_value;
    break;
  case GDF_INT32:
    fill_value.data.si32 = default_value;
    break;
  case GDF_INT64:
    fill_value.data.si64 = default_value;
    break;
  default:
    CUDF_FAIL("Invalid scatter map type");
  }
  
  cudf::fill(&gather_map, fill_value, 0, gather_map.size);

  // Configure grid for data kernel launch
  auto grid_config = cudf::util::cuda::grid_config_1d(source_table->num_rows(), 256);
  
  if (allow_negative_indices) {
    invert_map<<<grid_config.num_blocks, grid_config.num_threads_per_block>>>(
	static_cast<map_type*>(gather_map.data),
	destination_table->num_rows(),
	thrust::make_transform_iterator(
	    typed_scatter_map,
	    index_converter<map_type,index_conversion::NEGATIVE_TO_POSITIVE>{destination_table->num_rows()}),
	source_table->num_rows());
  }
  else {
    invert_map<<<grid_config.num_blocks, grid_config.num_threads_per_block>>>(
	static_cast<map_type*>(gather_map.data),
	destination_table->num_rows(),
	thrust::make_transform_iterator(
	    typed_scatter_map,
	    index_converter<map_type>{destination_table->num_rows()}),
	source_table->num_rows());
  }

  // We want to ignore out of bounds indices for scatter since it is possible that
  // some elements of the destination column are not modified. 
  detail::gather(source_table, gather_map, destination_table, false, true, true);

  gdf_column_free(&gather_map);

}
      

template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value>* = nullptr>
void operator()(table const *source_table, gdf_column const& scatter_map,
    table *destination_table, bool check_bounds, bool allow_negative_indices) {
  CUDF_FAIL("Scatter map must be an integral type.");
}

};

void scatter(table const* source_table, gdf_column const& scatter_map,
    table* destination_table, bool check_bounds, bool allow_negative_indices) {
  
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");
  
  if (0 == source_table->num_rows()) {
    return;
  }

  type_dispatcher(scatter_map.dtype, dispatch_map_type{},
      source_table,
      scatter_map,
      destination_table,
      check_bounds,
      true);

}

void scatter(table const* source_table, cudf::size_type const scatter_map[],
    table* destination_table, bool check_bounds, bool allow_negative_indices) {
  gdf_column scatter_map_column{};
  gdf_column_view(&scatter_map_column,
		  const_cast<cudf::size_type*>(scatter_map),
		  nullptr,
		  source_table->num_rows(),
		  gdf_dtype_of<cudf::size_type>());
  detail::scatter(source_table, scatter_map_column, destination_table, check_bounds, allow_negative_indices);
}

template<bool mark_true>
__global__ void marking_bitmask_kernel(
    bit_mask_t* destination_mask,
    cudf::size_type num_destination_rows,
    const cudf::size_type scatter_map[],
    cudf::size_type num_scatter_rows
){
  
  cudf::size_type row = threadIdx.x + blockIdx.x * blockDim.x;
 
  while (row < num_scatter_rows) {

    const cudf::size_type output_row = scatter_map[row];

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
                  cudf::size_type const scatter_map[], const cudf::size_type num_scatter_rows,
                  gdf_column* destination_column, cudaStream_t stream = 0) {
    
    const ColumnType source_data {
        *reinterpret_cast<ColumnType const*>(&source.data) };
    ColumnType* destination_data {
        reinterpret_cast<ColumnType*>(destination_column->data) };

    thrust::constant_iterator<ColumnType> const_iter(source_data);
    thrust::scatter(rmm::exec_policy(stream)->on(stream), const_iter,
                     const_iter + num_scatter_rows, scatter_map, 
                     destination_data);
    
    CHECK_CUDA(stream);
  
  }
};

void scalar_scatter(const std::vector<gdf_scalar>& source, 
                    cudf::size_type const scatter_map[],
                    cudf::size_type num_scatter_rows, table* destination_table){
 
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

inline bool validate_scatter_map(gdf_column const& scatter_map,
                          cudf::table const& input) {
  CUDF_EXPECTS(scatter_map.dtype == GDF_INT32,
      "scatter_map is not GDF_INT32 column.");
  CUDF_EXPECTS(not cudf::has_nulls(scatter_map),
      "Scatter map cannot contain null elements.");
  CUDF_EXPECTS(scatter_map.size == input.num_rows(),
      "scatter_map length is not equal to number of rows in input table.");
  if (scatter_map.size == 0 ||
      input.num_columns() == 0 ||
      input.num_rows() == 0)
    return false;
  return true;
}

std::vector<cudf::table>
ordered_scatter_to_tables(cudf::table const& input,
                          cudf::size_type const* scatter_array,
                          cudf::size_type num_groups) {
  std::vector<cudf::table> output_tables;
  output_tables.reserve(num_groups);
  for (cudf::size_type groupid = 0; groupid < num_groups; groupid++) {
    output_tables.push_back(
        detail::copy_if(input,
          [scatter_array, groupid] __device__ (cudf::size_type row)
          { return groupid==scatter_array[row];
          }));
  }
  return output_tables;
}

}  // namespace detail

table scatter(table const& source, gdf_column const& scatter_map,
    table const& target, bool check_bounds) {

  const cudf::size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(cudf::has_nulls(*source.get_column(i)) && !is_nullable(*target.get_column(i))){

      cudf::size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));

      cudf::size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));

    }
  }

  detail::scatter(&source, scatter_map, &output, check_bounds, true);
  nvcategory_gather_table(output, output);

  return output;

}


table scatter(table const& source, cudf::size_type const scatter_map[], 
    table const& target, bool check_bounds) {
  
  const cudf::size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(cudf::has_nulls(*source.get_column(i)) && !is_nullable(*target.get_column(i))){
      
      cudf::size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));
      
      cudf::size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));
    
    }
  }

  detail::scatter(&source, scatter_map, &output, check_bounds, true);
  nvcategory_gather_table(output, output);

  return output;

}

table scatter(std::vector<gdf_scalar> const& source, 
              cudf::size_type const scatter_map[],
              cudf::size_type num_scatter_rows, table const& target){

  const cudf::size_type n_cols = target.num_columns();

  table output = copy(target);
  for(int i = 0; i < n_cols; ++i){
    // Allocate bitmask for each column
    if(source[i].is_valid == false && !is_nullable(*target.get_column(i))){
      
      cudf::size_type valid_size = gdf_valid_allocation_size(target.get_column(i)->size);
      RMM_TRY(RMM_ALLOC(&output.get_column(i)->valid, valid_size, 0));
    	
      cudf::size_type valid_size_set = gdf_num_bitmask_elements(target.get_column(i)->size);
      CUDA_TRY(cudaMemset(output.get_column(i)->valid, 0xff, valid_size_set));
    
    }
  }

  detail::scalar_scatter(source, scatter_map, num_scatter_rows, &output);
  
  return output;
}

std::vector<cudf::table>
scatter_to_tables(cudf::table const& input, gdf_column const& scatter_map) {
  if(not detail::validate_scatter_map(scatter_map, input)) 
    return std::vector<cudf::table>();

  cudf::size_type* scatter_array =
    static_cast<cudf::size_type*>(scatter_map.data);

  gdf_scalar max_elem = cudf::reduction::max(scatter_map, scatter_map.dtype);
  cudf::size_type num_groups = max_elem.data.si32 + 1;
  return detail::ordered_scatter_to_tables(input,
                    scatter_array,
                    num_groups);
}

}  // namespace cudf
