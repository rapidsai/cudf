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

#ifndef GROUPBY_COMPUTE_API_H
#define GROUPBY_COMPUTE_API_H


#include <hash/managed.cuh>
#include "hash_groupby_kernels.cuh"
#include <table/device_table.cuh>
#include <copying/gather.hpp>
#include "rmm/thrust_rmm_allocator.h"
#include "types.hpp"
#include <hash/helper_functions.cuh>

#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <algorithm>

constexpr unsigned int THREAD_BLOCK_SIZE{256};

/* --------------------------------------------------------------------------*/
/**
 * @brief  This functor is used inside the hash table's insert function to 
 * compare equality between two keys in the hash table. 
 * 
 * If comparing a key to the map's unused key, simply performs the 
 * default key comparison defined in the map class. 
 *
 * Otherwise, the hash table keys refer to row indices in device_tables and the 
 * functor checks for equality between the two rows.
 */
/* ----------------------------------------------------------------------------*/
template <typename map_type>
struct row_comparator
{
  using key_type = typename map_type::key_type;
  using map_key_comparator = typename map_type::key_equal;

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Constructs a row_comparator functor to check for equality between
   * keys in the hash table.
   * 
   * @param map The hash table
   * @param l_table The left device_table
   * @param r_table The right device_table
   */
  /* ----------------------------------------------------------------------------*/
  row_comparator(map_type const & map,
                 device_table l_table,
                 device_table r_table) 
                : the_map{map}, 
                  left_table{l_table}, 
                  right_table{r_table},
                  unused_key{map.get_unused_key()}
  {
  
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Used in the hash table's insert function to check for equality between
   * two keys. Two cases are possible:
   *
   * 1. If left_index OR right_index is equal to the map's unused_key, then the functor
   * is being used to compare against an empty key. In this case, perform the default
   * key comparison defined in the map class.
   *
   * 2. Else, the functor is being used to compare two rows of device_tables. In this case,
   * the device_table rows_equal function is used to check if the two rows are equal.
   * 
   * @param left_index The left table index to compare
   * @param right_index The right table index to compare
   * 
   * @returns   
   */
  /* ----------------------------------------------------------------------------*/
  __device__ bool operator()(key_type const & left_index, 
                             key_type const & right_index) const
  {
    // The unused key is not a valid row index in the device_tables.
    // Therefore, if comparing against the unused key, use the map's default
    // comparison function
    if((unused_key == left_index) || (unused_key == right_index))
      return default_comparator(left_index, right_index);

    // Check for equality between the two rows of the two tables
    return rows_equal(left_table, left_index, right_table, right_index);
  }

  const map_key_comparator default_comparator{};
  const key_type unused_key;
  map_type const & the_map;
  device_table left_table;
  device_table right_table;
};

/* --------------------------------------------------------------------------*/
/** 
* @brief Performs the groupby operation for an arbtirary number of groupby columns and
* and a single aggregation column.
* 
* @param[in] input_keys The set of columns to groupby
* @param[in] in_aggregation_column The column to perform the aggregation on. These act as the hash table values
* @param[out] output_keys Preallocated buffer(s) for the groupby column result. This will hold a single
* entry for every unique row in the input table.
* @param[out] out_aggregation_column Preallocated output buffer for the resultant aggregation column that 
*                                     corresponds to the out_groupby_column where entry 'i' is the aggregation 
*                                     for the group out_groupby_column[i] 
* @param out_size The size of the output
* @param aggregation_op The aggregation operation to perform 
* @param sort_result Flag to optionally sort the output table
* 
* @returns   
*/
/* ----------------------------------------------------------------------------*/
template <typename aggregation_type, typename aggregation_operation>
gdf_error GroupbyHash(cudf::table const& input_keys,
                      const aggregation_type* const in_aggregation_column,
                      cudf::table& output_keys,
                      aggregation_type* out_aggregation_column,
                      gdf_size_type* out_size,
                      aggregation_operation aggregation_op,
                      bool sort_result = false) {
  const gdf_size_type input_num_rows = input_keys.num_rows();

  // The map will store (row index, aggregation value)
  // Where row index is the row number of the first row to be successfully inserted
  // for a given unique row
  using map_type = concurrent_unordered_map<gdf_size_type, 
                                            aggregation_type, 
                                            std::numeric_limits<gdf_size_type>::max(), 
                                            default_hash<gdf_size_type>, 
                                            equal_to<gdf_size_type>,
                                            legacy_allocator<thrust::pair<gdf_size_type, aggregation_type> > >;

  // The hash table occupancy and the input size determines the size of the hash table
  const size_t hash_table_size{ compute_hash_table_size(input_num_rows) };
  
  // Initialize the hash table with the aggregation operation functor's identity value
  std::unique_ptr<map_type> the_map(new map_type(hash_table_size, aggregation_operation::IDENTITY));

  const dim3 build_grid_size ((input_num_rows + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);
  const dim3 block_size (THREAD_BLOCK_SIZE, 1, 1);

  CUDA_TRY(cudaGetLastError());


  auto d_input_keys = device_table::create(input_keys);

  // Inserts (i, aggregation_column[i]) as a key-value pair into the
  // hash table. When a given key already exists in the table, the aggregation operation
  // is computed between the new and existing value, and the result is stored back.
  build_aggregation_table<<<build_grid_size, block_size>>>(the_map.get(), 
                                                           *d_input_keys, 
                                                           in_aggregation_column,
                                                           input_num_rows,
                                                           aggregation_op,
                                                           row_comparator<map_type>(*the_map, *d_input_keys, *d_input_keys));
  CUDA_TRY(cudaGetLastError());


  auto d_output_keys = device_table::create(output_keys);

  // Used by threads to coordinate where to write their results
  gdf_size_type * global_write_index{nullptr};
  RMM_TRY(RMM_ALLOC((void**)&global_write_index, sizeof(gdf_size_type), 0)); // TODO: non-default stream?
  CUDA_TRY(cudaMemset(global_write_index, 0, sizeof(gdf_size_type)));
  const dim3 extract_grid_size ((hash_table_size + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

  // Extracts every non-empty key and value into separate contiguous arrays,
  // which provides the result of the groupby operation
  extract_groupby_result<<<extract_grid_size, block_size>>>(the_map.get(),
                                                            hash_table_size,
                                                            *d_output_keys,
                                                            *d_input_keys,
                                                            out_aggregation_column,
                                                            global_write_index);
 
  CUDA_TRY(cudaGetLastError());

  // At the end of the extraction kernel, the global write index will be equal to
  // the size of the output. Update the output size.
  CUDA_TRY( cudaMemcpy(out_size, global_write_index, sizeof(gdf_size_type), cudaMemcpyDeviceToHost) );
  RMM_TRY( RMM_FREE(global_write_index, 0) );

  // Update the size of the output key columns
  std::for_each(output_keys.begin(), output_keys.end(),
                [out_size](gdf_column* col) { col->size = *out_size; });


  // Optionally sort the groupby/aggregation result columns
  if(true == sort_result) {

      rmm::device_vector<gdf_index_type> sorted_indices(*out_size);
      thrust::sequence(rmm::exec_policy()->on(0), sorted_indices.begin(), sorted_indices.end());

      gdf_column sorted_indices_col;
      gdf_error status = gdf_column_view(&sorted_indices_col, sorted_indices.data().get(), 
                            nullptr, *out_size, GDF_INT32);
      if (status != GDF_SUCCESS)
        return status;

      status = gdf_order_by(output_keys.begin(), nullptr,
                            output_keys.num_columns(), &sorted_indices_col, 0);

      if (status != GDF_SUCCESS)
        return status;

      // Reorder table according to indices from order_by
      cudf::detail::gather(&output_keys, sorted_indices.data().get(),
                           &output_keys);

      rmm::device_vector<aggregation_type> temporary_aggregation_buffer(*out_size);
      thrust::gather(rmm::exec_policy()->on(0),
               sorted_indices.begin(), sorted_indices.end(),
               out_aggregation_column,
               temporary_aggregation_buffer.begin());
      thrust::copy(rmm::exec_policy()->on(0),
                   temporary_aggregation_buffer.begin(),
                   temporary_aggregation_buffer.end(), out_aggregation_column);
  }

  return GDF_SUCCESS;
}
#endif
