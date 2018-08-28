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

#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "../../hashmap/managed.cuh"
#include "groupby_kernels.cuh"
#include "../../gdf_table.cuh"

// TODO: replace this with CUDA_TRY and propagate the error
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call ) 									   \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus ) {                                                             \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
        exit(1);										   \
    }												   \
}
#endif

// The occupancy of the hash table determines it's capacity. A value of 50 implies
// 50% occupancy, i.e., hash_table_size == 2 * input_size
constexpr unsigned int DEFAULT_HASH_TABLE_OCCUPANCY{50};

constexpr unsigned int THREAD_BLOCK_SIZE{256};

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This functor is used inside the hash table's insert function to 
 * compare equality between two keys in the hash table. 
 * 
 * If comparing a key to the map's unused key, simply performs the 
 * default key comparison defined in the map class. 
 *
 * Otherwise, the hash table keys refer to row indices in gdf_tables and the 
 * functor checks for equality between the two rows.
 */
/* ----------------------------------------------------------------------------*/
template <typename map_type,
          typename size_type>
struct row_comparator : public managed
{
  using key_type = typename map_type::key_type;
  using map_key_comparator = typename map_type::key_equal;

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Constructs a row_comparator functor to check for equality between
   * keys in the hash table.
   * 
   * @Param map The hash table
   * @Param l_table The left gdf_table
   * @Param r_table The right gdf_table
   */
  /* ----------------------------------------------------------------------------*/
  row_comparator(map_type const & map,
                 gdf_table<size_type> const & l_table,
                 gdf_table<size_type> const & r_table) 
                : the_map{map}, 
                  left_table{l_table}, 
                  right_table{r_table},
                  unused_key{map.get_unused_key()},
                  default_comparator{map_key_comparator()}
  {
  
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis Used in the hash table's insert function to check for equality between
   * two keys. Two cases are possible:
   *
   * 1. If left_index OR right_index is equal to the map's unused_key, then the functor
   * is being used to compare against an empty key. In this case, perform the default
   * key comparison defined in the map class.
   *
   * 2. Else, the functor is being used to compare two rows of gdf_tables. In this case,
   * the gdf_table rows_equal function is used to check if the two rows are equal.
   * 
   * @Param left_index The left table index to compare
   * @Param right_index The right table index to compare
   * 
   * @Returns   
   */
  /* ----------------------------------------------------------------------------*/
  __device__ bool operator()(key_type const & left_index, 
                             key_type const & right_index) const
  {

    // The unused key is not a valid row index in the gdf_tables.
    // Therefore, if comparing against the unused key, use the map's default
    // comparison function
    if((unused_key == left_index) || (unused_key == right_index))
      return default_comparator(left_index, right_index);

    // Check for equality between the two rows of the two tables
    return left_table.rows_equal(right_table, left_index, right_index);
  }

  const map_key_comparator default_comparator;
  const key_type unused_key;
  map_type const & the_map;
  gdf_table<size_type> const & left_table;
  gdf_table<size_type> const & right_table;
};

/* --------------------------------------------------------------------------*/
/** 
* @Synopsis Performs the groupby operation for an arbtirary number of groupby columns and
* and a single aggregation column.
* 
* @Param[in] groupby_input_table The set of columns to groupby
* @Param[in] in_aggregation_column The column to perform the aggregation on. These act as the hash table values
* @Param[out] groupby_output_table Preallocated buffer(s) for the groupby column result. This will hold a single
* entry for every unique row in the input table.
* @Param[out] out_aggregation_column Preallocated output buffer for the resultant aggregation column that 
*                                     corresponds to the out_groupby_column where entry 'i' is the aggregation 
*                                     for the group out_groupby_column[i] 
* @Param out_size The size of the output
* @Param aggregation_op The aggregation operation to perform 
* @Param sort_result Flag to optionally sort the output table
* 
* @Returns   
*/
/* ----------------------------------------------------------------------------*/
template< typename aggregation_type,
          typename size_type,
          typename aggregation_operation>
cudaError_t GroupbyHash(gdf_table<size_type> const & groupby_input_table,
                        const aggregation_type * const in_aggregation_column,
                        gdf_table<size_type> & groupby_output_table,
                        aggregation_type * out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op,
                        bool sort_result = false)
{
  cudaError_t error{cudaSuccess};

  const size_type input_num_rows = groupby_input_table.get_column_length();

  // The map will store (row index, aggregation value)
  // Where row index is the row number of the first row to be successfully inserted
  // for a given unique row
  using map_type = concurrent_unordered_map<size_type, 
                                            aggregation_type, 
                                            std::numeric_limits<size_type>::max(), 
                                            default_hash<size_type>, 
                                            equal_to<size_type>,
                                            legacy_allocator<thrust::pair<size_type, aggregation_type> > >;

  // The hash table occupancy and the input size determines the size of the hash table
  // e.g., for a 50% occupancy, the size of the hash table is twice that of the input
  const size_type hash_table_size = static_cast<size_type>((static_cast<uint64_t>(input_num_rows) * 100 / DEFAULT_HASH_TABLE_OCCUPANCY));
  
  // Initialize the hash table with the aggregation operation functor's identity value
  std::unique_ptr<map_type> the_map(new map_type(hash_table_size, aggregation_operation::IDENTITY));

  // Functor that will be used by the hash table's insert function to check for equality 
  // between two rows of the groupby_input_table
  std::unique_ptr<row_comparator<map_type,size_type>> the_comparator{new row_comparator<map_type, size_type>(*the_map, groupby_input_table, groupby_input_table)};

  const dim3 build_grid_size ((input_num_rows + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);
  const dim3 block_size (THREAD_BLOCK_SIZE, 1, 1);

  CUDA_RT_CALL(cudaGetLastError());

  // Inserts (i, aggregation_column[i]) as a key-value pair into the
  // hash table. When a given key already exists in the table, the aggregation operation
  // is computed between the new and existing value, and the result is stored back.
  build_aggregation_table<<<build_grid_size, block_size>>>(the_map.get(), 
                                                           groupby_input_table, 
                                                           in_aggregation_column,
                                                           input_num_rows,
                                                           aggregation_op,
                                                           *the_comparator);
  CUDA_RT_CALL(cudaGetLastError());

  // Used by threads to coordinate where to write their results
  size_type * global_write_index{nullptr};
  CUDA_RT_CALL(cudaMalloc(&global_write_index, sizeof(size_type)));
  CUDA_RT_CALL(cudaMemset(global_write_index, 0, sizeof(size_type)));

  const dim3 extract_grid_size ((hash_table_size + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

  // Extracts every non-empty key and value into separate contiguous arrays,
  // which provides the result of the groupby operation
  extract_groupby_result<<<extract_grid_size, block_size>>>(the_map.get(),
                                                            hash_table_size,
                                                            groupby_output_table,
                                                            groupby_input_table,
                                                            out_aggregation_column,
                                                            global_write_index);
  
  // FIXME Work around for above kernel failing to launch for some instantiations of the_map template class
  //map_type * map = the_map.get();
  //const size_type map_size = hash_table_size;
  //void *args[] = { &map, &map_size, &groupby_output_table, &out_aggregation_column, &global_write_index};

  //void (*func)(const map_type * const, 
  //             const size_type, 
  //             gdf_table<size_type> * const, 
  //             aggregation_type * const,
  //             size_type * const ) = &(extract_groupby_result<map_type,size_type,aggregation_type>);

  //CUDA_RT_CALL(cudaLaunchKernel((const void*) func, extract_grid_size, block_size, args, 0, 0));
  // FIXME End work around

  CUDA_RT_CALL(cudaDeviceSynchronize());

  // At the end of the extraction kernel, the global write index will be equal to
  // the size of the output. Update the output size.
  cudaMemcpy(out_size, global_write_index, sizeof(size_type), cudaMemcpyDeviceToHost);
  CUDA_RT_CALL(cudaFree(global_write_index));

  // TODO Need to define comparator to allow sorting rows of a gdf_table
  /*
  // Optionally sort the groupby/aggregation result columns
  if(true == sort_result)
  {
    // Allocate double buffers needed for the cub Radix Sort
    groupby_type * groupby_result_alt;
    CUDA_RT_CALL(cudaMalloc(&groupby_result_alt, *out_size * sizeof(groupby_type)));

    aggregation_type * aggregation_result_alt;
    CUDA_RT_CALL(cudaMalloc(&aggregation_result_alt, *out_size * sizeof(aggregation_type)));

    cub::DoubleBuffer<groupby_type>     d_keys(out_groupby_column, groupby_result_alt);
    cub::DoubleBuffer<aggregation_type> d_vals(out_aggregation_column, aggregation_result_alt);

    // When called with temp_storage == nullptr, simply returns the required allocation size in
    // temp_storage_bytes
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_RT_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, *out_size));
    
    // allocate temp storage here and call sort again to actually sort arrays
    CUDA_RT_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_RT_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_vals, *out_size));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    // Update output pointers with sorted result
    // TODO Find a better way to do this. 
    // Sorted output may be in a different buffer than what was originally passed in... need to copy it 
    CUDA_RT_CALL(cudaMemcpy(out_groupby_column, d_keys.Current(), *out_size * sizeof(groupby_type), cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy(out_aggregation_column, d_vals.Current(), *out_size * sizeof(aggregation_type), cudaMemcpyDefault));

    // Free work buffers
    CUDA_RT_CALL(cudaFree(d_temp_storage));
    CUDA_RT_CALL(cudaFree(groupby_result_alt));
    CUDA_RT_CALL(cudaFree(aggregation_result_alt));
  }

  */
  return error;
}
#endif
