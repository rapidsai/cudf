/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>

#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>

#include <rmm/thrust_rmm_allocator.h>

/* ----------------------------------------------------------------------------*/
/**
 * @brief Wrapper around row_inequality_comparator to pass row and row+1
 */
/* ----------------------------------------------------------------------------*/

template <bool nullable = true>
struct compare_with_neighbor{

    compare_with_neighbor(device_table t, bool nulls_are_smallest, int8_t *const asc_desc_flags = nullptr) 
        : compare(t,t, nulls_are_smallest, asc_desc_flags) {}
    
    __device__ bool operator()(gdf_index_type i){
       return compare(i, i + 1);
    }
    row_inequality_comparator<nullable> compare;
};

bool is_sorted(cudf::table const &table,
                       std::vector<int8_t>const & descending,
                       bool nulls_are_smallest = false)                       
{
  cudaStream_t stream = 0;
  bool sorted = false;

  CUDF_EXPECTS(static_cast <unsigned int>(table.num_columns()) == descending.size(), "Number of columns in the table doesn't match the vector descending's size .\n");
  
  if (table.num_columns() == 0)
  {
      return true;
  }

  auto exec = rmm::exec_policy(stream)->on(stream);
  auto device_input_table = device_table::create(table);
  bool nullable = cudf::has_nulls(table);
  rmm::device_vector<int8_t> asc_desc(descending);

  gdf_size_type nrows = table.num_rows();
  
  if (nullable){
    auto ineq_op = compare_with_neighbor<true>(*device_input_table, nulls_are_smallest, asc_desc.data().get());
    sorted = thrust::all_of (exec, thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrows-1), ineq_op);

  } else {
    auto ineq_op = compare_with_neighbor<false>(*device_input_table, nulls_are_smallest, asc_desc.data().get());
    sorted = thrust::all_of (exec, thrust::make_counting_iterator(0), thrust::make_counting_iterator(nrows-1), ineq_op);
  }

  return sorted;
}
