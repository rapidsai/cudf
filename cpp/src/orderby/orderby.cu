/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Jean Pierre Huaroto <jeanpierre@blazingdb.com>
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

#include <type_traits>
#include <algorithm>

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include "rmm/thrust_rmm_allocator.h"

#include "../sqls/sqls_rtti_comp.h"

namespace{ //annonymus

  gdf_error multi_col_order_by(gdf_column** cols,
                               int8_t* asc_desc,
                               size_t ncols,
                               gdf_column* output_indices,
                               bool flag_nulls_are_smallest, 
                               bool null_as_largest_for_multisort = false)
  {
    GDF_REQUIRE(cols != nullptr && output_indices != nullptr, GDF_DATASET_EMPTY);
    GDF_REQUIRE(cols[0]->size == output_indices->size, GDF_COLUMN_SIZE_MISMATCH);
    /* NOTE: providing support for indexes to be multiple different types explodes compilation time, such that it become infeasible */
    GDF_REQUIRE(output_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

    // Check for null so we can use a faster sorting comparator 
    bool const have_nulls{ std::any_of(cols, cols + ncols, [](gdf_column * col){ return col->null_count > 0; }) };

    rmm::device_vector<void*> d_cols(ncols);
    rmm::device_vector<gdf_valid_type*> d_valids(ncols);
    rmm::device_vector<int> d_types(ncols, 0);

    void** d_col_data = d_cols.data().get();
    gdf_valid_type** d_valids_data = d_valids.data().get();
    int* d_col_types = d_types.data().get();

    gdf_error gdf_status = soa_col_info(cols, ncols, d_col_data, d_valids_data, d_col_types);
    if(GDF_SUCCESS != gdf_status)
      return gdf_status;

    // if using null_as_largest_for_multisort = true, then you cant set ascending descending order (asc_desc)
    GDF_REQUIRE(((null_as_largest_for_multisort && (nullptr == asc_desc)) || !null_as_largest_for_multisort), GDF_INVALID_API_CALL);

		multi_col_sort(d_col_data, d_valids_data, d_col_types, asc_desc, ncols, cols[0]->size,
				have_nulls, static_cast<int32_t*>(output_indices->data), flag_nulls_are_smallest, null_as_largest_for_multisort);

    return GDF_SUCCESS;
  }

} //end unknown namespace



gdf_error gdf_order_by(gdf_column** input_columns,
                       int8_t* asc_desc,
                       size_t num_inputs,
                       gdf_column* output_indices,
                       gdf_context * context)                       
{

  bool flag_nulls_are_smallest = false;
  bool null_as_largest_for_multisort = false;
  if (context->flag_null_sort_behavior == GDF_NULL_AS_SMALLEST)
  /* When sorting NULLS will be treated as the smallest number */
    flag_nulls_are_smallest = true;
  else if (context->flag_null_sort_behavior == GDF_NULL_AS_LARGEST_FOR_MULTISORT) 
  /* When sorting a multi column data set, if there is a NULL in any of the
       columns for a row, then that row will be will be treated as the largest number */
    null_as_largest_for_multisort = true;

  return multi_col_order_by(input_columns, asc_desc, num_inputs, output_indices, flag_nulls_are_smallest, null_as_largest_for_multisort);
}
