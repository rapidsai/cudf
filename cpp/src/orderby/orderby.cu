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

#include "cudf.h"
#include "utilities/cudf_utils.h"

#include "rmm/thrust_rmm_allocator.h"

#include "../sqls/sqls_rtti_comp.h"

namespace{ //annonymus

  // thrust::device_vector set to use rmmAlloc and rmmFree.
  template<typename T>
  using Device_Vector = thrust::device_vector<T, rmm_allocator<T>>;

  gdf_error multi_col_order_by(gdf_column** cols,
                               size_t ncols,
                               char* asc_desc,
                               gdf_column* output_indices,
                               gdf_context* ctxt)
  {
    //TODO: don't assume type of output is size_t
    //TODO: make these allocations happen with the new memory manager when we can
    //also we are kind of assuming they will just work, yeesh!

    // Return error if the inputs or no output pointers are invalid
    if (cols == nullptr || output_indices == nullptr) { return GDF_DATASET_EMPTY; }

    // Check for null so we can use a faster sorting comparator 
    bool have_nulls = false;
    for (size_t i = 0; i < ncols; i++) {
      if (cols[i]->null_count > 0) {
        have_nulls = true;
        break;
      }
    }

    Device_Vector<void*> d_cols(ncols);
    Device_Vector<gdf_valid_type*> d_valids(ncols);
    Device_Vector<int> d_types(ncols, 0);

    void** d_col_data = d_cols.data().get();
    gdf_valid_type** d_valids_data = d_valids.data().get();
    int* d_col_types = d_types.data().get();

    soa_col_info(cols, ncols, d_col_data, d_valids_data, d_col_types);

    multi_col_sort(d_col_data,
                   d_valids_data,
                   d_col_types,
                   asc_desc,
                   ncols,
                   cols[0]->size,
                   (size_t *) output_indices->data,
                   have_nulls,
                   ctxt->flag_nulls_are_smallest);

    return GDF_SUCCESS;
  }

} //end unknown namespace


gdf_error gdf_order_by(gdf_column** cols,           //in: host-side array of gdf_columns
                       size_t ncols,                //in: # cols
                       gdf_column* output_indices,  //out: pre-allocated device-side gdf_column to be filled with sorted indices
                       gdf_context* ctxt)           //struct with additional info: bool flag_nulls_are_smallest
{
  return multi_col_order_by(cols, ncols, nullptr, output_indices, ctxt);
}

gdf_error gdf_order_by_asc_desc(gdf_column** cols,          //in: host-side array of gdf_columns
                                size_t ncols,               //in: array of chars where 0 is ascending order and 1 is descending order and for each input column
		                            char* asc_desc,             //in: # cols
                                gdf_column* output_indices, //out: pre-allocated device-side gdf_column to be filled with sorted indices
		                            gdf_context* ctxt)          //struct with additional info: bool flag_nulls_are_smallest
{
  return multi_col_order_by(cols, ncols, asc_desc, output_indices, ctxt);
}
