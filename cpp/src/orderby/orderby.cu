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
    //      also we are kind of assuming they will just work

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
                   have_nulls,
                   (size_t *) output_indices->data,
                   (ctxt ? ctxt->flag_nulls_are_smallest : false));

    return GDF_SUCCESS;
  }

} //end unknown namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @Param[in] cols Host-side array of gdf_columns
 * @Param[in] ncols # columns
 * @Param[in] ctxt Struct with additional info: bool flag_nulls_are_smallest
 * @Param[out] output_indices Pre-allocated gdf_column to be filled
 * with sorted indices
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_order_by(gdf_column** cols,
                       size_t ncols,
                       gdf_column* output_indices,
                       gdf_context* ctxt)
{
  return multi_col_order_by(cols, ncols, nullptr, output_indices, ctxt);
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @Param[in] cols Array of gdf_columns
 * @Param[in] asc_desc Device array of sort order types for each column
 * (0 is ascending order and 1 is descending)
 * @Param[in] ncols # columns
 * @Param[in] ctxt Struct with additional info: bool flag_nulls_are_smallest
 * @Param[out] output_indices Pre-allocated gdf_column to be filled
 * with sorted indices
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_order_by_asc_desc(gdf_column** cols,
		                            char* asc_desc,
                                size_t ncols,
                                gdf_column* output_indices,
		                            gdf_context* ctxt)
{
  return multi_col_order_by(cols, ncols, asc_desc, output_indices, ctxt);
}
