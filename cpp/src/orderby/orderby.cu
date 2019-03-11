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
                               bool flag_nulls_are_smallest)
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

		multi_col_sort(d_col_data, d_valids_data, d_col_types, asc_desc, ncols, cols[0]->size,
				have_nulls, static_cast<int32_t*>(output_indices->data), flag_nulls_are_smallest);

    return GDF_SUCCESS;
  }

} //end unknown namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @param[in] cols Array of gdf_columns
 * @param[in] asc_desc Device array of sort order types for each column
 * (0 is ascending order and 1 is descending). If NULL is provided defaults
 * to ascending order for evey column.
 * @param[in] ncols # columns
 * @param[in] flag_nulls_are_smallest Flag to indicate if nulls are to be considered
 * smaller than non-nulls or viceversa
 * @param[out] output_indices Pre-allocated gdf_column to be filled
 * with sorted indices
 * 
 * @returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_order_by(gdf_column** cols,
                       int8_t* asc_desc,
                       size_t ncols,
                       gdf_column* output_indices,
                       int flag_nulls_are_smallest)
{
  return multi_col_order_by(cols, asc_desc, ncols, output_indices, flag_nulls_are_smallest);
}
