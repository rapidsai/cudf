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

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "dataframe/type_dispatcher.hpp"

#include "rmm/thrust_rmm_allocator.h"

#include "../sqls/sqls_rtti_comp.h"

namespace{ //annonymus

  // thrust::device_vector set to use rmmAlloc and rmmFree.
  template<typename T>
  using Device_Vector = thrust::device_vector<T, rmm_allocator<T>>;

  struct multi_col_sort_forwarder {
    template <typename col_type>
    typename std::enable_if<std::is_integral<col_type>::value>::type
    operator()(void* const *           d_col_data,
               gdf_valid_type* const * d_valids_data,
               int*                    d_col_types,
               char*                   d_asc_desc,
               size_t                  ncols,
               size_t                  nrows,
               bool                    have_nulls,
               void*                   output_indices,
               bool                    nulls_are_smallest)
    {
      multi_col_sort(d_col_data,
                    d_valids_data,
                    d_col_types,
                    d_asc_desc,
                    ncols,
                    nrows,
                    have_nulls,
                    static_cast<col_type*>(output_indices),
                    nulls_are_smallest);
    }
    template <typename col_type>
    typename std::enable_if<!std::is_integral<col_type>::value>::type 
    operator()(void* const *           d_col_data,
               gdf_valid_type* const * d_valids_data,
               int*                    d_col_types,
               char*                   d_asc_desc,
               size_t                  ncols,
               size_t                  nrows,
               bool                    have_nulls,
               void*                   output_indices,
               bool                    nulls_are_smallest)
    {
    }
 };

  gdf_error multi_col_order_by(gdf_column** cols,
                               size_t ncols,
                               char* asc_desc,
                               gdf_column* output_indices,
                               bool flag_nulls_are_smallest)
  {
    // Return error if the inputs or no output pointers are invalid
    if (cols == nullptr || output_indices == nullptr) { return GDF_DATASET_EMPTY; }

    GDF_REQUIRE(cols[0]->size == output_indices->size, GDF_COLUMN_SIZE_MISMATCH);

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

    cudf::type_dispatcher(output_indices->dtype, multi_col_sort_forwarder{},
                          d_col_data,
                          d_valids_data,
                          d_col_types,
                          asc_desc,
                          ncols,
                          cols[0]->size,
                          have_nulls,
                          output_indices->data,
                          flag_nulls_are_smallest); 

    return GDF_SUCCESS;
  }

} //end unknown namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @Param[in] cols Host-side array of gdf_columns
 * @Param[in] ncols # columns
 * @Param[in] flag_nulls_are_smallest Flag to indicate if nulls are to be considered
 * smaller than non-nulls or viceversa
 * @Param[out] output_indices Pre-allocated gdf_column to be filled
 * with sorted indices
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_order_by(gdf_column** cols,
                       size_t ncols,
                       gdf_column* output_indices,
                       int flag_nulls_are_smallest)
{
  return multi_col_order_by(cols, ncols, nullptr, output_indices, flag_nulls_are_smallest);
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @Param[in] cols Array of gdf_columns
 * @Param[in] asc_desc Device array of sort order types for each column
 * (0 is ascending order and 1 is descending)
 * @Param[in] ncols # columns
 * @Param[in] flag_nulls_are_smallest Flag to indicate if nulls are to be considered
 * smaller than non-nulls or viceversa
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
                                int flag_nulls_are_smallest)
{
  return multi_col_order_by(cols, ncols, asc_desc, output_indices, flag_nulls_are_smallest);
}
