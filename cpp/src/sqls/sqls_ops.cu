/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "rmm/thrust_rmm_allocator.h"

#include "sqls_rtti_comp.h"

//using IndexT = int;//okay...
using IndexT = size_t;







//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's nevessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_filter(size_t nrows,     //in: # rows
                     gdf_column* cols, //in: host-side array of gdf_columns
                     size_t ncols,     //in: # cols
                     void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                     int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                     void** d_vals,    //in: device-side array of values to filter against (type-erased)
                     size_t* d_indx,   //out: device-side array of row indices that remain after filtering
                     size_t* new_sz)   //out: host-side # rows that remain after filtering
{
  //copy H-D:
  //
  GDF_REQUIRE(!cols->valid || !cols->null_count, GDF_VALIDITY_UNSUPPORTED);

  gdf_error gdf_status = soa_col_info(cols, ncols, d_cols, d_types);
  if(GDF_SUCCESS != gdf_status)
    return gdf_status;

  *new_sz = multi_col_filter(nrows,
                             ncols,
                             d_cols,
                             d_types,
                             d_vals,
                             d_indx);

  
  return GDF_SUCCESS;
}
