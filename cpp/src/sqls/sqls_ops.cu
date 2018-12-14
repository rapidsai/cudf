/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "rmm/thrust_rmm_allocator.h"

#include "sqls_rtti_comp.h"

//using IndexT = int;//okay...
using IndexT = size_t;

namespace{ //annonymus

  //helper functions:
  //
  //flatten AOS info from gdf_columns into SOA (2 arrays):
  //(1) column array pointers and (2) types;
  //
  void soa_col_info(gdf_column* cols, size_t ncols, void** d_cols, int* d_types)
  {
    std::vector<void*> v_cols(ncols,nullptr);
    std::vector<int>   v_types(ncols, 0);
    for(size_t i=0;i<ncols;++i)
      {
        v_cols[i] = cols[i].data;
        v_types[i] = cols[i].dtype;
      }

    void** h_cols = &v_cols[0];
    int* h_types = &v_types[0];
    cudaMemcpy(d_cols, h_cols, ncols*sizeof(void*), cudaMemcpyHostToDevice);//TODO: add streams
    cudaMemcpy(d_types, h_types, ncols*sizeof(int), cudaMemcpyHostToDevice);//TODO: add streams
  }




}//end unknown namespace

//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's nevessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_order_by(size_t nrows,     //in: # rows
                       gdf_column* cols, //in: host-side array of gdf_columns
                       size_t ncols,     //in: # cols
                       void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                       int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                       size_t* d_indx)   //out: device-side array of re-rdered row indices
{
  //copy H-D:
  //
  GDF_REQUIRE(!cols->valid || !cols->null_count, GDF_VALIDITY_UNSUPPORTED);
  soa_col_info(cols, ncols, d_cols, d_types);
  
  multi_col_order_by(nrows,
                     ncols,
                     d_cols,
                     d_types,
                     d_indx);
  
  return GDF_SUCCESS;
}

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
  soa_col_info(cols, ncols, d_cols, d_types);

  *new_sz = multi_col_filter(nrows,
                             ncols,
                             d_cols,
                             d_types,
                             d_vals,
                             d_indx);

  
  return GDF_SUCCESS;
}

