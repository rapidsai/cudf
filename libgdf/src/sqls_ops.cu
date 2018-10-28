/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include "thrust_rmm_allocator.h"

///#include "../include/sqls_rtti_comp.hpp" -- CORRECT: put me back
#include "sqls_rtti_comp.hpp"
#include "groupby/groupby.cuh"
#include "groupby/hash/aggregation_operations.cuh"
#include "nvtx_utils.h"

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

  // thrust::device_vector set to use rmmAlloc and rmmFree.
  template<typename T>
  using Vector = thrust::device_vector<T, rmm_allocator<T>>;

  void type_dispatcher(gdf_dtype col_type,
                       int col_index,
                       gdf_column** h_cols_in,
                       gdf_column** h_cols_out,
                       IndexT* d_indices,
                       size_t nrows_new)
  {
    cudaStream_t stream = 0; // TODO: non-default stream
    rmm_temp_allocator allocator(stream); 
    auto exec = thrust::cuda::par(allocator).on(stream);

    switch( col_type )
      {
      case GDF_INT8:
        {
          using ColType = int8_t;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);//pointer semantics (2)
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }
      case GDF_INT16:
        {
          using ColType = int16_t;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }
      case GDF_INT32:
        {
          using ColType = int32_t;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }
      case GDF_INT64:
        {
          using ColType = int64_t;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }
      case GDF_FLOAT32:
        {
          using ColType = float;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }
      case GDF_FLOAT64:
        {
          using ColType = double;

          ColType* d_in  = static_cast<ColType*>(h_cols_in[col_index]->data);
          ColType* d_out = static_cast<ColType*>(h_cols_out[col_index]->data);
          thrust::gather(exec,
                         d_indices, d_indices + nrows_new, //map of indices
                         d_in,                             //source
                         d_out);                           //=source[map]
          break;
        }

      default:
        assert( false );//type not handled
      }
    return;// State::True;
  }

  //copy from a set of gdf_columns:    h_cols_in
  //of size (#ncols):                  ncols
  //to another set of columns        : h_cols_out
  //by gathering via array of indices: d_indices
  //of size:                           nrows_new
  //
  void multi_gather_host(size_t ncols,  gdf_column** h_cols_in, gdf_column** h_cols_out, IndexT* d_indices, size_t nrows_new)
  {
    for(size_t col_index = 0; col_index<ncols; ++col_index)
      {
        gdf_dtype col_type = h_cols_in[col_index]->dtype;
        type_dispatcher(col_type,
                        col_index,
                        h_cols_in,
                        h_cols_out,
                        d_indices,
                        nrows_new);

        h_cols_out[col_index]->dtype = col_type;
        h_cols_out[col_index]->size = nrows_new;
        
        //TODO: h_cols_out[col_index]->valid
      }
  }

  int dtype_size(gdf_dtype col_type)
  {
    switch( col_type )
      {
      case GDF_INT8:
        {
          using ColType = int8_t;
	  
          return sizeof(ColType);
        }
      case GDF_INT16:
        {
          using ColType = int16_t;

          return sizeof(ColType);
        }
      case GDF_INT32:
        {
          using ColType = int32_t;

          return sizeof(ColType);
        }
      case GDF_INT64:
        {
          using ColType = int64_t;

          return sizeof(ColType);
        }
      case GDF_FLOAT32:
        {
          using ColType = float;

          return sizeof(ColType);
        }
      case GDF_FLOAT64:
        {
          using ColType = double;

          return sizeof(ColType);
        }

      default:
        assert( false );//type not handled
      }
      return 0;
  }

#ifdef DEBUG_
  void run_echo(size_t nrows,     //in: # rows
                gdf_column* cols, //in: host-side array of gdf_columns
                size_t ncols,     //in: # cols
                int flag_sorted,  //in: flag specifying if rows are pre-sorted (1) or not (0)
                gdf_column agg_in)//in: column to aggregate
  {
    std::cout<<"############# Echo: #############\n";
    std::cout<<"nrows: "<<nrows<<"\n";
    std::cout<<"ncols: "<<ncols<<"\n";
    std::cout<<"sorted: "<<flag_sorted<<"\n";

    std::cout<<"input cols:\n";
    for(auto i = 0; i < ncols; ++i)
      {
        switch(i)
          {
          case 0:
          case 1:
            {
              std::vector<int32_t> v(nrows);
              int32_t* p = &v[0];
              cudaMemcpy(p, cols[i].data, nrows*sizeof(int32_t), cudaMemcpyDeviceToHost);
              std::copy(v.begin(), v.end(), std::ostream_iterator<int32_t>(std::cout,","));
              std::cout<<"\n";
              break;
            }
          case 2:
            {
              std::vector<double> v(nrows);
              double* p = &v[0];
              cudaMemcpy(p, cols[i].data, nrows*sizeof(double), cudaMemcpyDeviceToHost);
              std::copy(v.begin(), v.end(), std::ostream_iterator<double>(std::cout,","));
              std::cout<<"\n";
              break;
            }
          }
      }


    std::cout<<"col to aggregate on:\n";
    std::vector<double> v(nrows);
    double* p = &v[0];
    cudaMemcpy(p, agg_in.data, nrows*sizeof(double), cudaMemcpyDeviceToHost);
    std::copy(v.begin(), v.end(), std::ostream_iterator<double>(std::cout,","));
    std::cout<<"\n";
  }
#endif
  





//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's nevessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_count(size_t nrows,     //in: # rows
                             gdf_column* cols, //in: host-side array of gdf_columns
                             size_t ncols,     //in: # cols
                             int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                             void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                             int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                             IndexT* d_indx,      //out: device-side array of row indices after sorting
                             IndexT* d_kout,      //out: device-side array of rows after gropu-by
                             gdf_column& c_vout,  //out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                             size_t* new_sz,   //out: host-side # rows of d_count
                             bool flag_distinct = false)
{
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  switch( c_vout.dtype )
    {
    case GDF_INT8:
      {
        using T = char;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
        
        break;
      }

    case GDF_INT16:
      {
        using T = short;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
        
        break;
      }
    case GDF_INT32:
      {
        using T = int;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
	
        break;
      }

    case GDF_INT64:
      {
        using T = long;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
	
        break;
      }

    case GDF_FLOAT32:
      {
        using T = float;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
	
        break;
      }

    case GDF_FLOAT64:
      {
        using T = double;

        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_count_sort(nrows,
                                                ncols,
                                                d_cols,
                                                d_types,
                                                d_indx,
                                                d_kout,
                                                d_vout,
                                                flag_sorted,
                                                flag_distinct);
	
        break;
      }

    default:
      return GDF_UNSUPPORTED_DTYPE;
    }
  
  return GDF_SUCCESS;
}

//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's necessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_sum(size_t nrows,     //in: # rows
                           gdf_column* cols, //in: host-side array of gdf_columns
                           size_t ncols,     //in: # cols
                           int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                           gdf_column& agg_in,//in: column to aggregate
                           void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                           int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                           IndexT* d_indx,      //out: device-side array of row indices after sorting
                           gdf_column& agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
                           IndexT* d_kout,      //out: device-side array of rows after group-by
                           gdf_column& c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                           size_t* new_sz)   //out: host-side # rows of d_count
{
  //not supported by g++-4.8:
  //
  //static_assert(std::is_trivially_copy_constructible<gdf_column>::value,
  //		"error: gdf_column must have shallow copy constructor; otherwise cannot pass output by copy.");

#ifdef DEBUG_
  run_echo(nrows,     //in: # rows
           cols, //in: host-side array of gdf_columns
           ncols,     //in: # cols
           flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
           agg_in);//in: column to aggregate
#endif

  assert( agg_in.dtype == agg_p.dtype );
  assert( agg_in.dtype == c_vout.dtype );
  
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  switch( agg_in.dtype )
    {
    case GDF_INT8:
      {
        using T = char;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT16:
      {
        using T = short;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);

        break;
      }
    case GDF_INT32:
      {
        using T = int;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT64:
      {
        using T = long;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT32:
      {
        using T = float;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT64:
      {
        using T = double;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_sum_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    default:
      return GDF_UNSUPPORTED_DTYPE;
    }

  return GDF_SUCCESS;
}


//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's necessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_min(size_t nrows,     //in: # rows
                           gdf_column* cols, //in: host-side array of gdf_columns
                           size_t ncols,     //in: # cols
                           int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                           gdf_column& agg_in,//in: column to aggregate
                           void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                           int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                           IndexT* d_indx,      //out: device-side array of row indices after sorting
                           gdf_column& agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
                           IndexT* d_kout,      //out: device-side array of rows after gropu-by
                           gdf_column& c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                           size_t* new_sz)   //out: host-side # rows of d_count
{
  //not supported by g++-4.8:
  //
  //static_assert(std::is_trivially_copy_constructible<gdf_column>::value,
  //		"error: gdf_column must have shallow copy constructor; otherwise cannot pass output by copy.");

  assert( agg_in.dtype == agg_p.dtype );
  assert( agg_in.dtype == c_vout.dtype );
  
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  switch( agg_in.dtype )
    {
    case GDF_INT8:
      {
        using T = char;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT16:
      {
        using T = short;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);

        break;
      }
    case GDF_INT32:
      {
        using T = int;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT64:
      {
        using T = long;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT32:
      {
        using T = float;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT64:
      {
        using T = double;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_min_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    default:
      return GDF_UNSUPPORTED_DTYPE;
    }

  return GDF_SUCCESS;
}


//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's necessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_max(size_t nrows,     //in: # rows
                           gdf_column* cols, //in: host-side array of gdf_columns
                           size_t ncols,     //in: # cols
                           int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                           gdf_column& agg_in,//in: column to aggregate
                           void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                           int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                           IndexT* d_indx,      //out: device-side array of row indices after sorting
                           gdf_column& agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
                           IndexT* d_kout,      //out: device-side array of rows after gropu-by
                           gdf_column& c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                           size_t* new_sz)   //out: host-side # rows of d_count
{
  //not supported by g++-4.8:
  //
  //static_assert(std::is_trivially_copy_constructible<gdf_column>::value,
  //		"error: gdf_column must have shallow copy constructor; otherwise cannot pass output by copy.");

  assert( agg_in.dtype == agg_p.dtype );
  assert( agg_in.dtype == c_vout.dtype );
  
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  switch( agg_in.dtype )
    {
    case GDF_INT8:
      {
        using T = char;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT16:
      {
        using T = short;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);

        break;
      }
    case GDF_INT32:
      {
        using T = int;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT64:
      {
        using T = long;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT32:
      {
        using T = float;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT64:
      {
        using T = double;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_max_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    default:
      return GDF_UNSUPPORTED_DTYPE;
    }

  return GDF_SUCCESS;
}

//apparent duplication of info between
//gdf_column array and two arrays:
//           d_cols = data slice of gdf_column array;
//           d_types = dtype slice of gdf_column array;
//but it's necessary because the gdf_column array is host
//(even though its data slice is on device)
//
gdf_error gdf_group_by_avg(size_t nrows,     //in: # rows
                           gdf_column* cols, //in: host-side array of gdf_columns
                           size_t ncols,     //in: # cols
                           int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
                           gdf_column& agg_in,//in: column to aggregate
                           void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
                           int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
                           IndexT* d_indx,      //out: device-side array of row indices after sorting
                           IndexT* d_cout,      //out: device-side array of (COUNT-ed) values as a result of group-by;
                           gdf_column& agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
                           IndexT* d_kout,      //out: device-side array of rows after gropu-by
                           gdf_column& c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
                           size_t* new_sz)   //out: host-side # rows of d_count
{
  //not supported by g++-4.8:
  //
  //static_assert(std::is_trivially_copy_constructible<gdf_column>::value,
  //		"error: gdf_column must have shallow copy constructor; otherwise cannot pass output by copy.");

  assert( agg_in.dtype == agg_p.dtype );
  assert( agg_in.dtype == c_vout.dtype );
  
  //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  switch( agg_in.dtype )
    {
    case GDF_INT8:
      {
        using T = char;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT16:
      {
        using T = short;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);

        break;
      }
    case GDF_INT32:
      {
        using T = int;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_INT64:
      {
        using T = long;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT32:
      {
        using T = float;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    case GDF_FLOAT64:
      {
        using T = double;

        T* d_agg   = static_cast<T*>(agg_in.data);
        T* d_agg_p = static_cast<T*>(agg_p.data);
        T* d_vout  = static_cast<T*>(c_vout.data);
        *new_sz = multi_col_group_by_avg_sort(nrows,
                                              ncols,
                                              d_cols,
                                              d_types,
                                              d_agg,
                                              d_indx,
                                              d_cout,
                                              d_agg_p,
                                              d_kout,
                                              d_vout,
                                              flag_sorted);
	
        break;
      }

    default:
      return GDF_UNSUPPORTED_DTYPE;
    }

  return GDF_SUCCESS;
}

gdf_error gdf_group_by_single(int ncols,                    // # columns
                              gdf_column** cols,            //input cols
                              gdf_column* col_agg,          //column to aggregate on
                              gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                              gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                            //(multi-gather based on indices, which are needed anyway)
                              gdf_column* out_col_agg,      //aggregation result
                              gdf_context* ctxt,            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
                              gdf_agg_op op)                //aggregation operation
{
  if((0 == ncols)
     || (nullptr == cols)
     || (nullptr == col_agg)
     || (nullptr == out_col_agg)
     || (nullptr == ctxt))
  {
    return GDF_DATASET_EMPTY;
  }
  for (int i = 0; i < ncols; ++i) {
	GDF_REQUIRE(!cols[i]->valid, GDF_VALIDITY_UNSUPPORTED);
  }
  GDF_REQUIRE(!col_agg->valid, GDF_VALIDITY_UNSUPPORTED);

  // If there are no rows in the input, set the output rows to 0 
  // and return immediately with success
  if( (0 == cols[0]->size )
      || (0 == col_agg->size))
  {
    if( (nullptr != out_col_agg) ){
      out_col_agg->size = 0;
    }
    if(nullptr != out_col_indices ) {
        out_col_indices->size = 0;
    }

    for(int col = 0; col < ncols; ++col){
      if(nullptr != out_col_values){
        if( nullptr != out_col_values[col] ){
          out_col_values[col]->size = 0;
        }
      }
    }
    return GDF_SUCCESS;
  }

  gdf_error gdf_error_code{GDF_SUCCESS};
  
  PUSH_RANGE("LIBGDF_GROUPBY", GROUPBY_COLOR);
  
  if( ctxt->flag_method == GDF_SORT )
    {
      std::vector<gdf_column> v_cols(ncols);
      for(auto i = 0; i < ncols; ++i)
        {
          v_cols[i] = *(cols[i]);
        }
      
      gdf_column* h_columns = &v_cols[0];
      size_t nrows = h_columns[0].size;

      size_t n_group = 0;

      Vector<IndexT> d_indx;//allocate only if necessary (see below)
      Vector<void*> d_cols(ncols, nullptr);
      Vector<int>   d_types(ncols, 0);
  
      void** d_col_data = d_cols.data().get();
      int* d_col_types = d_types.data().get();

      IndexT* ptr_d_indx = nullptr;
      if( out_col_indices )
        ptr_d_indx = static_cast<IndexT*>(out_col_indices->data);
      else
        {
          d_indx.resize(nrows);
          ptr_d_indx = d_indx.data().get();
        }

      Vector<IndexT> d_sort(nrows, 0);
      IndexT* ptr_d_sort = d_sort.data().get();
      
      gdf_column c_agg_p;
      c_agg_p.dtype = col_agg->dtype;
      c_agg_p.size = nrows;
      Vector<char> d_agg_p(nrows * dtype_size(c_agg_p.dtype));//purpose: avoids a switch-case on type;
      c_agg_p.data = d_agg_p.data().get();

      switch( op )
        {
        case GDF_SUM:
          gdf_group_by_sum(nrows,
                           h_columns,
                           static_cast<size_t>(ncols),
                           ctxt->flag_sorted,
                           *col_agg,
                           d_col_data, //allocated
                           d_col_types,//allocated
                           ptr_d_sort, //allocated
                           c_agg_p,    //allocated
                           ptr_d_indx, //allocated (or, passed in)
                           *out_col_agg,
                           &n_group);
          break;
          
        case GDF_MIN:
          gdf_group_by_min(nrows,
                           h_columns,
                           static_cast<size_t>(ncols),
                           ctxt->flag_sorted,
                           *col_agg,
                           d_col_data, //allocated
                           d_col_types,//allocated
                           ptr_d_sort, //allocated
                           c_agg_p,    //allocated
                           ptr_d_indx, //allocated (or, passed in)
                           *out_col_agg,
                           &n_group);
          break;

        case GDF_MAX:
          gdf_group_by_max(nrows,
                           h_columns,
                           static_cast<size_t>(ncols),
                           ctxt->flag_sorted,
                           *col_agg,
                           d_col_data, //allocated
                           d_col_types,//allocated
                           ptr_d_sort, //allocated
                           c_agg_p,    //allocated
                           ptr_d_indx, //allocated (or, passed in)
                           *out_col_agg,
                           &n_group);
          break;

        case GDF_AVG:
          {
            Vector<IndexT> d_cout(nrows, 0);
            IndexT* ptr_d_cout = d_cout.data().get();
            
            gdf_group_by_avg(nrows,
                             h_columns,
                             static_cast<size_t>(ncols),
                             ctxt->flag_sorted,
                             *col_agg,
                             d_col_data, //allocated
                             d_col_types,//allocated
                             ptr_d_sort, //allocated
                             ptr_d_cout, //allocated
                             c_agg_p,    //allocated
                             ptr_d_indx, //allocated (or, passed in)
                             *out_col_agg,
                             &n_group);
          }
          break;
        case GDF_COUNT_DISTINCT:
          {
            assert( out_col_agg );
            assert( out_col_agg->size >= 1);

            gdf_group_by_count(nrows,
                               h_columns,
                               static_cast<size_t>(ncols),
                               ctxt->flag_sorted,
                               d_col_data, //allocated
                               d_col_types,//allocated
                               ptr_d_sort, //allocated
                               ptr_d_indx, //allocated (or, passed in)
                               *out_col_agg, //passed in
                               &n_group,
                               true);
            
          }
          break;
        case GDF_COUNT:
          {
            assert( out_col_agg );

            gdf_group_by_count(nrows,
                               h_columns,
                               static_cast<size_t>(ncols),
                               ctxt->flag_sorted,
                               d_col_data, //allocated
                               d_col_types,//allocated
                               ptr_d_sort, //allocated
                               ptr_d_indx, //allocated (or, passed in)
                               *out_col_agg, //passed in
                               &n_group);
            
          }
          break;
        default: // To eliminate error for unhandled enumerant N_GDF_AGG_OPS
          gdf_error_code = GDF_INVALID_API_CALL;
        }

      if( out_col_values )
        {
          multi_gather_host(ncols, cols, out_col_values, ptr_d_indx, n_group);
        }

      out_col_agg->size = n_group;
      if( out_col_indices )
        out_col_indices->size = n_group;

      //TODO: out_<col>->valid = ?????
    }
  else if( ctxt->flag_method == GDF_HASH )
    {

      bool sort_result = false;

      if(1 == ctxt->flag_sort_result){
        sort_result = true;
      }

      switch(op)
      {
        case GDF_MAX:
          {
            gdf_error_code = gdf_group_by_hash<max_op>(ncols,
                                             cols,
                                             col_agg,
                                             out_col_values,
                                             out_col_agg,
                                             sort_result);
            break;
          }
        case GDF_MIN:
          {
            gdf_error_code = gdf_group_by_hash<min_op>(ncols,
                                             cols,
                                             col_agg,
                                             out_col_values,
                                             out_col_agg,
                                             sort_result);
            break;
          }
        case GDF_SUM:
          {
            gdf_error_code = gdf_group_by_hash<sum_op>(ncols,
                                             cols,
                                             col_agg,
                                             out_col_values,
                                             out_col_agg,
                                             sort_result);
            break;
          }
        case GDF_COUNT:
          {
            gdf_error_code = gdf_group_by_hash<count_op>(ncols,
                                               cols,
                                               col_agg,
                                               out_col_values,
                                               out_col_agg,
                                               sort_result);
            break;
          }
        case GDF_AVG:
          {
            gdf_error_code = gdf_group_by_hash_avg(ncols,
                                         cols,
                                         col_agg,
                                         out_col_values,
                                         out_col_agg);
            break;
          }
        default:
          std::cerr << "Unsupported aggregation method for hash-based groupby." << std::endl;
          gdf_error_code = GDF_UNSUPPORTED_METHOD;
      }
    }
  else
    {
      gdf_error_code = GDF_UNSUPPORTED_METHOD;
    }

  POP_RANGE();
  
  return gdf_error_code;
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
  GDF_REQUIRE(!cols->valid, GDF_VALIDITY_UNSUPPORTED);
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
  GDF_REQUIRE(!cols->valid, GDF_VALIDITY_UNSUPPORTED);
  soa_col_info(cols, ncols, d_cols, d_types);

  *new_sz = multi_col_filter(nrows,
                             ncols,
                             d_cols,
                             d_types,
                             d_vals,
                             d_indx);

  
  return GDF_SUCCESS;
}

gdf_error gdf_group_by_sum(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_SUM);
}

gdf_error gdf_group_by_min(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_MIN);
}

gdf_error gdf_group_by_max(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_MAX);
}

gdf_error gdf_group_by_avg(int ncols,                    // # columns
                           gdf_column** cols,            //input cols
                           gdf_column* col_agg,          //column to aggregate on
                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                           gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                           gdf_column* out_col_agg,      //aggregation result
                           gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{  
  return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_AVG);
}

gdf_error gdf_group_by_count(int ncols,                    // # columns
                             gdf_column** cols,            //input cols
                             gdf_column* col_agg,          //column to aggregate on
                             gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
                             gdf_column** out_col_values,  //if not null return the grouped-by columns
                                                         //(multi-gather based on indices, which are needed anyway)
                             gdf_column* out_col_agg,      //aggregation result
                             gdf_context* ctxt)            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
{
  if( ctxt->flag_distinct )
    return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_COUNT_DISTINCT);
  else
    return gdf_group_by_single(ncols, cols, col_agg, out_col_indices, out_col_values, out_col_agg, ctxt, GDF_COUNT);
}


