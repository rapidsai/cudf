/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include "../include/sqls_rtti_comp.hpp"

namespace{ //annonymus

  //helper function:
  //flatten AOS info from gdf_columns into SOA (2 arrays):
  //(1) column array pointers and (2) types;
  //
  void soa_col_info(gdf_column* cols, size_t ncols, void** d_cols, int* d_types)
  {
    std::vector<void*> v_cols(ncols,nullptr);
    std::vector<int>   v_types(ncols, 0);
    for(int i=0;i<ncols;++i)
      {
	v_cols[i] = cols[i].data;
	v_types[i] = cols[i].dtype;
      }

    void** h_cols = &v_cols[0];
    int* h_types = &v_types[0];
    cudaMemcpy(d_cols, h_cols, ncols*sizeof(void*), cudaMemcpyHostToDevice);//TODO: add streams
    cudaMemcpy(d_types, h_types, ncols*sizeof(int), cudaMemcpyHostToDevice);//TODO: add streams
  }
}

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
  soa_col_info(cols, ncols, d_cols, d_types);

  *new_sz = multi_col_filter(nrows,
		   ncols,
		   d_cols,
		   d_types,
		   d_vals,
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
gdf_error gdf_group_by_count(size_t nrows,     //in: # rows
			     gdf_column* cols, //in: host-side array of gdf_columns
			     size_t ncols,     //in: # cols
			     int flag_sorted,  //in: flag specififying if rows are pre-sorted (1) or not (0)
			     void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
			     int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
			     int* d_indx,      //out: device-side array of row indices after sorting
			     int* d_kout,      //out: device-side array of rows after gropu-by
			     int* d_count,     //out: device-side array of aggregated values (COUNT-ed) as a result of group-by;
			     size_t* new_sz)   //out: host-side # rows of d_count
{
   //copy H-D:
  //
  soa_col_info(cols, ncols, d_cols, d_types);

  *new_sz = multi_col_group_by_count_sort(nrows,
					  ncols,
					  d_cols,
					  d_types,
					  d_indx,
					  d_kout,
					  d_count,
					  flag_sorted);

  
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
			   gdf_column agg_in,//in: column to aggregate
			   void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
			   int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
			   int* d_indx,      //out: device-side array of row indices after sorting
			   gdf_column agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
			   int* d_kout,      //out: device-side array of rows after gropu-by
			   gdf_column c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
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
			   gdf_column agg_in,//in: column to aggregate
			   void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
			   int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
			   int* d_indx,      //out: device-side array of row indices after sorting
			   gdf_column agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
			   int* d_kout,      //out: device-side array of rows after gropu-by
			   gdf_column c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
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
			   gdf_column agg_in,//in: column to aggregate
			   void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
			   int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
			   int* d_indx,      //out: device-side array of row indices after sorting
			   gdf_column agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
			   int* d_kout,      //out: device-side array of rows after gropu-by
			   gdf_column c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
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
			   gdf_column agg_in,//in: column to aggregate
			   void** d_cols,    //out: pre-allocated device-side array to be filled with gdf_column::data for each column; slicing of gdf_column array (host)
			   int* d_types,     //out: pre-allocated device-side array to be filled with gdf_colum::dtype for each column; slicing of gdf_column array (host)
			   int* d_indx,      //out: device-side array of row indices after sorting
			   int* d_cout,      //out: device-side array of (COUNT-ed) values as a result of group-by;
			   gdf_column agg_p, //out: reordering of d_agg after sorting; requires shallow (trivial) copy-construction (see static_assert below);
			   int* d_kout,      //out: device-side array of rows after gropu-by
			   gdf_column c_vout,//out: aggregated column; requires shallow (trivial) copy-construction (see static_assert below);
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
