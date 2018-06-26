/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#pragma once

#include <iostream>
#include <cassert>
#include <iterator>

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/gather.h>

///#include "gdf/cffi/types.h"

struct LesserRTTI
{
  // LesserRTTI(const thrust::device_vector<void*>& cols,
  // 	     const thrust::device_vector<int>& types):
  //   columns_(cols.data().get()),
  //   rtti_(types.data().get()),
  //   sz_(cols.size())
  // {
  //   assert( types.size() == sz_ );
  // }

  __host__ __device__
  LesserRTTI(void* const* cols,
	     int* const types,
	     size_t sz):
    columns_(cols),
    rtti_(types),
    sz_(sz),
    vals_(nullptr)
  {
  }

  __host__ __device__
  LesserRTTI(void* const* cols,
	     int* const types,
	     size_t sz,
	     const void* const * vals):
    columns_(cols),
    rtti_(types),
    sz_(sz),
    vals_(vals)
  {
  }

  

  __host__ __device__
  bool equal(int row1, int row2) const
  {
    for(int col_index = 0; col_index < sz_; ++col_index)
      {
	gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

	OpEqual eq;
	switch( type_dispatcher(eq, col_type, col_index, row1, row2) )
	  {
	  case State::False:
	    return false;
	    
	  case State::True:
	  case State::Undecided:
	    break;
	  }
      }
    return true;
  }

  __host__ __device__
  bool equal_v(size_t row1) const
  {
    for(int col_index = 0; col_index < sz_; ++col_index)
      {
	gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

	OpEqualV eq(vals_);
	switch( type_dispatcher(eq, col_type, col_index, row1, 0) )
	  {
	  case State::False:
	    return false;
	    
	  case State::True:
	  case State::Undecided:
	    break;
	  }
      }
    return true;
  }

  __host__ __device__
  bool less(int row1, int row2) const
  {
    for(int col_index = 0; col_index < sz_; ++col_index)
      {
	gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

	OpLess less;
	switch( type_dispatcher(less, col_type, col_index, row1, row2) )
	  {
	  case State::False:
	    return false;
	  case State::True:
	    return true;
	  case State::Undecided:
	    break;
	  }
      }
    return false;
  }

  template<typename ColType>
  __host__ __device__
   static ColType at(int col_index,
		     int row,
		     const void* const * columns)
  {
    return (static_cast<const ColType* const>(columns[col_index]))[row];
  }
  
private:
  enum class State {False = 0, True = 1, Undecided = 2};

  struct OpLess
  {
     __host__ __device__
    OpLess(void)
    {
    }
    
     template<typename ColType>
     __host__ __device__
     State operator() (int col_index,
		       int row1,
		       int row2,
		       const void* const * columns,
		       ColType )
    {
      ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
      
      if( res1 < res2 )
	return State::True;
      else if( res1 == res2 )
	return State::Undecided;
      else
	return State::False;
    }
  };

  struct OpEqual
  {
     __host__ __device__
    OpEqual(void)
    {
    }
    
     template<typename ColType>
     __host__ __device__
     State operator() (int col_index,
		       int row1,
		       int row2,
		       const void* const * columns,
		       ColType )
    {
      ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
      
      if( res1 != res2 )
	return State::False;
      else
	return State::Undecided;
    }
  };

  struct OpEqualV
  {
     __host__ __device__
     explicit OpEqualV(const void* const * vals):
       target_vals_(vals)
    {
    }
    
     template<typename ColType>
     __host__ __device__
     State operator() (int col_index,
		       size_t row1,
		       size_t ,//ignored
		       const void* const * columns,
		       ColType )
    {
      ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      ColType res2 = LesserRTTI::at<ColType>(col_index, 0, target_vals_);
      
      if( res1 != res2 )
	return State::False;
      else
	return State::Undecided;
    }

  private:
    const void* const * target_vals_;
  }; 

  template<typename Predicate>
  __host__ __device__
  State type_dispatcher(Predicate pred, gdf_dtype col_type, int col_index, int row1, int row2) const
  {
    switch( col_type )
      {
      case GDF_INT8:
	{
	  using ColType = char;
	  
	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}
      case GDF_INT16:
	{
	  using ColType = short;

	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}
      case GDF_INT32:
	{
	  using ColType = int;

	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}
      case GDF_INT64:
	{
	  using ColType = long;

	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}
      case GDF_FLOAT32:
	{
	  using ColType = float;

	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}
      case GDF_FLOAT64:
	{
	  using ColType = double;

	  ColType dummy;
	  return pred(col_index, row1, row2, columns_, dummy);
	}

      default:
	assert( false );//type not handled
      }
    return State::Undecided;
  }
  
  const void* const * columns_;
  const int* const rtti_;
  size_t sz_;
  const void* const * vals_; //for filtering
};

//###########################################################################
//#                          Multi-column ORDER-BY:                         #
//###########################################################################
//Version with array of columns,
//using type erasure and RTTI at
//comparison operator level;
//
//args:
//Input:
// nrows    = # rows;
// ncols    = # columns;
// d_cols   = device array to ncols type erased columns;
// d_gdf_t  = device array to runtime column types;
// stream   = cudaStream to work in;
//
//Output:
// d_indx   = vector of indices re-ordered after sorting;
//
template<typename IndexT>
__host__ __device__
void multi_col_order_by(size_t nrows,
			size_t ncols,
			void* const* d_cols,
			int* const  d_gdf_t,
			IndexT*      d_indx,
			cudaStream_t stream = NULL)
{
  LesserRTTI f(d_cols, d_gdf_t, ncols);

  thrust::sequence(thrust::cuda::par.on(stream), d_indx, d_indx+nrows, 0);//cannot use counting_iterator
  //                                          2 reasons:
  //(1.) need to return a container result;
  //(2.) that container must be mutable;
  
  thrust::sort(thrust::cuda::par.on(stream),
		 d_indx, d_indx+nrows,
		 [f] __host__ __device__ (IndexT i1, IndexT i2){
		         return f.less(i1, i2);
		 });
}

//###########################################################################
//#                          Multi-column Filter:                           #
//###########################################################################
//Version with array of columns,
//using type erasure and RTTI at
//comparison operator level;
//
//Filter array of columns ("table")
//on given array of values;
//Input:
// nrows    = # rows;
// ncols    = # columns;
// d_cols   = device array to ncols type erased columns;
// d_gdf_t  = device array to runtime column types;
// d_vals      = tuple of values to filter against;
// stream     = cudaStream to work in;
//Output:
// d_flt_indx = device_vector of indx for which a match exists (no holes in array!);
//Return:
// new_sz     = new #rows after filtering;
//
template<typename IndexT>
__host__ __device__
size_t multi_col_filter(size_t nrows,
			size_t ncols,
			void* const* d_cols,
			int* const  d_gdf_t,
			void* const* d_vals,
			IndexT* ptr_d_flt_indx,
			cudaStream_t stream = NULL)
{
  LesserRTTI f(d_cols, d_gdf_t, ncols, d_vals);
  
  //actual filtering happens here:
  //
  auto ret_iter_last =
    thrust::copy_if(thrust::cuda::par.on(stream),
		    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nrows),
		    ptr_d_flt_indx,
		    [f] __host__ __device__ (size_t indx){
		      return f.equal_v(indx);
		    });

  size_t new_sz = thrust::distance(ptr_d_flt_indx,ret_iter_last);
  
  return new_sz;
}


//###########################################################################
//#                          Multi-column Group-By:                         #
//###########################################################################
//group-by is a reduce_by_key
//
//not only COUNT(*) makes sense with multicolumn:
//one can specify a multi-column GROUP-BY criterion
//and one specifc column to aggregate on;
//
//Input:
// nrows    = # rows;
// ncols    = # columns;
// d_cols   = device array to ncols type erased columns;
// d_gdf_t  = device array to runtime column types;
// stream   = cudaStream to work in;
//Output:
// d_indx   = reordering of indices after sorting;
//             (passed as argument to avoid allocations inside the stream)
// d_kout   = indices of rows after group by;
// d_vout   = aggregated values (COUNT-ed) as a result of group-by; 
// d_items  = device_vector of items corresponding to indices in d_flt_indx;
//Return:
// ret      = # rows after aggregation;
//
template<typename IndexT>
__host__ __device__
size_t
multi_col_group_by_count_sort(size_t         nrows,
			      size_t         ncols,
			      void* const*   d_cols,
			      int* const     d_gdf_t,
			      IndexT*        ptr_d_indx,
			      IndexT*        ptr_d_kout,
			      IndexT*        ptr_d_vout,
			      bool           sorted = false,
			      cudaStream_t   stream = NULL)
{
  if( !sorted )
    multi_col_order_by(nrows, ncols, d_cols, d_gdf_t, ptr_d_indx, stream);

  LesserRTTI f(d_cols, d_gdf_t, ncols);

  thrust::pair<IndexT*, IndexT*> ret =
    thrust::reduce_by_key(thrust::cuda::par.on(stream),
			  ptr_d_indx, ptr_d_indx+nrows,
			  thrust::make_constant_iterator(1),
			  ptr_d_kout,
			  ptr_d_vout,
			  [f] __host__ __device__(int key1, int key2){
			         return f.equal(key1, key2);
			  });

  //COUNT(*) for each distinct entry gets collected in d_vout;
  //DISTINCT COUNT(*) is just: thrust::distance(d_kout.begin(), ret.first)
			    
  size_t new_sz = thrust::distance(ptr_d_vout, ret.second);
  return new_sz;
}


//group-by is a reduce_by_key
//
//COUNT(*) isn't the only one that makes sense with multicolumn:
//one can specify a multi-column GROUP-BY criterion
//and one specifc column to aggregate on;
//
//Input:
// nrows    = # rows;
// ncols    = # columns;
// d_cols   = device array to ncols type erased columns;
// d_gdf_t  = device array to runtime column types;
// d_agg    = column (device vector) to get aggregated;
//fctr      = functor to perform the aggregation <Tagg(Tagg, Tagg)>       
//stream    = cudaStream to work in;
//Output:
//d_indx    = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_agg_p   = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout    = indices of rows after group by;
//d_vout    = aggregated values (counted) as a result of group-by; 
//Return:
//ret       = # rows after aggregation;
//
template<typename ValsT,
	 typename IndexT,
	 typename Reducer>
__host__ __device__
size_t multi_col_group_by_sort(size_t         nrows,
			       size_t         ncols,
			       void* const*   d_cols,
			       int* const     d_gdf_t,
			       const ValsT*   ptr_d_agg,
			       Reducer        fctr,
			       IndexT*        ptr_d_indx,
			       ValsT*         ptr_d_agg_p,
			       IndexT*        ptr_d_kout,
			       ValsT*         ptr_d_vout,
			       bool           sorted = false,
			       cudaStream_t   stream = NULL)
{
  if( !sorted )
    multi_col_order_by(nrows, ncols, d_cols, d_gdf_t, ptr_d_indx, stream);
  
  thrust::gather(thrust::cuda::par.on(stream),
                 ptr_d_indx, ptr_d_indx + nrows,  //map[i]
  		 ptr_d_agg,                    //source[i]
  		 ptr_d_agg_p);                 //source[map[i]]

  LesserRTTI f(d_cols, d_gdf_t, ncols);
  
  thrust::pair<IndexT*, ValsT*> ret =
    thrust::reduce_by_key(thrust::cuda::par.on(stream),
  			    ptr_d_indx, ptr_d_indx + nrows,
  			    ptr_d_agg_p,
  			    ptr_d_kout,
  			    ptr_d_vout,
  			    [f] __host__ __device__(int key1, int key2){
			          return f.equal(key1, key2);
  			    },
  			    fctr);

  size_t new_sz = thrust::distance(ptr_d_vout, ret.second);
  return new_sz;
}

template<typename ValsT,
	 typename IndexT>
__host__ __device__
size_t multi_col_group_by_sum_sort(size_t         nrows,
				   size_t         ncols,
				   void* const*   d_cols,
				   int* const     d_gdf_t,
				   const ValsT*   ptr_d_agg,
				   IndexT*        ptr_d_indx,
				   ValsT*         ptr_d_agg_p,
				   IndexT*        ptr_d_kout,
				   ValsT*         ptr_d_vout,
				   bool           sorted = false,
				   cudaStream_t   stream = NULL)
{
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
			      return x+y;
  };

  using ReducerT = decltype(lamb);

  return multi_col_group_by_sort(nrows,
				 ncols,
				 d_cols,
				 d_gdf_t,
				 ptr_d_agg,
				 lamb,
				 ptr_d_indx,
				 ptr_d_agg_p,
				 ptr_d_kout,
				 ptr_d_vout,
				 sorted,
				 stream);
}

template<typename ValsT,
	 typename IndexT>
__host__ __device__
size_t multi_col_group_by_min_sort(size_t         nrows,
				   size_t         ncols,
				   void* const*   d_cols,
				   int* const     d_gdf_t,
				   const ValsT*   ptr_d_agg,
				   IndexT*        ptr_d_indx,
				   ValsT*         ptr_d_agg_p,
				   IndexT*        ptr_d_kout,
				   ValsT*         ptr_d_vout,
				   bool           sorted = false,
				   cudaStream_t   stream = NULL)
{
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
    return (x<y?x:y);
  };

  using ReducerT = decltype(lamb);

  return multi_col_group_by_sort(nrows,
				 ncols,
				 d_cols,
				 d_gdf_t,
				 ptr_d_agg,
				 lamb,
				 ptr_d_indx,
				 ptr_d_agg_p,
				 ptr_d_kout,
				 ptr_d_vout,
				 sorted,
				 stream);
}

template<typename ValsT,
	 typename IndexT>
__host__ __device__
size_t multi_col_group_by_max_sort(size_t         nrows,
				   size_t         ncols,
				   void* const*   d_cols,
				   int* const     d_gdf_t,
				   const ValsT*   ptr_d_agg,
				   IndexT*        ptr_d_indx,
				   ValsT*         ptr_d_agg_p,
				   IndexT*        ptr_d_kout,
				   ValsT*         ptr_d_vout,
				   bool           sorted = false,
				   cudaStream_t   stream = NULL)
{
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
    return (x>y?x:y);
  };

  using ReducerT = decltype(lamb);

  return multi_col_group_by_sort(nrows,
				 ncols,
				 d_cols,
				 d_gdf_t,
				 ptr_d_agg,
				 lamb,
				 ptr_d_indx,
				 ptr_d_agg_p,
				 ptr_d_kout,
				 ptr_d_vout,
				 sorted,
				 stream);
}


template<typename ValsT,
	 typename IndexT>
__host__ __device__
size_t multi_col_group_by_avg_sort(size_t         nrows,
				   size_t         ncols,
				   void* const*   d_cols,
				   int* const     d_gdf_t,
				   const ValsT*   ptr_d_agg,
				   IndexT*        ptr_d_indx,
				   IndexT*        ptr_d_cout,
				   ValsT*         ptr_d_agg_p,
				   IndexT*        ptr_d_kout,
				   ValsT*         ptr_d_vout,
				   bool           sorted = false,
				   cudaStream_t   stream = NULL)
{
  multi_col_group_by_count_sort(nrows,
				ncols,
				d_cols,
				d_gdf_t,
				ptr_d_indx,
				ptr_d_kout,
				ptr_d_cout,
				sorted,
				stream);

  size_t new_sz =  multi_col_group_by_sum_sort(nrows,
					       ncols,
					       d_cols,
					       d_gdf_t,
					       ptr_d_agg,
					       ptr_d_indx,
					       ptr_d_agg_p,
					       ptr_d_kout,
					       ptr_d_vout,
					       sorted,
					       stream);

  thrust::transform(thrust::cuda::par.on(stream),
		    ptr_d_cout, ptr_d_cout + nrows,
		    ptr_d_vout,
		    ptr_d_vout,
		    [] __host__ __device__ (IndexT n, ValsT sum){
		      return sum/static_cast<ValsT>(n);
		    });

  return new_sz;
}


