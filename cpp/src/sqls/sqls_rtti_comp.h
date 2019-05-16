/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//Type-erasure C-style interface for Multi-column Filter, Order-By, and Group-By functionality

#pragma once

#include <cassert>
#include <iterator>

#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/gather.h>

#include "rmm/thrust_rmm_allocator.h"

#include <cstdint>

#include "utilities/cudf_utils.h"
#include "utilities/type_dispatcher.hpp"
#include "bitmask/legacy_bitmask.hpp"

#include "table/device_table_row_operators.cuh"

template<typename IndexT>
struct LesserRTTI
{
  LesserRTTI(void *const *cols,
             int *const types,
             size_t sz) : columns_(cols),
                          rtti_(types),
                          sz_(sz)
  {
  }

  LesserRTTI(void *const *cols,
             int *const types,
             size_t sz,
             const void *const *vals) : columns_(cols),
                                        rtti_(types),
                                        sz_(sz),
                                        vals_(vals)
  {
  }

 
   __device__
  bool equal(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpEqual{},
                                          row1,
                                          row2,
                                          col_index,
                                          columns_);
      switch( state )
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

  __device__
  bool equal_v(size_t row1) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpEqualV{},
                                          vals_,
                                          row1,
                                          col_index,
                                          columns_);
      switch( state )
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

  template<typename ColType>
  __device__
  static ColType at(int col_index,
	                  IndexT row,
		                const void* const * columns)
  {
    return (static_cast<const ColType*>(columns[col_index]))[row];
  }
  
private:
  enum class State {False = 0, True = 1, Undecided = 2};

  struct OpEqual
  {
    template<typename ColType>
    __device__
    State operator() (IndexT row1, IndexT row2,
                      int col_index,
		                  const void* const * columns)
    {
      const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
      
      if( res1 != res2 )
        return State::False;
      else
        return State::Undecided;
    }
  };

  struct OpEqualV
  {
    template<typename ColType>
    __device__
    State operator() (const void* const * vals,
                      IndexT row,
                      int col_index,
                      const void* const * columns)
    {
      const ColType res1 = LesserRTTI::at<ColType>(col_index, row, columns);
      const ColType res2 = LesserRTTI::at<ColType>(col_index, 0, vals);
      
      if( res1 != res2 )
        return State::False;
      else
        return State::Undecided;
    }
  };

  const void* const * columns_{nullptr};
  const int* const rtti_{nullptr};
  size_t sz_;
  const void* const * vals_{nullptr}; //for filtering  
};

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
size_t multi_col_filter(size_t nrows,
                        size_t ncols,
                        void* const* d_cols,
                        int* const  d_gdf_t,
                        void* const* d_vals,
                        IndexT* ptr_d_flt_indx,
                        cudaStream_t stream = NULL)
{
  LesserRTTI<size_t> f(d_cols, d_gdf_t, ncols, d_vals);//size_t, not IndexT, because of counting_iterator below;
  

  //actual filtering happens here:
  //
  auto ret_iter_last =
    thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_t>(0), 
                    thrust::make_counting_iterator<size_t>(nrows),
                    ptr_d_flt_indx,
                    [f] __device__ (size_t indx) {
                      return f.equal_v(indx);
                    });

  size_t new_sz = thrust::distance(ptr_d_flt_indx, ret_iter_last);
  
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
template<typename IndexT,
         typename CountT = IndexT>
size_t
multi_col_group_by_count_sort(size_t         nrows,
                              size_t         ncols,
                              void* const*   d_cols,
                              int* const     d_gdf_t,
                              IndexT*        ptr_d_indx,
                              IndexT*        ptr_d_kout,
                              CountT*        ptr_d_vout,
                              bool           sorted = false,
                              bool           distinct = false,
                              cudaStream_t   stream = NULL)
{
  if( !sorted )
    multi_col_sort(d_cols, nullptr, d_gdf_t, nullptr, ncols, nrows, false, ptr_d_indx, false, false, stream);

  LesserRTTI<IndexT> f(d_cols, d_gdf_t, ncols);


  thrust::pair<IndexT*, CountT*> ret =
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          ptr_d_indx, ptr_d_indx+nrows,
                          thrust::make_constant_iterator<CountT>(1),
                          ptr_d_kout,
                          ptr_d_vout,
                          [f] __host__ __device__(IndexT key1, IndexT key2){
                            return f.equal(key1, key2);
                          });
			    
  size_t new_sz = thrust::distance(ptr_d_vout, ret.second);

  //COUNT(*) for each distinct entry gets collected in d_vout;
  //DISTINCT COUNT(*) is just: thrust::distance(d_kout.begin(), ret.first)

  if( distinct )
  {
    CountT distinct_count = static_cast<CountT>(new_sz);
    cudaMemcpy(ptr_d_vout, &distinct_count, sizeof(CountT), cudaMemcpyHostToDevice);
    new_sz = 1;
  }
  
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
    multi_col_sort(d_cols, nullptr, d_gdf_t, nullptr, ncols, nrows, false, ptr_d_indx, false, false, stream);

  
  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 ptr_d_indx, ptr_d_indx + nrows,  //map[i]
  		 ptr_d_agg,                    //source[i]
  		 ptr_d_agg_p);                 //source[map[i]]

  LesserRTTI<IndexT> f(d_cols, d_gdf_t, ncols);
  
  thrust::pair<IndexT*, ValsT*> ret =
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          ptr_d_indx, ptr_d_indx + nrows,
                          ptr_d_agg_p,
                          ptr_d_kout,
                          ptr_d_vout,
                          [f] __device__(IndexT key1, IndexT key2) {
                              return f.equal(key1, key2);
                          },
                          fctr);

  size_t new_sz = thrust::distance(ptr_d_vout, ret.second);
  return new_sz;
}

template<typename ValsT,
	       typename IndexT>
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
  auto lamb = [] __device__ (ValsT x, ValsT y) {
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
  auto lamb = [] __device__ (ValsT x, ValsT y) {
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
  auto lamb = [] __device__ (ValsT x, ValsT y) {
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


  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    ptr_d_cout, ptr_d_cout + nrows,
                    ptr_d_vout,
                    ptr_d_vout,
                    [] __device__ (IndexT n, ValsT sum) {
                      return sum/static_cast<ValsT>(n);
                    });

  return new_sz;
}

