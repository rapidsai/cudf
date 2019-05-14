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

  LesserRTTI(void *const *cols,
             gdf_valid_type *const *valids,
             int *const types,
             int8_t *const asc_desc_flags,
             size_t sz,
             bool nulls_are_smallest) : columns_(cols),
                                        valids_(valids),
                                        rtti_(types),
                                        sz_(sz),
                                        asc_desc_flags_(asc_desc_flags),
                                        nulls_are_smallest_(nulls_are_smallest)
  {
  }

  /**
   * Should be used when you want to sort multiple columns using asc/desc flags for each column
   */
  __device__
  bool asc_desc_comparison(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);
      
      bool asc = true;
      if(asc_desc_flags_ != nullptr){
        asc = (asc_desc_flags_[col_index] == GDF_ORDER_ASC);
      }

      State state;
      if(asc){
        state = cudf::type_dispatcher(col_type, OpLess{},
                                      row1,
                                      row2,
                                      col_index,
                                      columns_);
      }else{
        state = cudf::type_dispatcher(col_type, OpGreater{},
                                      row1,
                                      row2,
                                      col_index,
                                      columns_);
      }

      switch( state )
      {
      case State::False:
        return false;
      case State::True:
        return true;
      case State::Undecided: // When doing a multi-column comparison, when you have an equals, your state is undecided and you need to check the next column.
        break;
      }
    }
    return false;
  }

  __device__
  bool asc_desc_comparison_with_nulls(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      bool asc = true;
      if(asc_desc_flags_ != nullptr){
        asc = (asc_desc_flags_[col_index] == GDF_ORDER_ASC);
      }

      State state;
      if(asc){
        state = cudf::type_dispatcher(col_type, OpLess_with_nulls{},
                                      row1,
                                      row2,
                                      col_index,
                                      columns_,
                                      valids_,
                                      nulls_are_smallest_);
      }else{
        state = cudf::type_dispatcher(col_type, OpGreater_with_nulls{},
                                      row1,
                                      row2,
                                      col_index,
                                      columns_,
                                      valids_,
                                      nulls_are_smallest_);
      }

      switch( state )
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

    __host__ __device__
  bool equal_with_nulls(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpEqual_with_nulls{},
                                          row1,
                                          row2,
                                          col_index,
                                          columns_,
                                          valids_);
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

  __device__
  bool less(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpLess{},
                                          row1,
                                          row2,
                                          col_index,
                                          columns_);
      switch( state )
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

    __device__
  bool less_with_nulls(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpLess_with_nulls{},
                                          row1,
                                          row2,
                                          col_index,
                                          columns_,
                                          valids_,
                                          nulls_are_smallest_);
      switch( state )
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


    __device__
  bool less_with_nulls_always_false(IndexT row1, IndexT row2) const
  {
    for(size_t col_index = 0; col_index < sz_; ++col_index)
    {
      gdf_dtype col_type = static_cast<gdf_dtype>(rtti_[col_index]);

      State state = cudf::type_dispatcher(col_type, OpLess_with_nulls_always_false{},
                                          row1,
                                          row2,
                                          col_index,
                                          columns_,
                                          valids_);
      switch( state )
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
  __device__
  static ColType at(int col_index,
	                  IndexT row,
		                const void* const * columns)
  {
    return (static_cast<const ColType*>(columns[col_index]))[row];
  }
  
  __device__
  static bool is_valid(int col_index,
                       IndexT row,
                       const gdf_valid_type* const * valids)
  {
    return gdf_is_valid(valids[col_index], row);
  }

private:
  enum class State {False = 0, True = 1, Undecided = 2};

  struct OpLess
  {
    template<typename ColType>
    __device__
    State operator() (IndexT row1, IndexT row2,
                      int col_index,
                      const void* const * columns)
    {
      const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
      
      if( res1 < res2 )
        return State::True;
      else if( res1 == res2 )
        return State::Undecided;
      else
	      return State::False;
    }
  };

  struct OpLess_with_nulls
  {
    template<typename ColType>
	  __device__
	  State operator() (IndexT row1, IndexT row2,
                      int col_index,
                      const void* const * columns,
                      const gdf_valid_type* const * valids,
                      bool nulls_are_smallest)
	  {
		  const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
		  const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
		  const bool isValid1 = LesserRTTI::is_valid(col_index, row1, valids);
		  const bool isValid2 = LesserRTTI::is_valid(col_index, row2, valids);

		  if (!isValid2 && !isValid1)
			  return State::Undecided;
		  else if( isValid1 && isValid2)
		  {
			  if( res1 < res2 )
				  return State::True;
			  else if( res1 == res2 )
				  return State::Undecided;
			  else
				  return State::False;
		  }
		  else if (!isValid1 && nulls_are_smallest)
			  return State::True;
	  	else if (!isValid2 && !nulls_are_smallest)
	  		return State::True;
		  else
			  return State::False;
	  }
  };

struct OpLess_with_nulls_always_false
  {
    template<typename ColType>
	  __device__
	  State operator() (IndexT row1, IndexT row2,
                      int col_index,
                      const void* const * columns,
                      const gdf_valid_type* const * valids)
	  {
		  const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
		  const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
		  const bool isValid1 = LesserRTTI::is_valid(col_index, row1, valids);
		  const bool isValid2 = LesserRTTI::is_valid(col_index, row2, valids);

		  if (!isValid2 || !isValid1)
			  return State::False;
		  else if( res1 < res2 )
        return State::True;
      else if( res1 == res2 )
        return State::Undecided;
      else
        return State::False;
	  }
  };

  struct OpGreater
  {
    template<typename ColType>
    __device__
    State operator() (IndexT row1, IndexT row2,
                      int col_index,
  		                const void* const * columns)
    {
      const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);

      if( res1 > res2 )
  	    return State::True;
      else if( res1 == res2 )
  	    return State::Undecided;
      else
  	    return State::False;
    }
  };

  struct OpGreater_with_nulls
  {
    template<typename ColType>
	  __device__
	  State operator() (IndexT row1, IndexT row2,
                      int col_index,
                      const void* const * columns,
                      const gdf_valid_type* const * valids,
                      bool nulls_are_smallest)
	  {
		  const ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
		  const ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
		  const bool isValid1 = LesserRTTI::is_valid(col_index, row1, valids);
		  const bool isValid2 = LesserRTTI::is_valid(col_index, row2, valids);

		  if (!isValid2 && !isValid1)
			  return State::Undecided;
		  else if( isValid1 && isValid2)
		  {
			  if( res1 > res2 )
				  return State::True;
			  else if( res1 == res2 )
				  return State::Undecided;
			  else
				  return State::False;
		  }
		  else if (!isValid1 && nulls_are_smallest)
			  return State::False;
	  	else if (!isValid2 && !nulls_are_smallest)
	  		return State::False;
		  else
			  return State::True;
	  }
  };

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

  struct OpEqual_with_nulls
  {
    template<typename ColType>
    __device__
    State operator() (IndexT row1, IndexT row2,
                       int col_index,
                      const void* const * columns,
                      const gdf_valid_type* const * valids)
    {
      ColType res1 = LesserRTTI::at<ColType>(col_index, row1, columns);
      ColType res2 = LesserRTTI::at<ColType>(col_index, row2, columns);
      bool isValid1 = LesserRTTI::is_valid(col_index, row1, valids);
		  bool isValid2 = LesserRTTI::is_valid(col_index, row2, valids);
      
      if (!isValid2 && !isValid1)
			  return State::Undecided;
		  else if( isValid1 && isValid2 && res1 == res2)
        return State::True;
      else
        return State::False;      
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
  const gdf_valid_type* const * valids_{nullptr};
  const int* const rtti_{nullptr};
  size_t sz_;
  const void* const * vals_{nullptr}; //for filtering
  const int8_t* const asc_desc_flags_{nullptr}; //array of 0 and 1 that allows us to know whether or not a column should be sorted ascending or descending
  bool nulls_are_smallest_{false};  // when sorting if there are nulls in the data if this is true, then they will be treated as the smallest value, otherwise they will be treated as the largest value
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


/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of columns, using type erasure and RTTI at
 * comparison operator level.
 * 
 * @param[in] d_cols Device array to ncols type erased columns
 * @param[in] d_valids Device array to ncols gdf_valid_type columns
 * @param[in] d_col_types Device array of runtime column types
 * @param[in] d_asc_desc Device array of column sort order types
 * @param[in] ncols # columns
 * @param[in] nrows # rows
 * @param[in] have_nulls Whether or not any column have null values
 * @param[in] nulls_are_smallest Whether or not nulls are smallest
 * @Param[in] null_as_largest_for_multisort. If set to true, nulls will always trigger less than equal to false
 * @param[in] stream CudaStream to work in
 * @param[out] d_indx Device array of re-ordered indices after sorting
 * @tparam IndexT The type of d_indx array 
 * 
 * @returns
 */
/* ----------------------------------------------------------------------------*/
template<typename IndexT>
void multi_col_sort(void* const *           d_cols,
                    gdf_valid_type* const * d_valids,
                    int*                    d_col_types,
                    int8_t*                 d_asc_desc,
                    size_t                  ncols,
                    size_t                  nrows,
                    bool                    have_nulls,
                    IndexT*                 d_indx,
                    bool                    nulls_are_smallest = false,
                    bool                    null_as_largest_for_multisort = false,
                    cudaStream_t            stream = NULL)
{
  //cannot use counting_iterator 2 reasons:
  //(1.) need to return a container result;
  //(2.) that container must be mutable;
  thrust::sequence(rmm::exec_policy(stream)->on(stream), d_indx, d_indx+nrows, 0);
  
  LesserRTTI<IndexT> comp(d_cols, d_valids, d_col_types, d_asc_desc, ncols, nulls_are_smallest);
  if (d_valids != nullptr && have_nulls) {
    if (null_as_largest_for_multisort){ //Pandas 
      thrust::sort(rmm::exec_policy(stream)->on(stream),
                  d_indx, d_indx+nrows,
                  [comp] __device__ (IndexT i1, IndexT i2){
                      return comp.less_with_nulls_always_false(i1, i2);
                  });
    } else { // SQL
       thrust::sort(rmm::exec_policy(stream)->on(stream),
                  d_indx, d_indx+nrows,
                  [comp] __device__ (IndexT i1, IndexT i2){
                      return comp.asc_desc_comparison_with_nulls(i1, i2);
                  });
    }
  }
  else { // no hay nulos!
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                d_indx, d_indx+nrows,
                [comp] __device__ (IndexT i1, IndexT i2) {
                  return comp.asc_desc_comparison(i1, i2);
                });
  }
}
