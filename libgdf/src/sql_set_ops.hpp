/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

//C++ style of interface for Multi-column Filter, Order-By, and Group-By functionality

# pragma once

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
#include "thrust_rmm_allocator.h"

//Potential PROBLEM: thrust::tuple limited to 10 type args
//
template<typename ...Ts>
using Tuple = thrust::tuple<Ts...>;

// thrust::device_vector set to use rmmAlloc and rmmFree.
template <typename T>
using Vector = thrust::device_vector<T, rmm_allocator<T>>;


template<typename T, typename Allocator, template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}

template<typename T,
	 typename Allocator,
	 template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, typename Vector<T, Allocator>::const_iterator pos, std::ostream& os)
{ 
  thrust::copy(v.begin(), pos, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T,
	 typename Allocator,
	 template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, size_t n, std::ostream& os)
{ 
  thrust::copy_n(v.begin(), n, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}


//Multi-column Ordering policies: via Sort, or Hash (for now)
//
enum class ColumnOrderingPolicy { Sort = 0, Hash };


//RTI for "opaque" tuple comparison fctr;
//("opaque" as opposed to "transparent"
// variadic pack expansion, which doesn't
// compile with device functions) 
//
//generic-case template:
//Note: need two indices as template args
//in order to avoid reverting the tuple
//for proper comparison order
//
template<typename Tuple, size_t i, size_t imax>
struct LesserIndex
{
  //for sort:
  //
  __host__ __device__ 
  static bool less(const Tuple& left, const Tuple& right, int i1, int i2)
  {
    if( thrust::get<imax-i>(left)[i1] < thrust::get<imax-i>(right)[i2] )
      return true;
    else if( thrust::get<imax-i>(left)[i1] == thrust::get<imax-i>(right)[i2] )
      {
	return LesserIndex<Tuple, i-1, imax>::less(left, right, i1, i2);
      }
    else
      return false;
  }

  //for unique_by_key, reduce_by_key:
  //
  __host__ __device__ 
  static bool equal(const Tuple& left, const Tuple& right, int i1, int i2)
  {
    if( thrust::get<imax-i>(left)[i1] == thrust::get<imax-i>(right)[i2] )
      return LesserIndex<Tuple, i-1, imax>::equal(left, right, i1, i2);
    else
      return false;
  }

  //for filter:
  //
  template<typename TplVals>
   __host__ __device__ 
  static bool equal(const Tuple& left, const TplVals& right, int i1)
  {
    if( thrust::get<imax-i>(left)[i1] == thrust::get<imax-i>(right) )
      return LesserIndex<Tuple, i-1, imax>::equal(left, right, i1);
    else
      return false;
  }
};

//Partial specialization for bottom of RTI recursion:
//
template<typename Tuple, size_t imax>
struct LesserIndex<Tuple, 0, imax>
{
  //for sort:
  //
  __host__ __device__ 
  static bool less(const Tuple& left, const Tuple& right, int i1, int i2)
  {
    if( thrust::get<imax>(left)[i1] < thrust::get<imax>(right)[i2] )
      return true;
    else
      return false;
  }

  //for unique_by_key, reduce_by_key:
  //
  __host__ __device__ 
  static bool equal(const Tuple& left, const Tuple& right, int i1, int i2)
  {
    return thrust::get<imax>(left)[i1] == thrust::get<imax>(right)[i2];
  }

  //for filter:
  //
  template<typename TplVals>
  __host__ __device__ 
  static bool equal(const Tuple& left, const TplVals& right, int i1)
  {
    return thrust::get<imax>(left)[i1] == thrust::get<imax>(right);
  }
};

//RTI for "opaque" tuple of pairs comparison fctr;
//("opaque" as opposed to "transparent"
// variadic pack expansion, which doesn't
// compile with device functions) 
//
//generic-case template:
//Note: need two indices as template args
//in order to avoid reverting the tuple
//for proper comparison order
//
template<typename TuplePairPtr, size_t i, size_t imax>
struct PairsComparer
{
  //version with tuple of pair of pointers:
  //one pointer to an array,
  //the other to just one value (of same type)
  //(for filter)
  //
  __host__ __device__ 
  static bool equal(const TuplePairPtr& tplpairptrs, size_t i1)
  {
    if( thrust::get<imax-i>(tplpairptrs)[i1] ==
  	*(thrust::get<imax-i+1>(tplpairptrs)) )
      return PairsComparer<TuplePairPtr, i-2, imax>::equal(tplpairptrs, i1);
    else
      return false;
  }
};

//Partial specialization for bottom of RTI recursion:
//
template<typename TuplePairPtr, size_t imax>
struct PairsComparer<TuplePairPtr, 1, imax>
{
  //version with tuple of pair of pointers:
  //one pointer to an array,
  //the other to just one value (of same type)
  //(for filter)
  //
  __host__ __device__ 
  static bool equal(const TuplePairPtr& tplpairptrs, size_t i1)
  {
    return (thrust::get<imax-1>(tplpairptrs)[i1] == *(thrust::get<imax>(tplpairptrs)));
  }
};

//###########################################################################
//#                          Multi-column ORDER-BY:                         #
//###########################################################################
//args:
//Input:
// sz    =  # rows;
// tv1   = table as a tuple of columns (pointers to device arrays)
//stream = cudaStream to work in;
//
//Output:
// v = vector of indices re-ordered after sorting;
//
template<typename TplPtrs,
	 typename IndexT>
__host__ __device__
void multi_col_order_by(size_t sz,
			const TplPtrs& tv1,
			IndexT* ptr_d_v,
			cudaStream_t stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  thrust::sequence(thrust::cuda::par(allocator).on(stream), ptr_d_v, ptr_d_v+sz, 0);//cannot use counting_iterator
  //                                          2 reasons:
  //(1.) need to return a container result;
  //(2.) that container must be mutable;
  
  thrust::sort(thrust::cuda::par(allocator).on(stream),
		 ptr_d_v, ptr_d_v+sz,
		 [tv1] __host__ __device__ (int i1, int i2){
		   //C+11 variadic pack expansion doesn't work with
		   //__host__ __device__ code:
		   //
		   //Tuple<Ts...> t1(thrust::get<Is>(tv1)[i1]...);
		   //Tuple<Ts...> t2(thrust::get<Is>(tv1)[i2]...);		   
		   //return (t1 < t2);

		   //use RTI (C++03 style) instead:
		   //
		   return LesserIndex<TplPtrs, thrust::tuple_size<TplPtrs>::value-1, thrust::tuple_size<TplPtrs>::value-1>::less(tv1, tv1, i1, i2);
		 });
}

template<typename TplPtrs,
	 typename VectorIndexT>
void multi_col_order_by(size_t sz,
			const TplPtrs& tv1,
			VectorIndexT& v,
			cudaStream_t stream = NULL)
{
  ///VectorT<int> v(sz, 0);
  assert( v.size() >= sz );
  
  multi_col_order_by(sz, tv1, v.data().get(), stream);
}


//###########################################################################
//#                          Multi-column Filter:                           #
//###########################################################################
//Filter tuple of device vectors ("table")
//on given tuple of values;
//Input:
//nrows      = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//tvals      = tuple of values to filter against;
//stream     = cudaStream to work in;
//Output:
//d_flt_indx = device_vector of indx for which a match exists (no holes in array!);
//new_sz     = new #rows after filtering;
//Return:
//ret                       = true if filtered result is non-empty, false otherwise;
//
template<typename TplPtrs,
	 typename TplVals,
	 typename IndexT>
__host__ __device__
size_t multi_col_filter(size_t nrows,
			const TplPtrs& tptrs,
			const TplVals& tvals,
			IndexT* ptr_d_flt_indx,
			cudaStream_t stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  //actual filtering happens here:
  //
  auto ret_iter_last =
    thrust::copy_if(thrust::cuda::par(allocator).on(stream),
		    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(nrows),
		    ptr_d_flt_indx,
		    [tptrs, tvals] __host__ __device__ (size_t indx){
		      return LesserIndex<TplPtrs, thrust::tuple_size<TplPtrs>::value-1, thrust::tuple_size<TplPtrs>::value-1>::equal(tptrs, tvals, indx);
		    });

  size_t new_sz = thrust::distance(ptr_d_flt_indx,ret_iter_last);
  
  return new_sz;
}

//version with tuple of pairs of pointers
//(adjacent pointers of same type; Example:
// thrust::tuple<T1*, T1*, T2*, T2*,...,Tn*, Tn*>)
//
//CAVEAT: all adjacent pairs of
//        pointers must reside on __device__!
//
//(All must be passed by pointers because of type erasure in cudf
// and, consequently, the NestedIfThenElser expects uniform type erasure)
//
template<typename TplPairsPtrs,
	 typename IndexT>
__host__ __device__
size_t multi_col_filter(size_t nrows,
			const TplPairsPtrs& tptrs,
			IndexT* ptr_d_flt_indx,
			cudaStream_t stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  //actual filtering happens here:
  //
  auto ret_iter_last =
    thrust::copy_if(thrust::cuda::par(allocator).on(stream),
		    thrust::make_counting_iterator<IndexT>(0), thrust::make_counting_iterator<IndexT>(nrows),
		    ptr_d_flt_indx,
		    [tptrs] __host__ __device__ (IndexT indx){
		      return PairsComparer<TplPairsPtrs, thrust::tuple_size<TplPairsPtrs>::value-1, thrust::tuple_size<TplPairsPtrs>::value-1>::equal(tptrs, indx);
		    });

  size_t new_sz = thrust::distance(ptr_d_flt_indx,ret_iter_last);
  
  return new_sz;
}

template<typename TplPtrs,
	 typename TplVals,
	 typename VectorIndexT>
size_t multi_col_filter(size_t nrows,
			const TplPtrs& tptrs,
			const TplVals& tvals,
			VectorIndexT& d_flt_indx,
			cudaStream_t stream = NULL)
{
  assert( d_flt_indx.size() >= nrows );
  
  size_t new_sz = multi_col_filter(nrows, tptrs, tvals ,d_flt_indx.data().get(), stream);

  assert( new_sz <= nrows );
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
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allocations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (COUNT-ed) as a result of group-by; 
//d_items    = device_vector of items corresponding to indices in d_flt_indx;
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename TplPtrs,
	 typename IndexT>
__host__ __device__
thrust::pair<IndexT*, IndexT*>
multi_col_group_by_count_via_sort(size_t sz,
				  const TplPtrs& tptrs,
				  IndexT* ptr_d_indx,
				  IndexT* ptr_d_kout,
				  IndexT* ptr_d_vout,
				  bool sorted = false,
				  cudaStream_t stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  if( !sorted )
    multi_col_order_by(sz, tptrs, ptr_d_indx, stream);

  thrust::pair<IndexT*, IndexT*> ret =
    thrust::reduce_by_key(thrust::cuda::par(allocator).on(stream),
			  ptr_d_indx, ptr_d_indx+sz,
			  thrust::make_constant_iterator(1),
			  ptr_d_kout,
			  ptr_d_vout,
			  [tptrs] __host__ __device__(int key1, int key2){
			    return LesserIndex<TplPtrs, thrust::tuple_size<TplPtrs>::value-1, thrust::tuple_size<TplPtrs>::value-1>::equal(tptrs, tptrs, key1, key2);
			  });

  //COUNT(*) for each distinct entry gets collected in d_vout;
  //DISTINCT COUNT(*) is just: thrust::distance(d_kout.begin(), ret.first)
			    
  return ret;
}

//generic (purposely do-nothing) implementation
//
template<typename TplPtrs,
	 typename IndexT,
	 ColumnOrderingPolicy orderp>
struct MultiColGroupByCount
{
  //this should NEVER get called
  //(hence the assert(false);)
  //
  __host__ __device__
  thrust::pair<IndexT*, IndexT*>
  operator() (size_t sz,
	      const TplPtrs& tptrs,
	      IndexT* ptr_d_indx,
	      IndexT* ptr_d_kout,
	      IndexT* ptr_d_vout,
	      bool sorted = false,
	      cudaStream_t stream = NULL)
  {
    assert( false );
    return thrust::make_pair(nullptr, nullptr);
  }
};

//partial specialization for ColumnOrderingPolicy::Sort
//
template<typename TplPtrs,
	 typename IndexT>
struct MultiColGroupByCount<TplPtrs, IndexT, ColumnOrderingPolicy::Sort>
{
  __host__ __device__
  thrust::pair<IndexT*, IndexT*> operator() (size_t sz,
					     const TplPtrs& tptrs,
					     IndexT* ptr_d_indx,
					     IndexT* ptr_d_kout,
					     IndexT* ptr_d_vout,
					     bool sorted = false,
					     cudaStream_t stream = NULL)
  {
    return multi_col_group_by_count_via_sort(sz, tptrs, ptr_d_indx, ptr_d_kout, ptr_d_vout, sorted, stream);
  }
};

template<typename TplPtrs,
	 typename VectorIndexT,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorIndexT::iterator>
multi_col_group_by_count(size_t sz,
			 const TplPtrs& tptrs,
			 VectorIndexT& d_indx,
			 VectorIndexT& d_kout,
			 VectorIndexT& d_vout,
			 bool sorted = false,
			 cudaStream_t stream = NULL)
{
  assert( d_indx.size() >= sz );
  assert( d_kout.size() >= sz );
  assert( d_vout.size() >= sz );

  using IndexT = typename VectorIndexT::value_type;

  MultiColGroupByCount<TplPtrs, IndexT, orderp> mcgbc;

  auto ret = mcgbc(sz, tptrs, d_indx.data().get(), d_kout.data().get(), d_vout.data().get(), sorted, stream);

  typename VectorIndexT::iterator fst = d_kout.begin();
  typename VectorIndexT::iterator snd = d_vout.begin();
  thrust::advance(fst, ret.first - d_kout.data().get());
  thrust::advance(snd, ret.second - d_vout.data().get());

  return thrust::make_pair(fst, snd);
}





//group-by is a reduce_by_key
//
//not only COUNT(*) makes sense with multicolumn:
//one can specify a multi-column GROUP-BY criterion
//and one specifc column to aggregate on;
//
//Input:
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//d_agg      = column (device vector) to get aggregated;
//fctr       = functor to perform the aggregation <Tagg(Tagg, Tagg)>       
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_agg_p    = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (counted) as a result of group-by; 
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename ValsT,
	 typename IndexT,
	 typename TplPtrs,
	 typename Reducer>
__host__ __device__
thrust::pair<IndexT*, ValsT*>
      multi_col_group_by_via_sort(size_t         sz,
				  const TplPtrs& tptrs,
				  const ValsT*   ptr_d_agg,
				  Reducer        fctr,
				  IndexT*        ptr_d_indx,
				  ValsT*         ptr_d_agg_p,
				  IndexT*        ptr_d_kout,
				  ValsT*         ptr_d_vout,
				  bool           sorted = false,
				  cudaStream_t   stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  if( !sorted )
    multi_col_order_by(sz, tptrs, ptr_d_indx, stream);
  
  thrust::gather(thrust::cuda::par(allocator).on(stream),
                 ptr_d_indx, ptr_d_indx + sz,  //map[i]
  		 ptr_d_agg,                    //source[i]
  		 ptr_d_agg_p);                 //source[map[i]]

  thrust::pair<IndexT*, ValsT*> ret =
    thrust::reduce_by_key(thrust::cuda::par(allocator).on(stream),
  			    ptr_d_indx, ptr_d_indx + sz,
  			    ptr_d_agg_p,
  			    ptr_d_kout,
  			    ptr_d_vout,
  			    [tptrs] __host__ __device__(int key1, int key2){
  			      return LesserIndex<TplPtrs, thrust::tuple_size<TplPtrs>::value-1, thrust::tuple_size<TplPtrs>::value-1>::equal(tptrs, tptrs, key1, key2);
  			    },
  			    fctr);
			    
  return ret;
}

//SUM
//
template<typename ValsT,
	 typename IndexT,
	 typename TplPtrs>
__host__ __device__
thrust::pair<IndexT*, ValsT*>
      multi_col_group_by_sum_sort(size_t         sz,
				  const TplPtrs& tptrs,
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
  
  return multi_col_group_by_via_sort<ValsT, IndexT, TplPtrs, ReducerT>(sz,
								       tptrs,
								       ptr_d_agg,
								       lamb,
								       ptr_d_indx,
								       ptr_d_agg_p,
								       ptr_d_kout,
								       ptr_d_vout,
								       sorted,
								       stream);
}

//MIN
//
template<typename ValsT,
	 typename IndexT,
	 typename TplPtrs>
__host__ __device__
thrust::pair<IndexT*, ValsT*>
      multi_col_group_by_min_sort(size_t         sz,
				  const TplPtrs& tptrs,
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
  
  return multi_col_group_by_via_sort<ValsT, IndexT, TplPtrs, ReducerT>(sz,
								       tptrs,
								       ptr_d_agg,
								       lamb,
								       ptr_d_indx,
								       ptr_d_agg_p,
								       ptr_d_kout,
								       ptr_d_vout,
								       sorted,
								       stream);
}

//MAX
//
template<typename ValsT,
	 typename IndexT,
	 typename TplPtrs>
__host__ __device__
thrust::pair<IndexT*, ValsT*>
      multi_col_group_by_max_sort(size_t         sz,
				  const TplPtrs& tptrs,
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
  
  return multi_col_group_by_via_sort<ValsT, IndexT, TplPtrs, ReducerT>(sz,
								       tptrs,
								       ptr_d_agg,
								       lamb,
								       ptr_d_indx,
								       ptr_d_agg_p,
								       ptr_d_kout,
								       ptr_d_vout,
								       sorted,
								       stream);
}

//AVERAGE
//
template<typename ValsT,
	 typename IndexT,
	 typename TplPtrs>
__host__ __device__
thrust::pair<IndexT*, ValsT*>
      multi_col_group_by_avg_sort(size_t         sz,
				  const TplPtrs& tptrs,
				  const ValsT*   ptr_d_agg,
				  IndexT*        ptr_d_indx,
				  IndexT*        ptr_d_cout,
				  ValsT*         ptr_d_agg_p,
				  IndexT*        ptr_d_kout,
				  ValsT*         ptr_d_vout,
				  bool           sorted = false,
				  cudaStream_t   stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  auto pair_count = multi_col_group_by_count_via_sort(sz,
						      tptrs,
						      ptr_d_indx,
						      ptr_d_kout,
						      ptr_d_cout,
						      sorted,
						      stream);
  
  auto pair_sum = multi_col_group_by_sum_sort(sz,
					      tptrs,
					      ptr_d_agg,
					      ptr_d_indx,
					      ptr_d_agg_p,
					      ptr_d_kout,
					      ptr_d_vout,
					      sorted,
					      stream);

  thrust::transform(thrust::cuda::par(allocator).on(stream),
		    ptr_d_cout, ptr_d_cout + sz,
		    ptr_d_vout,
		    ptr_d_vout,
		    [] __host__ __device__ (IndexT n, ValsT sum){
		      return sum/static_cast<ValsT>(n);
		    });

  return pair_sum;
}

//generic (purposely do-nothing) implementation
//
template<typename TplPtrs,
	 typename IndexT,
	 typename ValsT,
	 typename Reducer,
	 ColumnOrderingPolicy orderp>
struct MultiColGroupBy
{
  //this should NEVER get called
  //(hence the assert(false);)
  //
  __host__ __device__
  thrust::pair<IndexT*, ValsT*>
  operator() (size_t         sz,
	      const TplPtrs& tptrs,
	      const ValsT*   ptr_d_agg,
	      Reducer        fctr,
	      IndexT*        ptr_d_indx,
	      ValsT*         ptr_d_agg_p,
	      IndexT*        ptr_d_kout,
	      ValsT*         ptr_d_vout,
	      cudaStream_t   stream = NULL)
  {
    assert( false );
    return thrust::make_pair(nullptr, nullptr);
  }
};

//partial specialization for ColumnOrderingPolicy::Sort
//
template<typename TplPtrs,
	 typename IndexT,
	 typename ValsT,
	 typename Reducer>
struct MultiColGroupBy<TplPtrs, IndexT, ValsT, Reducer, ColumnOrderingPolicy::Sort>
{
  __host__ __device__
  thrust::pair<IndexT*, ValsT*>
  operator() (size_t         sz,
	      const TplPtrs& tptrs,
	      const ValsT*   ptr_d_agg,
	      Reducer        fctr,
	      IndexT*        ptr_d_indx,
	      ValsT*         ptr_d_agg_p,
	      IndexT*        ptr_d_kout,
	      ValsT*         ptr_d_vout,
	      bool           sorted = false,
	      cudaStream_t   stream = NULL)
  {
    return multi_col_group_by_via_sort(sz,
				       tptrs,
				       ptr_d_agg,
				       fctr,
				       ptr_d_indx,
				       ptr_d_agg_p,
				       ptr_d_kout,
				       ptr_d_vout,
				       sorted,
				       stream);
  }
};

template<typename VectorValsT,
	 typename VectorIndexT,
	 typename TplPtrs,
	 typename Reducer,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator>
      multi_col_group_by(size_t sz,
			 const TplPtrs& tptrs,
			 const VectorValsT& d_agg,
			 Reducer fctr,
			 VectorIndexT& d_indx,
			 VectorValsT& d_agg_p,
			 VectorIndexT& d_kout,
			 VectorValsT& d_vout,
			 bool sorted = false,
			 cudaStream_t stream = NULL)
{
  

  assert( d_indx.size() >= sz );
  assert( d_kout.size() >= sz );
  assert( d_vout.size() >= sz );
  assert( d_agg_p.size() >= sz );

  // multi_col_order_by(sz, tptrs, d_indx, stream);
  
  // rmm_temp_allocator allocator(stream);
  // thrust::gather(thrust::cuda::par(allocator).on(stream),
  //                d_indx.begin(), d_indx.end(), //map[i]
  // 		 d_agg.begin(),                //source[i]
  // 		 d_agg_p.begin());             //source[map[i]]

  // thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator> ret =
  //   thrust::reduce_by_key(thrust::cuda::par(allocator).on(stream),
  // 			    d_indx.begin(), d_indx.end(),
  // 			    d_agg_p.begin(),
  // 			    d_kout.begin(),
  // 			    d_vout.begin(),
  // 			    [tptrs] __host__ __device__(int key1, int key2){
  // 			      return LesserIndex<TplPtrs, thrust::tuple_size<TplPtrs>::value-1, thrust::tuple_size<TplPtrs>::value-1>::equal(tptrs, tptrs, key1, key2);
  // 			    },
  // 			    fctr);
  // return ret;

  //DISTINCT AGG(*) does NOT make sense for any AGG other than COUNT

  using IndexT = typename VectorIndexT::value_type;
  using ValsT  = typename VectorValsT::value_type;
  
  MultiColGroupBy<TplPtrs, IndexT, ValsT, Reducer, orderp> mcgb;
  auto ret = mcgb(sz, tptrs, d_agg.data().get(), fctr, d_indx.data().get(), d_agg_p.data().get(), d_kout.data().get(), d_vout.data().get(), sorted, stream);

  typename VectorIndexT::iterator fst = d_kout.begin();
  typename VectorValsT::iterator  snd = d_vout.begin();
  thrust::advance(fst, ret.first - d_kout.data().get());
  thrust::advance(snd, ret.second - d_vout.data().get());

  return thrust::make_pair(fst, snd);
}

//Multi-column group-by SUM:
//
//Input:
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//d_agg      = column (device vector) to get aggregated;       
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_agg_p    = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (SUM-ed) as a result of group-by; 
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename VectorValsT,
	 typename VectorIndexT,
	 typename TplPtrs,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator>
      multi_col_group_by_sum(size_t sz,
			     const TplPtrs& tptrs,
			     const VectorValsT& d_agg,
			     VectorIndexT& d_indx,
			     VectorValsT& d_agg_p,
			     VectorIndexT& d_kout,
			     VectorValsT& d_vout,
			     bool sorted = false,
			     cudaStream_t stream = NULL)
{
  using ValsT  = typename VectorValsT::value_type;
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
			      return x+y;
  };

  using ReducerT = decltype(lamb);
  
  return multi_col_group_by<VectorValsT, VectorIndexT, TplPtrs, ReducerT, orderp>(sz, tptrs, d_agg,
			    lamb,
			    d_indx, d_agg_p, d_kout, d_vout, sorted, stream);
}

//Multi-column group-by MIN:
//
//Input:
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//d_agg      = column (device vector) to get aggregated;       
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_agg_p    = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (MINIMIZ-ed) as a result of group-by; 
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename VectorValsT,
	 typename VectorIndexT,
	 typename TplPtrs,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator>
      multi_col_group_by_min(size_t sz,
			     const TplPtrs& tptrs,
			     const VectorValsT& d_agg,
			     VectorIndexT& d_indx,
			     VectorValsT& d_agg_p,
			     VectorIndexT& d_kout,
			     VectorValsT& d_vout,
			     bool sorted = false,
			     cudaStream_t stream = NULL)
{
  using ValsT  = typename VectorValsT::value_type;
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
    return (x<y?x:y);
  };

  using ReducerT = decltype(lamb);
  
  return multi_col_group_by<VectorValsT, VectorIndexT, TplPtrs, ReducerT, orderp>(sz, tptrs, d_agg,
			    lamb,
			    d_indx, d_agg_p, d_kout, d_vout, sorted, stream);
}

//Multi-column group-by MAX:
//
//Input:
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//d_agg      = column (device vector) to get aggregated;       
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_agg_p    = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (MXIMIZ-ed) as a result of group-by; 
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename VectorValsT,
	 typename VectorIndexT,
	 typename TplPtrs,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator>
      multi_col_group_by_max(size_t sz,
			     const TplPtrs& tptrs,
			     const VectorValsT& d_agg,
			     VectorIndexT& d_indx,
			     VectorValsT& d_agg_p,
			     VectorIndexT& d_kout,
			     VectorValsT& d_vout,
			     bool sorted = false,
			     cudaStream_t stream = NULL)
{
  using ValsT  = typename VectorValsT::value_type;
  auto lamb = [] __host__ __device__ (ValsT x, ValsT y){
    return (x>y?x:y);
  };

  using ReducerT = decltype(lamb);
  
  return multi_col_group_by<VectorValsT, VectorIndexT, TplPtrs, ReducerT, orderp>(sz, tptrs, d_agg,
			    lamb,
			    d_indx, d_agg_p, d_kout, d_vout, sorted, stream);
}

//Multi-column group-by AVERAGE:
//
//Input:
//sz         = # rows
//tptrs      = table as a tuple of columns (pointers to device arrays);
//d_agg      = column (device vector) to get aggregated;       
//stream     = cudaStream to work in;
//Output:
//d_indx     = reordering of indices after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_cout     = (COUNT-ed) values as a result of group-by;
//d_agg_p    = reordering of d_agg after sorting;
//             (passed as argument to avoid allcoations inside the stream)
//d_kout     = indices of rows after group by;
//d_vout     = aggregated values (AVERAGE-d) as a result of group-by; 
//Return:
//ret        = pair of iterators into (d_kout, d_vout), respectively;
//
template<typename VectorValsT,
	 typename VectorIndexT,
	 typename TplPtrs,
	 ColumnOrderingPolicy orderp = ColumnOrderingPolicy::Sort>
thrust::pair<typename VectorIndexT::iterator, typename VectorValsT::iterator>
      multi_col_group_by_avg(size_t sz,
			     const TplPtrs& tptrs,
			     const VectorValsT& d_agg,
			     VectorIndexT& d_indx,
			     VectorIndexT& d_cout,
			     VectorValsT& d_agg_p,
			     VectorIndexT& d_kout,
			     VectorValsT& d_vout,
			     bool sorted = false,
			     cudaStream_t stream = NULL)
{
  rmm_temp_allocator allocator(stream);

  using IndexT = typename VectorIndexT::value_type;
  using ValsT  = typename VectorValsT::value_type;
  
  auto pair_count = multi_col_group_by_count<TplPtrs, VectorIndexT, orderp>(sz, tptrs, d_indx, d_kout, d_cout, sorted, stream);
  
  auto pair_sum = multi_col_group_by_sum<VectorValsT, VectorIndexT, TplPtrs, orderp>(sz, tptrs, d_agg, d_indx, d_agg_p, d_kout, d_vout, sorted, stream);

  thrust::transform(thrust::cuda::par(allocator).on(stream),
		    d_cout.begin(), d_cout.end(),
		    d_vout.begin(),
		    d_vout.begin(),
		    [] __host__ __device__ (IndexT n, ValsT sum){
		      return sum/static_cast<ValsT>(n);
		    });

  return pair_sum;
}
