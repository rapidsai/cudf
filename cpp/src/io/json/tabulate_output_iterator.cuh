/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

// Tabulate Output iterator
#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace cudf {
namespace detail {

// Proxy reference that calls BinaryFunction with index value and the rhs of assignment operator
template <typename BinaryFunction, typename IndexT>
class tabulate_output_iterator_proxy {
 public:
  __host__ __device__ tabulate_output_iterator_proxy(const IndexT index, BinaryFunction fun)
    : index(index), fun(fun)
  {
  }
  template <typename T>
  __host__ __device__ tabulate_output_iterator_proxy operator=(const T& rhs_value)
  {
    fun(index, rhs_value);
    return *this;
  }

 private:
  IndexT index;
  BinaryFunction fun;
};

/**
 * @brief Transform output iterator with custom binary function which takes index and value.
 *
 * @code {.cpp}
 * #include "tabulate_output_iterator.cuh"
 * #include <thrust/device_vector.h>
 * #include <thrust/iterator/counting_iterator.h>
 * #include <thrust/iterator/transform_iterator.h>
 *
 * struct set_bits_field {
 *   int* bitfield;
 *   __device__ inline void set_bit(size_t bit_index)
 *   {
 *     atomicOr(&bitfield[bit_index/32], (int{1} << (bit_index % 32)));
 *   }
 *   __device__ inline void clear_bit(size_t bit_index)
 *   {
 *     atomicAnd(&bitfield[bit_index / 32], ~(int{1} << (bit_index % 32)));
 *   }
 *   // Index, value
 *   __device__ void operator()(size_t i, bool x)
 *   {
 *     if (x)
 *       set_bit(i);
 *     else
 *       clear_bit(i);
 *   }
 * };
 *
 * thrust::device_vector<int> v(1, 0x00000000);
 * auto result_begin = thrust::make_tabulate_output_iterator(set_bits_field{v.data().get()});
 * auto value = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
 *   [] __device__ (int x) {   return x%2; });
 * thrust::copy(thrust::device, value, value+32, result_begin);
 * assert(v[0] == 0xaaaaaaaa);
 * @endcode
 *
 *
 * @tparam BinaryFunction Binary function to be called with the Iterator value and the rhs of
 * assignment operator.
 * @tparam Iterator iterator type that acts as index of the output.
 */
template <typename BinaryFunction, typename IndexT = ptrdiff_t>
class tabulate_output_iterator
  : public thrust::iterator_adaptor<tabulate_output_iterator<BinaryFunction, IndexT>,
                                    thrust::counting_iterator<IndexT>,
                                    thrust::use_default,
                                    thrust::use_default,
                                    thrust::use_default,
                                    tabulate_output_iterator_proxy<BinaryFunction, IndexT>> {
 public:
  // parent class.
  using super_t = thrust::iterator_adaptor<tabulate_output_iterator<BinaryFunction, IndexT>,
                                           thrust::counting_iterator<IndexT>,
                                           thrust::use_default,
                                           thrust::use_default,
                                           thrust::use_default,
                                           tabulate_output_iterator_proxy<BinaryFunction, IndexT>>;
  // friend thrust::iterator_core_access to allow it access to the private interface dereference()
  friend class thrust::iterator_core_access;
  __host__ __device__ tabulate_output_iterator(BinaryFunction fun) : fun(fun) {}

 private:
  BinaryFunction fun;

  // thrust::iterator_core_access accesses this function
  __host__ __device__ typename super_t::reference dereference() const
  {
    return tabulate_output_iterator_proxy<BinaryFunction, IndexT>(*this->base(), fun);
  }
};

template <typename BinaryFunction>
tabulate_output_iterator<BinaryFunction> __host__ __device__
make_tabulate_output_iterator(BinaryFunction fun)
{
  return tabulate_output_iterator<BinaryFunction>(fun);
}  // end make_tabulate_output_iterator

}  // namespace detail
}  // namespace cudf

// Register tabulate_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <class BinaryFunction, class IndexT>
struct thrust::detail::is_proxy_reference<
  cudf::detail::tabulate_output_iterator_proxy<BinaryFunction, IndexT>>
  : public thrust::detail::true_type {};
