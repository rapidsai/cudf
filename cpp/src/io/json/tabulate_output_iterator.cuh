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

// Tabulate output iterator
#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

THRUST_NAMESPACE_BEGIN


// forward declaration of tabulate_output_iterator
template <typename, typename>
class tabulate_output_iterator;
namespace detail {

// Proxy reference that calls BinaryFunction with Iterator value and the rhs of assignment operator
template <typename BinaryFunction, typename Iterator>
class tabulate_output_iterator_proxy {
 public:
  __host__ __device__ tabulate_output_iterator_proxy(const Iterator& index_iter, BinaryFunction fun)
    : index_iter(index_iter), fun(fun)
  {
  }
  template <typename T>
  __host__ __device__ tabulate_output_iterator_proxy operator=(const T& x)
  {
    fun(*index_iter, x);
    return *this;
  }

 private:
  Iterator index_iter;
  BinaryFunction fun;
};

// Register tabulate_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <class BinaryFunction, class Iterator>
struct is_proxy_reference<tabulate_output_iterator_proxy<BinaryFunction, Iterator>>
  : public thrust::detail::true_type {
};

template<typename BinaryFunction, typename System = use_default>
  struct tabulate_output_iterator_base
{
  // XXX value_type should actually be void
  //     but this interferes with zip_iterator<discard_iterator>
  // typedef any_assign         value_type;
  // typedef any_assign&        reference;
  using incrementable = std::ptrdiff_t;

  using base_iterator = typename thrust::counting_iterator<
    incrementable,
    System,
    thrust::random_access_traversal_tag
  >;
  using type = typename thrust::iterator_adaptor<
    tabulate_output_iterator<BinaryFunction, System>,
    base_iterator,
    thrust::use_default,
    typename thrust::iterator_system<base_iterator>::type,
    typename thrust::iterator_traversal<base_iterator>::type,
    thrust::detail::tabulate_output_iterator_proxy<BinaryFunction, base_iterator>
  >;
}; // end tabulate_output_iterator_base


}  // namespace detail

/**
 * @brief Tabulate output iterator with custom writer binary function which takes index and value.
 *
 * @code {.cpp}
 * #include <thrust/iterator/tabulate_output_iterator.cuh>
 * #include <thrust/device_vector.h>
 * #include <thrust/iterator/counting_iterator.h>
 * #include <thrust/iterator/transform_iterator.h>
 * #include <thrust/execution_policy.h>
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
template <typename BinaryFunction, typename System = use_default>
class tabulate_output_iterator
  : public detail::tabulate_output_iterator_base<BinaryFunction, System>::type
   {
 public:
  // parent class.
  using super_t = typename detail::tabulate_output_iterator_base<BinaryFunction, System>::type;
  using incrementable = typename detail::tabulate_output_iterator_base<BinaryFunction, System>::incrementable;
  using base_iterator = typename detail::tabulate_output_iterator_base<BinaryFunction, System>::base_iterator;
  
  // friend thrust::iterator_core_access to allow it access to the private interface dereference()
  friend class thrust::iterator_core_access;
  __host__ __device__ tabulate_output_iterator(BinaryFunction fun, incrementable const &i = incrementable())
    : super_t(base_iterator(i)), fun(fun)
  {
  }

 private:
  BinaryFunction fun;

  // thrust::iterator_core_access accesses this function
  __host__ __device__ typename super_t::reference dereference() const
  {
    return thrust::detail::tabulate_output_iterator_proxy<BinaryFunction, base_iterator>(
      this->base_reference(), fun);
  }
};

template <typename BinaryFunction>
tabulate_output_iterator<BinaryFunction> __host__ __device__
make_tabulate_output_iterator(BinaryFunction fun)
{
  return tabulate_output_iterator<BinaryFunction>(fun);
}  // end make_tabulate_output_iterator
THRUST_NAMESPACE_END