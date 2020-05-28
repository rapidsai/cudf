/*
 *  Copyright 2008-2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// Copy-pasted from thrust/iterator/detail/transform_output_iterator.inl

//#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
  class transform_mutable_iterator;

namespace detail 
{

// Proxy reference that uses Unary Functiont o transform the rhs of assigment
// operator before writing the result to OutputIterator
template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
  class transform_mutable_iterator_proxy
{
  public:
    using IteratorValue = typename std::iterator_traits<Iterator>::value_type;
    using Value = typename std::result_of<ForwardFunction(IteratorValue)>::type;

    __host__ __device__
    transform_mutable_iterator_proxy(Iterator const& it, ForwardFunction forward, ReverseFunction reverse) :
      it(it), forward(forward), reverse(reverse)
    {
    }

    __thrust_exec_check_disable__
    __host__ __device__
    operator Value const() const {
      return forward(*it);
    }

    __thrust_exec_check_disable__
    __host__ __device__
    transform_mutable_iterator_proxy operator=(const Value& x)
    {
      *it = reverse(x);
      return *this;
    }

    __thrust_exec_check_disable__
    __host__ __device__
    transform_mutable_iterator_proxy operator=(const transform_mutable_iterator_proxy& x)
    {
      *it = reverse(Value{x});
      return *this;
    }

  private:
    Iterator it;
    ForwardFunction forward;
    ReverseFunction reverse;
};

// Compute the iterator_adaptor instantiation to be used for transform_output_iterator
template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
struct transform_mutable_iterator_base
{
    typedef thrust::iterator_adaptor
    <
        transform_mutable_iterator<ForwardFunction, ReverseFunction, Iterator>
      , Iterator
      , typename transform_mutable_iterator_proxy<ForwardFunction, ReverseFunction, Iterator>::Value
      , thrust::use_default
      , thrust::use_default
      , transform_mutable_iterator_proxy<ForwardFunction, ReverseFunction, Iterator>
    > type;
};

// Register trasnform_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
// NOTE the original thrust code had a typo here where these params were reversed
template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
struct is_proxy_reference<
    transform_mutable_iterator_proxy<ForwardFunction, ReverseFunction, Iterator> >
    : public thrust::detail::true_type {};

} // end detail
} // end thrust

