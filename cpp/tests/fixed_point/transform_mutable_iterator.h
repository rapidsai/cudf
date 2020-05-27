/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Vesion 2.0 (the "License");
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

/*! \file thrust/iterator/transform_output_iterator.h
 *  \brief An output iterator which adapts another output iterator by applying a
 *         function to the result of its dereference before writing it.
 */

// Copy-pasted from thrust/iterator/transform_output_iterator.h

#pragma once

#include <thrust/detail/config.h>
#include "detail/transform_mutable_iterator.inl"

namespace thrust
{

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p transform_output_iterator is a special kind of output iterator which
 * transforms a value written upon dereference. This iterator is useful
 * for transforming an output from algorithms without explicitly storing the
 * intermediate result in the memory and applying subsequent transformation, 
 * thereby avoiding wasting memory capacity and bandwidth.
 * Using \p transform_iterator facilitates kernel fusion by deferring execution
 * of transformation until the value is written while saving both memory
 * capacity and bandwidth.
 *
 * The following code snippet demonstrated how to create a
 * \p transform_output_iterator which applies \c sqrtf to the assigning value.
 *
 * \code
 * #include <thrust/iterator/transform_output_iterator.h>
 * #include <thrust/device_vector.h>
 *
 * // note: functor inherits form unary function
 *  // note: functor inherits from unary_function
 *  struct square_root : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) const
 *    {
 *      return sqrtf(x);
 *    }
 *  };
 *  
 *  int main()
 *  {
 *    thrust::device_vector<float> v(4);
 *
 *    typedef thrust::device_vector<float>::iterator FloatIterator;
 *    thrust::transform_output_iterator<square_root, FloatIterator> iter(v.begin(), square_root());
 *
 *    iter[0] =  1.0f;    // stores sqrtf( 1.0f) 
 *    iter[1] =  4.0f;    // stores sqrtf( 4.0f)
 *    iter[2] =  9.0f;    // stores sqrtf( 9.0f)
 *    iter[3] = 16.0f;    // stores sqrtf(16.0f)
 *    // iter[4] is an out-of-bounds error
 *                                                                                           
 *    v[0]; // returns 1.0f;
 *    v[1]; // returns 2.0f;
 *    v[2]; // returns 3.0f;
 *    v[3]; // returns 4.0f;
 *                                                                                           
 *  }
 *  \endcode
 *
 *  \see make_transform_output_iterator
 */

template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
  class transform_mutable_iterator
    : public detail::transform_mutable_iterator_base<ForwardFunction, ReverseFunction, Iterator>::type
{

  /*! \cond
   */

  public:

    typedef typename
    detail::transform_mutable_iterator_base<ForwardFunction, ReverseFunction, Iterator>::type
    super_t;

    friend class thrust::iterator_core_access;
  /*! \endcond
   */

  /*! This constructor takes as argument an \c OutputIterator and an \c
   * UnaryFunction and copies them to a new \p transform_output_iterator
   *
   * \param out An \c OutputIterator pointing to the output range whereto the result of 
   *            \p transform_output_iterator's \c UnaryFunction will be written.
   * \param fun An \c UnaryFunction used to transform the objects assigned to
   *            this \p transform_output_iterator.
   */
    __host__ __device__
    transform_mutable_iterator(Iterator& out, ForwardFunction forward, ReverseFunction reverse) :
      super_t(out), forward(forward), reverse(reverse)
    {
    }

    /*! \cond
     */
  private:

    __host__ __device__
    typename super_t::reference dereference() const
    {
      return detail::transform_mutable_iterator_proxy<
        ForwardFunction, ReverseFunction, Iterator
      >(this->base_reference(), forward, reverse);
    }

    ForwardFunction forward;
    ReverseFunction reverse;

    /*! \endcond
     */
}; // end transform_output_iterator

/*! \p make_transform_output_iterator creates a \p transform_output_iterator from
 *  an \c OutputIterator and \c UnaryFunction.
 *
 *  \param out The \c OutputIterator pointing to the output range of the newly
 *            created \p transform_output_iterator
 *  \param fun The \c UnaryFunction transform the object before assigning it to
 *            \c out by the newly created \p transform_output_iterator
 *  \see transform_output_iterator
 */
template <typename ForwardFunction, typename ReverseFunction, typename Iterator>
transform_mutable_iterator<ForwardFunction, ReverseFunction, Iterator>
__host__ __device__
make_transform_mutable_iterator(Iterator out, ForwardFunction forward, ReverseFunction reverse)
{
    return transform_mutable_iterator<ForwardFunction, ReverseFunction, Iterator>(out, forward, reverse);
} // end make_transform_output_iterator

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

} // end thrust

