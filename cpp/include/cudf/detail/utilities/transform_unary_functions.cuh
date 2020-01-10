/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/** --------------------------------------------------------------------------*
 * @brief unary functions for thrust::transform_iterator
 * @file transform_unary_functions.cuh
 *
 * These are designed for using as AdaptableUnaryFunction
 * for thrust::transform_iterator.
 * For the detail of example cases,
 * @see iterator.cuh iterator_test.cu
 * -------------------------------------------------------------------------**/

#pragma once

#include <cudf/cudf.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

namespace cudf {
/** -------------------------------------------------------------------------*
 * @brief intermediate struct to calculate mean and variance
 * This is an example case to output a struct from column input.
 *
 * this will be used to calculate and hold `sum of values`, 'sum of squares',
 * 'sum of valid count'.
 * Those will be used to compute `mean` (= sum / count)
 * and `variance` (= sum of squares / count - mean^2).
 *
 * @tparam ElementType  element data type of value and value_squared.
 * -------------------------------------------------------------------------**/
template<typename ElementType>
struct meanvar
{
    ElementType value;                /// the value
    ElementType value_squared;        /// the value of squared
    cudf::size_type count;   /// the count

    CUDA_HOST_DEVICE_CALLABLE
    meanvar(ElementType _value=0, ElementType _value_squared=0, cudf::size_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};

    using this_t = cudf::meanvar<ElementType>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared),
            (this->count + rhs.count)
        );
    };

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared) &&
            (this->count == rhs.count)
        );
    };
};

// --------------------------------------------------------------------------
// transformers

/** -------------------------------------------------------------------------*
 * @brief Transforms a scalar by first casting to another type, and then squaring the result.
 *
 * This struct transforms the output value as
 * `value * value`.
 *
 * This will be used to compute "sum of squares".
 *
 * @tparam  ResultType  scalar data type of output
 * -------------------------------------------------------------------------**/
template<typename ElementType>
struct transformer_squared
{
    CUDA_HOST_DEVICE_CALLABLE
    ElementType operator() (ElementType const & value)
    {
        return (value*value);
    };
};

/** -------------------------------------------------------------------------*
 * @brief Uses a scalar value to construct a `meanvar` object.
 * This transforms `thrust::pair<ElementType, bool>` into
 * `ResultType = meanvar<ElementType>` form.
 *
 * This struct transforms the value and the squared value and the count at once.
 *
 * @tparam  ElementType         scalar data type of input
 * -------------------------------------------------------------------------**/
template<typename ElementType>
struct transformer_meanvar
{
    using ResultType = meanvar<ElementType>;

    CUDA_HOST_DEVICE_CALLABLE
    ResultType operator() (thrust::pair<ElementType, bool> const& pair)
    {
        ElementType v = pair.first;
        return ResultType(v, v*v, (pair.second)? 1 : 0 );
    };
};

} // namespace cudf
