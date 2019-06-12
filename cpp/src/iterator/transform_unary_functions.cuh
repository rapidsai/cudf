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
 * @file trasnsform_unary_functions.cuh
 *
 * These are designed for using as AdaptableUnaryFunction
 * for thrust::transform_iterator.
 * For the detail of example cases,
 * @see iteraror.cuh iterator_test.cu
 * -------------------------------------------------------------------------**/

#ifndef CUDF_TRANSFORM_UNARY_FUNCTIONS_CUH
#define CUDF_TRANSFORM_UNARY_FUNCTIONS_CUH

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE

#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

namespace cudf
{
/** -------------------------------------------------------------------------*
 * @brief intermediate struct to calculate mean and variance
 * This is an example case to output a struct from column input.
 *
 * this will be used to calculate and hold `sum of values`, 'sum of squares',
 * 'sum of valid count'.
 * Those will be used to compute `mean` (= sum / count)
 * and `variance` (= sum of squares / count - mean^2).
 *
  @tparam  T  element data type of value and value_squared.
 * -------------------------------------------------------------------------**/
template<typename T>
struct meanvar
{
    T value;                /// the value
    T value_squared;        /// the value of squared
    gdf_index_type count;   /// the count

    CUDA_HOST_DEVICE_CALLABLE
    meanvar(T _value=0, T _value_squared=0, gdf_index_type _count=0)
    : value(_value), value_squared(_value_squared), count(_count)
    {};

    using this_t = cudf::meanvar<T>;

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
 * @brief Transforms a scalar by casting it to another scalar type
 *
 * It casts `T_element` into `T_output`.
 *
 * This struct transforms the output value as `static_cast<T_output>(value)`.
 *
 * @tparam  T_element scalar data type of input
 * @tparam  T_output  scalar data type of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output>
struct scalar_cast_transformer
{
    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (T_element const & value)
    {
        return static_cast<T_output>(value);
    };
};

/** -------------------------------------------------------------------------*
 * @brief Transforms a scalar by first casting to another type, and then squaring the result.
 *
 * This struct transforms the output value as
 * `(static_cast<T_output>(_value))^2`.
 *
 * This will be used to compute "sum of squares".
 *
 * @tparam  T_element scalar data type of input
 * @tparam  T_output  scalar data type of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output=T_element>
struct transformer_squared
{
    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (T_element const & value)
    {
        T_output v = static_cast<T_output>(value);
        return (v*v);
    };
};

/** -------------------------------------------------------------------------*
 * @brief Uses a scalar value to construct a `meanvar` object.
 * This transforms `thrust::pair<T_element, bool>` into
 * `T_output = meanvar<T_output_element>` form.
 *
 * This struct transforms the value and the squared value and the count at once.
 *
 * @tparam  T_element         scalar data type of input
 * @tparam  T_output_element  scalar data type of the element of output
 * -------------------------------------------------------------------------**/
template<typename T_element, typename T_output_element=T_element>
struct transformer_meanvar
{
    using T_output = meanvar<T_output_element>;

    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (thrust::pair<T_element, bool> const& pair)
    {
        T_output_element v = static_cast<T_output_element>(pair.first);
        return T_output(v, v*v, (pair.second)? 1 : 0 );
    };
};



} // namespace cudf

#endif