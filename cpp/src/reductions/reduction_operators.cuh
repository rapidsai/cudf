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

#pragma once

#include <utilities/cudf_utils.h>   //for CUDA_HOST_DEVICE_CALLABLE
#include <cudf/detail/iterator.cuh>
#include <iterator/transform_unary_functions.cuh>
#include <utilities/device_operators.cuh>

#include <cmath>
#include <thrust/functional.h>

namespace cudf {
namespace experimental {
namespace reduction {

// intermediate data structure to compute `var`, `std`
template<typename ResultType>
struct var_std
{
    ResultType value;                /// the value
    ResultType value_squared;        /// the value of squared

    CUDA_HOST_DEVICE_CALLABLE
    var_std(ResultType _value=0, ResultType _value_squared=0)
    : value(_value), value_squared(_value_squared)
    {};

    using this_t = var_std<ResultType>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared)
        );
    };
};

// transformer for `struct var_std` in order to compute `var`, `std`
template<typename ResultType>
struct transformer_var_std
{
    using OutputType = var_std<ResultType>;

    CUDA_HOST_DEVICE_CALLABLE
    OutputType operator() (ResultType const & value)
    {
        return OutputType(value, value*value);
    };
};

// ------------------------------------------------------------------------
// Definitions of device struct for reduction operation
// all `op::xxx` must have `Op` and `transformer`
// `Op`  is used to compute the reduction at device
// `transformer` is used to convert elements for computing the reduction at device
// By default `transformer` is static type conversion to ResultType
// In some cases, it could be square or abs or complex operations

namespace op {

// `sum`, `product`, `sum_of_squares`, `min`, `max`
// are used at simple_reduction

struct sum {
    using Op = cudf::DeviceSum;

    template<typename ResultType>
    using transformer = thrust::identity<ResultType>;
};

struct product {
    using Op = cudf::DeviceProduct;

    template<typename ResultType>
    using transformer = thrust::identity<ResultType>;
};

struct sum_of_squares {
    using Op = cudf::DeviceSum;

    template<typename ResultType>
    using transformer = cudf::transformer_squared<ResultType>;
};

struct min {
    using Op = cudf::DeviceMin;

    template<typename ResultType>
    using transformer = thrust::identity<ResultType>;
};

struct max {
    using Op = cudf::DeviceMax;

    template<typename ResultType>
    using transformer = thrust::identity<ResultType>;
};

/**
 * @brief  Compound Operator CRTP Base class
 * This template class defines the interface for compound operators
 *
 * @tparam Derived compound operators derived from CompoundOp
 */
template <typename Derived>
struct CompoundOp {

  //Call this using Derived::template compute_result<T>(...);
  template<typename ResultType, typename IntermediateType>
  CUDA_HOST_DEVICE_CALLABLE static ResultType
  compute_result(IntermediateType& input, 
                 cudf::size_type count,
                 cudf::size_type ddof) {
    //Enforced interface
    return Derived::template intermediate<ResultType>::compute_result(input, count, ddof);
  }

  private:
    CompoundOp(){};
    friend Derived;
};

// `mean`, `variance`, `standard_deviation` are used at compound_reduction
// compound_reduction requires intermediate::IntermediateType and intermediate::compute_result
// IntermediateType is the intermediate data structure type of a single reduction call,
// it is also used as OutputType of cudf::reduction::detail::reduce at compound_reduction.
// compute_result computes the final ResultType from the IntermediateType.
// intemediate::compute_result method is enforced by CRTP base class CompoundOp<>

// operator for `mean`
struct mean : public CompoundOp<mean> {
    using Op = cudf::DeviceSum;

    template<typename ResultType>
    using transformer = thrust::identity<ResultType>;

    template<typename ResultType>
    struct intermediate{
        using IntermediateType = ResultType;  // sum value

        // compute `mean` from intermediate type `IntermediateType`
        CUDA_HOST_DEVICE_CALLABLE
        static ResultType compute_result(const IntermediateType& input, cudf::size_type count, cudf::size_type ddof)
        {
            return (input / count);
        };

    };
};

// operator for `variance`
struct variance : public CompoundOp<variance> {
    using Op = cudf::DeviceSum;

    template<typename ResultType>
    using transformer = cudf::experimental::reduction::transformer_var_std<ResultType>;

    template<typename ResultType>
    struct intermediate{
        using IntermediateType = var_std<ResultType>; //with sum of value, and sum of squared value

        // compute `variance` from intermediate type `IntermediateType`
        CUDA_HOST_DEVICE_CALLABLE
        static ResultType compute_result(const IntermediateType& input, cudf::size_type count, cudf::size_type ddof)
        {
            ResultType mean = input.value / count;
            ResultType asum = input.value_squared;
            cudf::size_type div = count -ddof;
            ResultType var = asum / div - ((mean * mean) * count) /div;

            return var;
        };
    };
};

// operator for `standard deviation`
struct standard_deviation : public CompoundOp<standard_deviation> {
    using Op = cudf::DeviceSum;

    template<typename ResultType>
    using transformer = cudf::experimental::reduction::transformer_var_std<ResultType>;

    template<typename ResultType>
    struct intermediate{
        using IntermediateType = var_std<ResultType>; //with sum of value, and sum of squared value

        // compute `standard deviation` from intermediate type `IntermediateType`
        CUDA_HOST_DEVICE_CALLABLE
        static ResultType compute_result(const IntermediateType& input, cudf::size_type count, cudf::size_type ddof)
        {
            using intermediateOp = typename variance::template intermediate<ResultType>;
            ResultType var = intermediateOp::compute_result(input, count, ddof);

            return static_cast<ResultType>(std::sqrt(var));
        };
    };
};

} // namespace op

template <typename ElementType, typename ResultType, typename Op>
auto make_reduction_iterator(column_device_view const& column) {
    return thrust::make_transform_iterator(
        experimental::detail::make_null_replacement_iterator( column, Op::Op::template identity<ElementType>()),
        typename Op::template transformer<ResultType>{});
}

} // namespace reduction
} // namespace experimental
} // namespace cudf
