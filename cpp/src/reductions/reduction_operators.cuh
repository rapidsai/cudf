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

#ifndef CUDF_REDUCTION_OPERATORS_CUH
#define CUDF_REDUCTION_OPERATORS_CUH

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/error_utils.hpp>
#include <iterator/iterator.cuh>

#include <utilities/device_operators.cuh>

#include <cmath>

namespace cudf {
namespace reductions {

template<typename T>
struct meanvar_no_count
{
    T value;                /// the value
    T value_squared;        /// the value of squared

    CUDA_HOST_DEVICE_CALLABLE
    meanvar_no_count(T _value=0, T _value_squared=0)
    : value(_value), value_squared(_value_squared)
    {};

    using this_t = cudf::reductions::meanvar_no_count<T>;

    CUDA_HOST_DEVICE_CALLABLE
    this_t operator+(this_t const &rhs) const
    {
        return this_t(
            (this->value + rhs.value),
            (this->value_squared + rhs.value_squared)
        );
    };

    CUDA_HOST_DEVICE_CALLABLE
    bool operator==(this_t const &rhs) const
    {
        return (
            (this->value == rhs.value) &&
            (this->value_squared == rhs.value_squared)
        );
    };
};

template<typename T_element>
struct transformer_meanvar_no_count
{
    using T_output = cudf::reductions::meanvar_no_count<T_element>;

    CUDA_HOST_DEVICE_CALLABLE
    T_output operator() (T_element const & value)
    {
        return T_output(value, value*value);
    };
};

// ------------------------------------------------------------------------
// difinitions of device struct for binary operation

struct ReductionSum {
    using Op = cudf::DeviceSum;
};

struct ReductionProduct {
    using Op = cudf::DeviceProduct;
};

struct ReductionSumOfSquares {
    using Op = cudf::DeviceSum;
};

struct ReductionMin{
    using Op = cudf::DeviceMin;
};

struct ReductionMax{
    using Op = cudf::DeviceMax;
};

struct ReductionMean{
    using Op = cudf::DeviceSum;

    template<typename T>
    struct Intermediate{
        using IType = T;

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(const gdf_column* column, T_out identity){
            return cudf::make_iterator<has_nulls, T_in, T_out>(*column, identity);
        }

        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            return (input / count);
        };

    };
};

struct ReductionVar{
    using Op = cudf::DeviceSum;

    template<typename T>
    struct Intermediate{
        using IType = meanvar_no_count<T>;

        static IType identity() {
            return T{0};
        }

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(const gdf_column* column, T_out identity){
            auto transformer = cudf::reductions::transformer_meanvar_no_count<T>{};
            auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(*column, identity);
            return thrust::make_transform_iterator(it_raw, transformer);
        }

        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            T mean = input.value / count;
            T asum = input.value_squared;
            gdf_size_type div = count -ddof;

            T var = asum / div - ((mean * mean) * count) /div;
            return var;
        };
    };
};

struct ReductionStd{
    using Op = cudf::DeviceSum;

    template<typename T>
    struct Intermediate{
        using IType = meanvar_no_count<T>;

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(const gdf_column* column, T_out identity){
            auto transformer = cudf::reductions::transformer_meanvar_no_count<T>{};
            auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(*column, identity);
            return thrust::make_transform_iterator(it_raw, transformer);
        }

        static
        T ComputeResult(IType& input, gdf_size_type count, gdf_size_type ddof = 1)
        {
            using intermediateOp = typename ReductionVar::template Intermediate<T>;
            T var = intermediateOp::ComputeResult(input, count, ddof);

            return static_cast<T>(std::sqrt(var));
        };
    };
};


} // namespace reductions
} // namespace cudf

#endif