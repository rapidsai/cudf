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

namespace op {

struct sum {
    using Op = cudf::DeviceSum;
};

struct product {
    using Op = cudf::DeviceProduct;
};

struct sum_of_squares {
    using Op = cudf::DeviceSum;
};

struct min {
    using Op = cudf::DeviceMin;
};

struct max {
    using Op = cudf::DeviceMax;
};

// operator for `mean`
struct mean {
    using Op = cudf::DeviceSum;

    template<typename T>
    struct intermediate{
        using IType = T;

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(gdf_column const& column, T_out identity)
        {
            return cudf::make_iterator<has_nulls, T_in, T_out>(column, identity);
        }

        // compute `mean` from intermediate type `IType`
        static T compute_result(const IType& input, gdf_size_type count, gdf_size_type ddof)
        {
            return (input / count);
        };

    };
};

// operator for `variance`
struct variance {
    using Op = cudf::DeviceSum;

    template<typename T>
    struct intermediate{
        using IType = meanvar_no_count<T>;

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(gdf_column const& column, T_out identity)
        {
            auto transformer = cudf::reductions::transformer_meanvar_no_count<T>{};
            auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(column, identity);
            return thrust::make_transform_iterator(it_raw, transformer);
        }

        // compute `variance` from intermediate type `IType`
        static T compute_result(const IType& input, gdf_size_type count, gdf_size_type ddof)
        {
            T mean = input.value / count;
            T asum = input.value_squared;
            gdf_size_type div = count -ddof;
            T var = asum / div - ((mean * mean) * count) /div;

            return var;
        };
    };
};

// operator for `standard deviation`
struct standard_deviation {
    using Op = cudf::DeviceSum;

    template<typename T>
    struct intermediate{
        using IType = meanvar_no_count<T>;

        template<bool has_nulls, typename T_in, typename T_out>
        static auto make_iterator(gdf_column const& column, T_out identity)
        {
            auto transformer = cudf::reductions::transformer_meanvar_no_count<T>{};
            auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(column, identity);
            return thrust::make_transform_iterator(it_raw, transformer);
        }

        // compute `standard deviation` from intermediate type `IType`
        static T compute_result(const IType& input, gdf_size_type count, gdf_size_type ddof)
        {
            using intermediateOp = typename cudf::reductions::op::variance::template intermediate<T>;
            T var = intermediateOp::compute_result(input, count, ddof);

            return static_cast<T>(std::sqrt(var));
        };
    };
};
} // namespace op


} // namespace reductions
} // namespace cudf

#endif