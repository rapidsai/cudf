/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

//Quantile (percentile) functionality

#include <cudf/cudf.h>
#include <cudf/quantiles.hpp>
#include <utilities/cudf_utils.h>
#include <utilities/release_assert.cuh>

namespace cudf {
namespace interpolate {

template <typename T_out, typename T_in>
CUDA_HOST_DEVICE_CALLABLE
void linear(T_out& result, T_in lhs, T_in rhs, double frac)
{
    // TODO: safe operation to avoid overflow/underflow
    // double can fully represent int8-32 value range.
    // Since the fractoin part of double is 52 bits,
    // double cannot fully represent int64.
    // The underflow will be occurs at converting int64 to double
    // detail: https://github.com/rapidsai/cudf/issues/1417

    double dlhs = static_cast<double>(lhs);
    double drhs = static_cast<double>(rhs);
    double one_minus_frac = 1.0 - frac;

//    result = static_cast<T_out>(static_cast<T_out>(lhs) + frac*static_cast<T_out>(rhs-lhs));
    result = static_cast<T_out>(one_minus_frac * dlhs + frac * drhs);
}


template <typename T_out, typename T_in>
CUDA_HOST_DEVICE_CALLABLE
void midpoint(T_out& result, T_in lhs, T_in rhs)
{
    // TODO: try std::midpoint (C++20) if available
    double dlhs = static_cast<double>(lhs);
    double drhs = static_cast<double>(rhs);

    result = static_cast<T_out>( dlhs /2 + drhs /2 );
}

// -------------------------------------------------------------------------
// @overloads

template <typename T_out>
CUDA_HOST_DEVICE_CALLABLE
void midpoint(T_out& result, int64_t lhs, int64_t rhs)
{
    // caring to avoid integer overflow and underflow between int64_t and T_out( double )
    int64_t half = lhs/2 + rhs/2;
    int64_t rest = (lhs%2 + rhs%2);

    result = static_cast<T_out>(static_cast<T_out>(half) + static_cast<T_out>(rest)*0.5);
}


template <>
CUDA_HOST_DEVICE_CALLABLE
void midpoint(int64_t& result, int64_t lhs, int64_t rhs)
{
    // caring to avoid integer overflow
    int64_t half = lhs/2 + rhs/2;
    int64_t rest = (lhs%2 + rhs%2);
    result = half;

    // rounding toward zero
    result += ( half >= 0 && rest != 0 )? rest/2 : 0;
    result += ( half < 0  && rest != 0 )? 1 : 0;
}

} // end of namespace interpolate

namespace detail {

/**
 * @brief Helper struct that calculates the values needed to get quantile values
 * by interpolation.
 * 
 * For a quantile that lies between indices i and j, this struct calculates 
 * i (lower_bound),
 * j (upper_bound),
 * index nearest to quantile between i and j (nearest),
 * and the fractional distance that the quantile lies ahead i (fraction)
 * 
 */
struct quantile_index {
    gdf_size_type lower_bound;
    gdf_size_type upper_bound;
    gdf_size_type nearest;
    double fraction;

    CUDA_HOST_DEVICE_CALLABLE
    quantile_index(gdf_size_type length, double quant)
    {
        // clamp quant value.
        // Todo: use std::clamp if c++17 is supported.
        quant = std::min(std::max(quant, 0.0), 1.0);

        // since gdf_size_type is int32_t, there is no underflow/overflow
        double val = quant*(length -1);
        lower_bound = std::floor(val);
        upper_bound = static_cast<size_t>(std::ceil(val));
        nearest = static_cast<size_t>(std::nearbyint(val));
        fraction = val - lower_bound;
    }
};

template <typename T>
CUDA_HOST_DEVICE_CALLABLE
T get_array_value(T const* devarr, gdf_size_type location)
{
    T result;
#if defined(__CUDA_ARCH__)
    result = devarr[location];
#else
    CUDA_TRY( cudaMemcpy(&result, devarr + location, sizeof(T), cudaMemcpyDeviceToHost) );
#endif
    return result;
}

/**
 * @brief Gets the value at the given quantile
 * 
 * @param devarr array of values to get quantile from
 * @param size size of @p devarr
 * @param quantile Number in [0,1] indicating quantile to get
 * @param interpolation method to calculate value when quantile lies at inexact index
 * @return value at quantile 
 */
template <typename T,
          typename RetT = double>
CUDA_HOST_DEVICE_CALLABLE
RetT select_quantile(T const* devarr, gdf_size_type size, double quantile,
                       interpolation interpolation)
{
    T temp[2];
    RetT result;
    
    if( size < 2 )
    {
        temp[0] = get_array_value(devarr, 0);
        result = static_cast<RetT>( temp[0] );
        return result;
    }

    quantile_index qi(size, quantile);

    switch( interpolation )
    {
    case LINEAR:
        temp[0] = get_array_value(devarr, qi.lower_bound);
        temp[1] = get_array_value(devarr, qi.upper_bound);
        cudf::interpolate::linear(result, temp[0], temp[1], qi.fraction);
        break;
    case MIDPOINT:
        temp[0] = get_array_value(devarr, qi.lower_bound);
        temp[1] = get_array_value(devarr, qi.upper_bound);
        cudf::interpolate::midpoint(result, temp[0], temp[1]);
        break;
    case LOWER:
        temp[0] = get_array_value(devarr, qi.lower_bound);
        result = static_cast<RetT>( temp[0] );
        break;
    case HIGHER:
        temp[0] = get_array_value(devarr, qi.upper_bound);
        result = static_cast<RetT>( temp[0] );
        break;
    case NEAREST:
        temp[0] = get_array_value(devarr, qi.nearest);
        result = static_cast<RetT>( temp[0] );
        break;
    default:
        #if defined(__CUDA_ARCH__)
            release_assert(false && "Invalid interpolation operation for quantiles");
        #else
            CUDF_FAIL("Invalid interpolation operation for quantiles");
        #endif
    }

    return result;
}

} // namespace detail

} // end of namespace cudf

