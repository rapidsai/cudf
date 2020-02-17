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

#include <cmath>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/utilities/release_assert.cuh>

namespace cudf {
namespace experimental {
namespace detail {

template <typename Result, typename T>
CUDA_HOST_DEVICE_CALLABLE
Result get_array_value(T const* devarr, size_type location)
{
    T result;
#if defined(__CUDA_ARCH__)
    result = devarr[location];
#else
    CUDA_TRY( cudaMemcpy(&result, devarr + location, sizeof(T), cudaMemcpyDeviceToHost) );
#endif
    return static_cast<Result>(result);
}

namespace interpolate {

template <typename Result, typename T>
CUDA_HOST_DEVICE_CALLABLE
Result linear(T lhs, T rhs, double frac)
{
    // TODO: safe operation to avoid overflow/underflow
    // double can fully represent int8-32 value range.
    // Since the fraction part of double is 52 bits,
    // double cannot fully represent int64.
    // Underflow may occur when converting int64 to double
    // detail: https://github.com/rapidsai/cudf/issues/1417

    double dlhs = static_cast<double>(lhs);
    double drhs = static_cast<double>(rhs);
    double one_minus_frac = 1.0 - frac;
    return static_cast<Result>(one_minus_frac * dlhs + frac * drhs);
}

template <typename Result, typename T>
CUDA_HOST_DEVICE_CALLABLE
Result midpoint(T lhs, T rhs)
{
    // TODO: try std::midpoint (C++20) if available
    double dlhs = static_cast<double>(lhs);
    double drhs = static_cast<double>(rhs);
    return static_cast<Result>(dlhs / 2 + drhs / 2);
}

template <typename Result>
CUDA_HOST_DEVICE_CALLABLE
Result midpoint(int64_t lhs, int64_t rhs)
{
    // caring to avoid integer overflow and underflow between int64_t and Result( double )
    int64_t half = lhs / 2 + rhs / 2;
    int64_t rest = lhs % 2 + rhs % 2;
    return static_cast<Result>(static_cast<Result>(half) +
                               static_cast<Result>(rest) * 0.5);
}

template <>
CUDA_HOST_DEVICE_CALLABLE
int64_t midpoint(int64_t lhs, int64_t rhs)
{
    // caring to avoid integer overflow
    int64_t half = lhs / 2 + rhs / 2;
    int64_t rest = lhs % 2 + rhs % 2;
    int64_t result = half;

    // rounding toward zero
    result += ( half >= 0 && rest != 0 ) ? rest/2 : 0;
    result += ( half <  0 && rest != 0 ) ? 1 : 0;

    return result;
}

} // namespace interpolate

struct quantile_index
{
    size_type lower;
    size_type higher;
    size_type nearest;
    double fraction;

    CUDA_HOST_DEVICE_CALLABLE
    quantile_index(size_type count, double quantile)
    {
        quantile = std::min(std::max(quantile, 0.0), 1.0);

        double val = quantile * (count - 1);
        lower = std::floor(val);
        higher = static_cast<size_type>(std::ceil(val));
        nearest = static_cast<size_type>(std::nearbyint(val));
        fraction = val - lower;
    }
};

#pragma nv_exec_check_disable
/* @brief computes a quantile value.
 *
 * Computes a value for a quantile by interpolating between two values on either
 * side of the desired quantile.
 *
 * `get_value` must have signature: `T <T>(size_type)` where T can be
 * `static_cast` to `Result`.
 *
 * @param[in] get_value Gets the value at a given index in range [0, size].
 * @param[in] size      Number of values indexed by `get_value`.
 * @param[in] q         Desired quantile in range [0, 1].
 * @param[in] interp    Strategy used to interpolate between the two values
 *                      on either side of the desired quantile.
 *
 * @returns Value of the desired quantile.
 */
template<typename Result, typename ValueAccessor>
CUDA_HOST_DEVICE_CALLABLE
Result
select_quantile(ValueAccessor get_value,
                size_type size,
                double q,
                interpolation interp)
{
    if (size < 2) {
        return get_value(0);
    }

    quantile_index idx(size, q);

    switch (interp) {
    case interpolation::LINEAR:
        return interpolate::linear<Result>(get_value(idx.lower),
                                           get_value(idx.higher),
                                           idx.fraction);

    case interpolation::MIDPOINT:
        return interpolate::midpoint<Result>(get_value(idx.lower),
                                             get_value(idx.higher));

    case interpolation::LOWER:
        return static_cast<Result>(get_value(idx.lower));

    case interpolation::HIGHER:
        return static_cast<Result>(get_value(idx.higher));

    case interpolation::NEAREST:
        return static_cast<Result>(get_value(idx.nearest));

    default:
        #if defined(__CUDA_ARCH__)
            release_assert(false && "Invalid interpolation operation for quantiles");
            return Result();
        #else
            CUDF_FAIL("Invalid interpolation operation for quantiles.");
        #endif
    }
}

} // namespace detail
} // namespace experimental
} // namespace cudf
