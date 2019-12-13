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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace experimental {
namespace detail {

template <typename TResult, typename T>
CUDA_HOST_DEVICE_CALLABLE
TResult get_array_value(T const* devarr, size_type location)
{
    T result;
#if defined(__CUDA_ARCH__)
    result = devarr[location];
#else
    CUDA_TRY( cudaMemcpy(&result, devarr + location, sizeof(T), cudaMemcpyDeviceToHost) );
#endif
    return static_cast<TResult>(result);
}

namespace interpolate {

template <typename TResult, typename T>
CUDA_HOST_DEVICE_CALLABLE
TResult linear(T lhs, T rhs, double frac)
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
//    result = static_cast<TResult>(static_cast<TResult>(lhs) + frac*static_cast<TResult>(rhs-lhs));
    return static_cast<TResult>(one_minus_frac * dlhs + frac * drhs);
}

template <typename TResult, typename T>
CUDA_HOST_DEVICE_CALLABLE
TResult midpoint(T lhs, T rhs)
{
    // TODO: try std::midpoint (C++20) if available
    double dlhs = static_cast<double>(lhs);
    double drhs = static_cast<double>(rhs);
    return static_cast<TResult>(dlhs / 2 + drhs / 2);
}

template <typename TResult>
CUDA_HOST_DEVICE_CALLABLE
TResult midpoint(int64_t lhs, int64_t rhs)
{
    // caring to avoid integer overflow and underflow between int64_t and TResult( double )
    int64_t half = lhs / 2 + rhs / 2;
    int64_t rest = lhs % 2 + rhs % 2;
    return static_cast<TResult>(static_cast<TResult>(half) +
                                static_cast<TResult>(rest) * 0.5);
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
    result += ( half >= 0 && rest != 0 )? rest/2 : 0;
    result += ( half < 0  && rest != 0 )? 1 : 0;

    return result;
}

} // namespace interpolate
} // namespace detail
} // namespace experimental
} // namespace cudf
