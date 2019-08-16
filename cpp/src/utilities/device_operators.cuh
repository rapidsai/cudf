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

#ifndef DEVICE_OPERATORS_CUH
#define DEVICE_OPERATORS_CUH

/** ---------------------------------------------------------------------------*
 * @brief definition of the device operators
 * @file device_operators.cuh
 *
 * ---------------------------------------------------------------------------**/

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>       // need for CUDA_HOST_DEVICE_CALLABLE
#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

// ------------------------------------------------------------------------
// Binary operators
/* @brief binary `sum` operator */
struct DeviceSum {
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

/* @brief `count` operator - used in rolling windows */
struct DeviceCount {
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &, const T &rhs) {
        return rhs + T{1};
    }

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

/* @brief binary `min` operator */
struct DeviceMin{
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

/* @brief binary `max` operator */
struct DeviceMax{
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};

/* @brief binary `product` operator */
struct DeviceProduct {
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{1}; }
};


/* @brief binary `and` operator */
struct DeviceAnd{
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return (lhs & rhs );
    }
};

/* @brief binary `or` operator */
struct DeviceOr{
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return (lhs | rhs );
    }
};

/* @brief binary `xor` operator */
struct DeviceXor{
    template<typename T>
    CUDA_HOST_DEVICE_CALLABLE
    T operator() (const T &lhs, const T &rhs) {
        return (lhs ^ rhs );
    }
};

} // namespace cudf


#endif
