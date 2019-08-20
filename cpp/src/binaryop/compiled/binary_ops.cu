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

#include <algorithm>

#include "launcher.cuh"
#include "binary_ops.hpp"
#include <utilities/cudf_utils.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <cudf/binaryop.hpp>

namespace cudf {
namespace binops {
namespace compiled {

// Arithmeitc

struct DeviceAdd {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs + rhs;
    }
};

struct DeviceSub {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs - rhs;
    }
};

struct DeviceMul {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs * rhs;
    }
};

struct DeviceFloorDiv {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return std::floor((double)lhs / (double)rhs);
    }
};

struct DeviceDiv {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs / rhs;
    }
};

// logical

struct DeviceGt {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs > rhs);
    }
};

struct DeviceGe {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs >= rhs);
    }
};

struct DeviceLt {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs < rhs);
    }
};

struct DeviceLe {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs <= rhs);
    }
};

struct DeviceEq {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs == rhs);
    }
};


struct DeviceNe {
    template<typename T>
    __device__
    cudf::bool8 apply(T lhs, T rhs) {
        return static_cast<cudf::bool8>(lhs != rhs);
    }
};

// bitwise

struct DeviceBitwiseAnd {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs & rhs;
    }
};


struct DeviceBitwiseOr {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs | rhs;
    }
};


struct DeviceBitwiseXor {
    template<typename T>
    __device__
    T apply(T lhs, T rhs) {
        return lhs ^ rhs;
    }
};

template<typename F>
struct ArithOp {
private:
    template <typename T>
    static constexpr bool is_supported() {
        if (std::is_same<F, DeviceDiv>::value)
            return std::is_floating_point<T>::value;
        else
            return std::is_arithmetic<T>::value;
    }

public:
    // static
    template <typename T>
    typename std::enable_if_t<is_supported<T>(), gdf_error>
    operator()(gdf_column *lhs, gdf_column *rhs, gdf_column *out) {
        GDF_REQUIRE(out->dtype == lhs->dtype, GDF_UNSUPPORTED_DTYPE);
        return BinaryOp<T, T, F>::launch(lhs, rhs, out);
    }

    template <typename T>
    typename std::enable_if_t<!is_supported<T>(), gdf_error>
    operator()(gdf_column *lhs, gdf_column *rhs, gdf_column *out) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


template<typename F>
struct LogicalOp {
    // static
    template <typename T>
    gdf_error operator()(gdf_column *lhs, gdf_column *rhs, gdf_column *out) {
        GDF_REQUIRE(out->dtype == GDF_BOOL8, GDF_UNSUPPORTED_DTYPE);
        return BinaryOp<T, cudf::bool8, F>::launch(lhs, rhs, out);
    }
};


template<typename F>
struct BitwiseOp {
    // static
    template <typename T>
    typename std::enable_if_t<std::is_integral<T>::value, gdf_error>
    operator()(gdf_column *lhs, gdf_column *rhs, gdf_column *out) {
        GDF_REQUIRE(out->dtype == lhs->dtype, GDF_UNSUPPORTED_DTYPE);
        return BinaryOp<T, T, F>::launch(lhs, rhs, out);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_integral<T>::value, gdf_error>
    operator()(gdf_column *lhs, gdf_column *rhs, gdf_column *out) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


gdf_error binary_operation(gdf_column* out,
                           gdf_column* lhs,
                           gdf_column* rhs,
                           gdf_binary_operator ope)
{
    // Compiled ops are not made for heterogeneous input types
    GDF_REQUIRE(rhs->dtype == lhs->dtype, GDF_UNSUPPORTED_DTYPE);
    switch (ope)
    {
    case GDF_ADD:
        return cudf::type_dispatcher(lhs->dtype,
                                    ArithOp<DeviceAdd>{},
                                    lhs, rhs, out);
    case GDF_SUB:
        return cudf::type_dispatcher(lhs->dtype,
                                    ArithOp<DeviceSub>{},
                                    lhs, rhs, out);
    case GDF_MUL:
        return cudf::type_dispatcher(lhs->dtype,
                                    ArithOp<DeviceMul>{},
                                    lhs, rhs, out);
    case GDF_DIV:
        return cudf::type_dispatcher(lhs->dtype,
                                    ArithOp<DeviceDiv>{},
                                    lhs, rhs, out);
    case GDF_FLOOR_DIV:
        return cudf::type_dispatcher(lhs->dtype,
                                    ArithOp<DeviceFloorDiv>{},
                                    lhs, rhs, out);
    case GDF_EQUAL:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceEq>{},
                                    lhs, rhs, out);
    case GDF_NOT_EQUAL:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceNe>{},
                                    lhs, rhs, out);
    case GDF_LESS:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceLt>{},
                                    lhs, rhs, out);
    case GDF_GREATER:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceGt>{},
                                    lhs, rhs, out);
    case GDF_LESS_EQUAL:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceLe>{},
                                    lhs, rhs, out);
    case GDF_GREATER_EQUAL:
        return cudf::type_dispatcher(lhs->dtype,
                                    LogicalOp<DeviceGe>{},
                                    lhs, rhs, out);
    case GDF_BITWISE_AND:
        return cudf::type_dispatcher(lhs->dtype,
                                    BitwiseOp<DeviceBitwiseAnd>{},
                                    lhs, rhs, out);
    case GDF_BITWISE_OR:
        return cudf::type_dispatcher(lhs->dtype,
                                    BitwiseOp<DeviceBitwiseOr>{},
                                    lhs, rhs, out);
    case GDF_BITWISE_XOR:
        return cudf::type_dispatcher(lhs->dtype,
                                    BitwiseOp<DeviceBitwiseXor>{},
                                    lhs, rhs, out);
    
    default:
        return GDF_INVALID_API_CALL;
    }
}

} // namespace compiled
} // namespace binops
} // namespace cudf
