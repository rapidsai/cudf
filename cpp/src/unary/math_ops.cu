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

#include "unary_ops.cuh"

#include <utilities/type_dispatcher.hpp>

#include <cmath>
#include <algorithm>
#include <type_traits>

// trig functions

struct DeviceSin {
    template<typename T>
    __device__
    T apply(T data) {
        return std::sin(data);
    }
};

struct DeviceCos {
    template<typename T>
    __device__
    T apply(T data) {
        return std::cos(data);
    }
};

struct DeviceTan {
    template<typename T>
    __device__
    T apply(T data) {
        return std::tan(data);
    }
};

struct DeviceArcSin {
    template<typename T>
    __device__
    T apply(T data) {
        return std::asin(data);
    }
};

struct DeviceArcCos {
    template<typename T>
    __device__
    T apply(T data) {
        return std::acos(data);
    }
};

struct DeviceArcTan {
    template<typename T>
    __device__
    T apply(T data) {
        return std::atan(data);
    }
};

// exponential functions

struct DeviceExp {
    template<typename T>
    __device__
    T apply(T data) {
        return std::exp(data);
    }
};

struct DeviceLog {
    template<typename T>
    __device__
    T apply(T data) {
        return std::log(data);
    }
};

struct DeviceSqrt {
    template<typename T>
    __device__
    T apply(T data) {
        return std::sqrt(data);
    }
};

// rounding functions

struct DeviceCeil {
    template<typename T>
    __device__
    T apply(T data) {
        return std::ceil(data);
    }
};

struct DeviceFloor {
    template<typename T>
    __device__
    T apply(T data) {
        return std::floor(data);
    }
};

struct DeviceAbs {
    template<typename T>
    __device__
    T apply(T data) {
        return std::abs(data);
    }
};

// bitwise op

struct DeviceInvert {
    // TODO: maybe sfinae overload this for cudf::bool8
    template<typename T>
    __device__
    T apply(T data) {
        return ~data;
    }
};

// logical op

struct DeviceNot {
    template<typename T>
    __device__
    cudf::bool8 apply(T data) {
        return static_cast<cudf::bool8>( !data );
    }
};


template<typename T, typename F>
static gdf_error launch(gdf_column *input, gdf_column *output) {
    return cudf::unary::Launcher<T, T, F>::launch(input, output);
}


template <typename F>
struct MathOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_arithmetic<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


template <typename F>
struct BitwiseOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_integral<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_integral<T>::value, gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


template <typename F>
struct LogicalOpDispatcher {
private:
    template <typename T>
    static constexpr bool is_supported() {
        return std::is_arithmetic<T>::value ||
               std::is_same<T, cudf::bool8>::value;

        // TODO: try using member detector
        // std::is_member_function_pointer<decltype(&T::operator!)>::value;
    }

public:
    template <typename T>
    typename std::enable_if_t<is_supported<T>(), gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return cudf::unary::Launcher<T, cudf::bool8, F>::launch(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!is_supported<T>(), gdf_error>
    operator()(gdf_column *input, gdf_column *output) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


gdf_error gdf_unary_math(gdf_column *input, gdf_column *output, gdf_unary_math_op op) {
    cudf::unary::handleChecksAndValidity(input, output);

    switch(op){
        case GDF_SIN:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceSin>{},
                                        input, output);
        case GDF_COS:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceCos>{},
                                        input, output);
        case GDF_TAN:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceTan>{},
                                        input, output);
        case GDF_ARCSIN:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceArcSin>{},
                                        input, output);
        case GDF_ARCCOS:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceArcCos>{},
                                        input, output);
        case GDF_ARCTAN:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceArcTan>{},
                                        input, output);
        case GDF_EXP:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceExp>{},
                                        input, output);
        case GDF_LOG:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceLog>{},
                                        input, output);
        case GDF_SQRT:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceSqrt>{},
                                        input, output);
        case GDF_CEIL:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceCeil>{},
                                        input, output);
        case GDF_FLOOR:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceFloor>{},
                                        input, output);
        case GDF_ABS:
            return cudf::type_dispatcher(input->dtype,
                                        MathOpDispatcher<DeviceAbs>{},
                                        input, output);
        case GDF_BIT_INVERT:
            return cudf::type_dispatcher(input->dtype,
                                        BitwiseOpDispatcher<DeviceInvert>{},
                                        input, output);
        case GDF_NOT:
            return cudf::type_dispatcher(input->dtype,
                                        LogicalOpDispatcher<DeviceNot>{},
                                        input, output);
        default:
            return GDF_INVALID_API_CALL;
    }
}
