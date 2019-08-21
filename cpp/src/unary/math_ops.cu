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
#include <cudf/unary.hpp>
#include <cudf/copying.hpp>

#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace cudf {

namespace detail {

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
static void launch(gdf_column const* input, gdf_column *output) {
    cudf::unary::Launcher<T, T, F>::launch(input, output);
}


template <typename F>
struct MathOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, void>
    operator()(gdf_column const* input, gdf_column *output) {
        launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_arithmetic<T>::value, void>
    operator()(gdf_column const* input, gdf_column *output) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};


template <typename F>
struct BitwiseOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_integral<T>::value, void>
    operator()(gdf_column const* input, gdf_column *output) {
        launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_integral<T>::value, void>
    operator()(gdf_column const* input, gdf_column *output) {
        CUDF_FAIL("Unsupported datatype for operation");
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
    typename std::enable_if_t<is_supported<T>(), void>
    operator()(gdf_column const* input, gdf_column *output) {
        cudf::unary::Launcher<T, cudf::bool8, F>::launch(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!is_supported<T>(), void>
    operator()(gdf_column const* input, gdf_column *output) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};

} // namespace detail

gdf_column unary_operation(gdf_column const& input, unary_op op) {

    gdf_column output{};

    if (op == unary_op::NOT)
    {
        // TODO: replace this with a proper column constructor once
        // cudf::column is implemented
        bool allocate_mask = (input.valid != nullptr);
        output = cudf::allocate_column(GDF_BOOL8, input.size, allocate_mask);
    }
    else
        output = cudf::allocate_like(input);

    if (input.size == 0) return output;

    cudf::unary::handleChecksAndValidity(input, output);

    switch(op){
        case unary_op::SIN:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceSin>{},
                &input, &output);
            break;
        case unary_op::COS:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceCos>{},
                &input, &output);
            break;
        case unary_op::TAN:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceTan>{},
                &input, &output);
            break;
        case unary_op::ARCSIN:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceArcSin>{},
                &input, &output);
            break;
        case unary_op::ARCCOS:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceArcCos>{},
                &input, &output);
            break;
        case unary_op::ARCTAN:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceArcTan>{},
                &input, &output);
            break;
        case unary_op::EXP:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceExp>{},
                &input, &output);
            break;
        case unary_op::LOG:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceLog>{},
                &input, &output);
            break;
        case unary_op::SQRT:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceSqrt>{},
                &input, &output);
            break;
        case unary_op::CEIL:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceCeil>{},
                &input, &output);
            break;
        case unary_op::FLOOR:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceFloor>{},
                &input, &output);
            break;
        case unary_op::ABS:
            cudf::type_dispatcher(
                input.dtype,
                detail::MathOpDispatcher<detail::DeviceAbs>{},
                &input, &output);
            break;
        case unary_op::BIT_INVERT:
            cudf::type_dispatcher(
                input.dtype,
                detail::BitwiseOpDispatcher<detail::DeviceInvert>{},
                &input, &output);
            break;
        case unary_op::NOT:
            cudf::type_dispatcher(
                input.dtype,
                detail::LogicalOpDispatcher<detail::DeviceNot>{},
                &input, &output);
            break;
        default:
            CUDF_FAIL("Undefined unary operation");
    }
    return output;
}

} // namespace cudf
