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
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace cudf {
namespace experimental {
namespace detail {

// trig functions

template<typename T,
         typename Op,
         typename std::enable_if_t<!std::is_same<T, cudf::experimental::bool8>::value>* = nullptr>
__device__
T normalized_unary_op(T data, Op op) {
    return op(data);
}

template<typename T,
         typename Op,
         typename std::enable_if_t<std::is_same<T, cudf::experimental::bool8>::value>* = nullptr>
__device__
T normalized_unary_op(T data, Op op) {
    return static_cast<T>(op(static_cast<float>(data)));
}

struct DeviceSin {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::sin(e); });
    }
};

struct DeviceCos {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::cos(e); });
    }
};

struct DeviceTan {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::tan(e); });
    }
};

struct DeviceArcSin {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::asin(e); });
    }
};

struct DeviceArcCos {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::acos(e); });
    }
};

struct DeviceArcTan {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::atan(e); });
    }
};

// exponential functions

struct DeviceExp {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::exp(e); });
    }
};

struct DeviceLog {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::log(e); });
    }
};

struct DeviceSqrt {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::sqrt(e); });
    }
};

// rounding functions

struct DeviceCeil {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::ceil(e); });
    }
};

struct DeviceFloor {
    template<typename T>
    __device__
    T operator()(T data) {
        return normalized_unary_op(data, [] (auto e) { return std::floor(e); });
    }
};

struct DeviceAbs {
    template<typename T>
    __device__
    T operator()(T data) {
        return std::abs(data);
    }
};

// bitwise op

struct DeviceInvert {
    // TODO: maybe sfinae overload this for cudf::experimental::bool8
    template<typename T>
    __device__
    T operator()(T data) {
        return ~data;
    }
};

// logical op

struct DeviceNot {
    template<typename T>
    __device__
    cudf::experimental::bool8 operator()(T data) {
        return static_cast<cudf::experimental::bool8>( !data );
    }
};


template<typename T, typename F>
static void launch(cudf::column_view const& input, cudf::mutable_column_view& output) {
    cudf::experimental::unary::launcher<T, T, F>::launch(input, output);
}


template <typename F>
struct MathOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_arithmetic<T>::value, void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_arithmetic<T>::value, void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};


template <typename F>
struct BitwiseOpDispatcher {
    template <typename T>
    typename std::enable_if_t<std::is_integral<T>::value, void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        launch<T, F>(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!std::is_integral<T>::value, void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};


template <typename F>
struct LogicalOpDispatcher {
private:
    template <typename T>
    static constexpr bool is_supported() {
        return std::is_arithmetic<T>::value ||
               std::is_same<T, cudf::experimental::bool8>::value;

        // TODO: try using member detector
        // std::is_member_function_pointer<decltype(&T::operator!)>::value;
    }

public:
    template <typename T>
    typename std::enable_if_t<is_supported<T>(), void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        cudf::experimental::unary::launcher<T, cudf::experimental::bool8, F>::launch(input, output);
    }

    template <typename T>
    typename std::enable_if_t<!is_supported<T>(), void>
    operator()(cudf::column_view const& input, cudf::mutable_column_view& output) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};

} // namespace detail

std::unique_ptr<cudf::column>
unary_operation(cudf::column_view const& input,
                cudf::experimental::unary_op op,
                cudaStream_t stream,
                rmm::mr::device_memory_resource* mr) {

    std::unique_ptr<cudf::column> output = [&] {
        if (op == cudf::experimental::unary_op::NOT) {

            auto type = cudf::data_type{cudf::BOOL8};
            auto size = input.size();

            return std::make_unique<column>(
                type, size,
                rmm::device_buffer{size * cudf::size_of(type), 0, mr},
                copy_bitmask(input, 0, mr),
                input.null_count());

        } else {
            return cudf::experimental::allocate_like(input);
        }
    } ();

    if (input.size() == 0) return output;

    auto output_view = output->mutable_view();

    switch(op){
        case cudf::experimental::unary_op::SIN:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceSin>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::COS:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceCos>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::TAN:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceTan>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::ARCSIN:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcSin>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::ARCCOS:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcCos>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::ARCTAN:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcTan>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::EXP:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceExp>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::LOG:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceLog>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::SQRT:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceSqrt>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::CEIL:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceCeil>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::FLOOR:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceFloor>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::ABS:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceAbs>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::BIT_INVERT:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::BitwiseOpDispatcher<detail::DeviceInvert>{},
                input, output_view);
            break;
        case cudf::experimental::unary_op::NOT:
            cudf::experimental::type_dispatcher(
                input.type(),
                detail::LogicalOpDispatcher<detail::DeviceNot>{},
                input, output_view);
            break;
        default:
            CUDF_FAIL("Undefined unary operation");
    }
    return output;
}

} // namespace experimental
} // namespace cudf
