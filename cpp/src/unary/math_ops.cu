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

#include <unary/unary_ops.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/unary.hpp>

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
static std::unique_ptr<cudf::column>
launch(cudf::column_view const& input,
       cudf::experimental::unary_op op,
       rmm::mr::device_memory_resource* mr,
       cudaStream_t stream) {
    return cudf::experimental::unary::launcher<T, T, F>::launch(input, op, mr, stream);
}


template <typename F>
struct MathOpDispatcher {
    template <typename T,
              typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
        return launch<T, F>(input, op, mr, stream);
    }

    template <typename T,
              typename std::enable_if_t<!std::is_arithmetic<T>::value>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};


template <typename F>
struct BitwiseOpDispatcher {
    template <typename T,
              typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
        return launch<T, F>(input, op, mr, stream);
    }

    template <typename T,
              typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
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
    template <typename T,
              typename std::enable_if_t<is_supported<T>()>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
        return cudf::experimental::unary::launcher<T, cudf::experimental::bool8, F>::launch(input, op, mr, stream);
    }

    template <typename T,
              typename std::enable_if_t<!is_supported<T>()>* = nullptr>
    std::unique_ptr<cudf::column>
    operator()(cudf::column_view const& input,
               cudf::experimental::unary_op op,
               rmm::mr::device_memory_resource* mr,
               cudaStream_t stream) {
        CUDF_FAIL("Unsupported datatype for operation");
    }
};

std::unique_ptr<cudf::column>
unary_operation(cudf::column_view const& input,
                cudf::experimental::unary_op op,
                rmm::mr::device_memory_resource* mr,
                cudaStream_t stream) {

    switch(op) {
        case cudf::experimental::unary_op::SIN:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceSin>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::COS:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceCos>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::TAN:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceTan>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::ARCSIN:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcSin>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::ARCCOS:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcCos>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::ARCTAN:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceArcTan>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::EXP:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceExp>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::LOG:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceLog>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::SQRT:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceSqrt>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::CEIL:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceCeil>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::FLOOR:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceFloor>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::ABS:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::MathOpDispatcher<detail::DeviceAbs>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::BIT_INVERT:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::BitwiseOpDispatcher<detail::DeviceInvert>{},
                input, op, mr, stream);
        case cudf::experimental::unary_op::NOT:
            return cudf::experimental::type_dispatcher(
                input.type(),
                detail::LogicalOpDispatcher<detail::DeviceNot>{},
                input, op, mr, stream);
        default:
            CUDF_FAIL("Undefined unary operation");
    }
}

} // namespace detail

std::unique_ptr<cudf::column>
unary_operation(cudf::column_view const& input,
                cudf::experimental::unary_op op,
                rmm::mr::device_memory_resource* mr)
{
    return detail::unary_operation(input, op, mr);
}

} // namespace experimental
} // namespace cudf
