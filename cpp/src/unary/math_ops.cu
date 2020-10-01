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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <unary/unary_ops.cuh>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace cudf {
namespace detail {
// trig functions

template <typename T, typename Op>
__device__ T normalized_unary_op(T data, Op op)
{
  return op(data);
}

struct DeviceSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::sin(e); });
  }
};

struct DeviceCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::cos(e); });
  }
};

struct DeviceTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::tan(e); });
  }
};

struct DeviceArcSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::asin(e); });
  }
};

struct DeviceArcCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::acos(e); });
  }
};

struct DeviceArcTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::atan(e); });
  }
};

struct DeviceSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::sinh(e); });
  }
};

struct DeviceCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::cosh(e); });
  }
};

struct DeviceTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::tanh(e); });
  }
};

struct DeviceArcSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::asinh(e); });
  }
};

struct DeviceArcCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::acosh(e); });
  }
};

struct DeviceArcTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::atanh(e); });
  }
};

// exponential functions

struct DeviceExp {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::exp(e); });
  }
};

struct DeviceLog {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::log(e); });
  }
};

struct DeviceSqrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::sqrt(e); });
  }
};

struct DeviceCbrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::cbrt(e); });
  }
};

// rounding functions

struct DeviceCeil {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::ceil(e); });
  }
};

struct DeviceFloor {
  template <typename T>
  __device__ T operator()(T data)
  {
    return normalized_unary_op(data, [](auto e) { return std::floor(e); });
  }
};

struct DeviceAbs {
  template <typename T>
  std::enable_if_t<std::is_signed<T>::value, T> __device__ operator()(T data)
  {
    return std::abs(data);
  }
  template <typename T>
  std::enable_if_t<!std::is_signed<T>::value, T> __device__ operator()(T data)
  {
    return data;
  }
};

struct DeviceRInt {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, T> __device__ operator()(T data)
  {
    return std::rint(data);
  }

  // Dummy to handle other types, will never be executed
  template <typename T>
  std::enable_if_t<!std::is_floating_point<T>::value, T> __device__ operator()(T data)
  {
    return data;
  }
};

// bitwise op

struct DeviceInvert {
  template <typename T>
  __device__ T operator()(T data)
  {
    return ~data;
  }
};

// logical op

struct DeviceNot {
  template <typename T>
  __device__ bool operator()(T data)
  {
    return !data;
  }
};

template <typename T, typename F>
static std::unique_ptr<cudf::column> launch(cudf::column_view const& input,
                                            cudf::unary_op op,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream)
{
  return cudf::unary::launcher<T, T, F>::launch(input, op, mr, stream);
}

template <typename F>
struct MathOpDispatcher {
  template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return launch<T, F>(input, op, mr, stream);
  }

  template <typename T, typename std::enable_if_t<!std::is_arithmetic<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

template <typename F>
struct BitwiseOpDispatcher {
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return launch<T, F>(input, op, mr, stream);
  }

  template <typename T, typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

template <typename F>
struct LogicalOpDispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<T>::value || std::is_same<T, bool>::value;

    // TODO: try using member detector
    // std::is_member_function_pointer<decltype(&T::operator!)>::value;
  }

 public:
  template <typename T, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return cudf::unary::launcher<T, bool, F>::launch(input, op, mr, stream);
  }

  template <typename T, typename std::enable_if_t<!is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::unary_op op,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_op op,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream)
{
  switch (op) {
    case cudf::unary_op::SIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSin>{}, input, op, mr, stream);
    case cudf::unary_op::COS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCos>{}, input, op, mr, stream);
    case cudf::unary_op::TAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTan>{}, input, op, mr, stream);
    case cudf::unary_op::ARCSIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSin>{}, input, op, mr, stream);
    case cudf::unary_op::ARCCOS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCos>{}, input, op, mr, stream);
    case cudf::unary_op::ARCTAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTan>{}, input, op, mr, stream);
    case cudf::unary_op::SINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSinH>{}, input, op, mr, stream);
    case cudf::unary_op::COSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCosH>{}, input, op, mr, stream);
    case cudf::unary_op::TANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTanH>{}, input, op, mr, stream);
    case cudf::unary_op::ARCSINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSinH>{}, input, op, mr, stream);
    case cudf::unary_op::ARCCOSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCosH>{}, input, op, mr, stream);
    case cudf::unary_op::ARCTANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTanH>{}, input, op, mr, stream);
    case cudf::unary_op::EXP:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceExp>{}, input, op, mr, stream);
    case cudf::unary_op::LOG:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceLog>{}, input, op, mr, stream);
    case cudf::unary_op::SQRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSqrt>{}, input, op, mr, stream);
    case cudf::unary_op::CBRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCbrt>{}, input, op, mr, stream);
    case cudf::unary_op::CEIL:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCeil>{}, input, op, mr, stream);
    case cudf::unary_op::FLOOR:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceFloor>{}, input, op, mr, stream);
    case cudf::unary_op::ABS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceAbs>{}, input, op, mr, stream);
    case cudf::unary_op::RINT:
      CUDF_EXPECTS(
        (input.type().id() == type_id::FLOAT32) or (input.type().id() == type_id::FLOAT64),
        "rint expects floating point values");
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceRInt>{}, input, op, mr, stream);
    case cudf::unary_op::BIT_INVERT:
      return cudf::type_dispatcher(
        input.type(), detail::BitwiseOpDispatcher<detail::DeviceInvert>{}, input, op, mr, stream);
    case cudf::unary_op::NOT:
      return cudf::type_dispatcher(
        input.type(), detail::LogicalOpDispatcher<detail::DeviceNot>{}, input, op, mr, stream);
    default: CUDF_FAIL("Undefined unary operation");
  }
}

}  // namespace detail

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_op op,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::unary_operation(input, op, mr);
}

}  // namespace cudf
