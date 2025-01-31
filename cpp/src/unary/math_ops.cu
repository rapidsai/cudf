/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

#include <cmath>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {

// trig functions

struct DeviceSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::sin(data);
  }
};

struct DeviceCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::cos(data);
  }
};

struct DeviceTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::tan(data);
  }
};

struct DeviceArcSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::asin(data);
  }
};

struct DeviceArcCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::acos(data);
  }
};

struct DeviceArcTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::atan(data);
  }
};

struct DeviceSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::sinh(data);
  }
};

struct DeviceCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::cosh(data);
  }
};

struct DeviceTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::tanh(data);
  }
};

struct DeviceArcSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::asinh(data);
  }
};

struct DeviceArcCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::acosh(data);
  }
};

struct DeviceArcTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::atanh(data);
  }
};

// exponential functions

struct DeviceExp {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::exp(data);
  }
};

struct DeviceLog {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::log(data);
  }
};

struct DeviceSqrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::sqrt(data);
  }
};

struct DeviceCbrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::cbrt(data);
  }
};

// rounding functions

struct DeviceCeil {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::ceil(data);
  }
};

struct DeviceFloor {
  template <typename T>
  __device__ T operator()(T data)
  {
    return std::floor(data);
  }
};

struct DeviceAbs {
  template <typename T>
  std::enable_if_t<std::is_signed_v<T>, T> __device__ operator()(T data)
  {
    return std::abs(data);
  }
  template <typename T>
  std::enable_if_t<!std::is_signed_v<T>, T> __device__ operator()(T data)
  {
    return data;
  }
};

struct DeviceRInt {
  template <typename T>
  std::enable_if_t<std::is_floating_point_v<T>, T> __device__ operator()(T data)
  {
    return std::rint(data);
  }

  // Dummy to handle other types, will never be executed
  template <typename T>
  std::enable_if_t<!std::is_floating_point_v<T>, T> __device__ operator()(T data)
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

// negation

struct DeviceNegate {
  template <typename T>
  T __device__ operator()(T data)
  {
    return -data;
  }
};

// fixed_point ops

/*
 * Ceiling is calculated using integer division. When we divide by `n`, we get the integer part of
 * the `fixed_point` number. For a negative number, this is all that is needed since the ceiling
 * operation is defined as the least integer greater than the value. For a positive number, we may
 * need to round up if the `fixed_point` number has a fractional part. This is handled by comparing
 * the truncated value to the original value and if they are not equal, the result needs to be
 * incremented by `n`.
 */
template <typename T>
struct fixed_point_ceil {
  T n;  // 10^-scale (value required to determine integer part of fixed_point number)
  __device__ T operator()(T data)
  {
    T const a = (data / n) * n;                  // result of integer division
    return a + (data > 0 && a != data ? n : 0);  // add 1 if positive and not round number
  }
};

/*
 * Floor is calculated using integer division. When we divide by `n`, we get the integer part of
 * the `fixed_point` number. For a positive number, this is all that is needed since the floor
 * operation is defined as the greatest integer less than the value. For a negative number, we may
 * need to round down if the `fixed_point` number has a fractional part. This is handled by
 * comparing the truncated value to the original value and if they are not equal, the result needs
 * to be decremented by `n`.
 */
template <typename T>
struct fixed_point_floor {
  T n;  // 10^-scale (value required to determine integer part of fixed_point number)
  __device__ T operator()(T data)
  {
    T const a = (data / n) * n;                  // result of integer division
    return a - (data < 0 && a != data ? n : 0);  // subtract 1 if negative and not round number
  }
};

template <typename T>
struct fixed_point_abs {
  T n;
  __device__ T operator()(T data) { return numeric::detail::abs(data); }
};

template <typename T>
struct fixed_point_negate {
  T n;
  __device__ T operator()(T data) { return -data; }
};

template <typename T, template <typename> typename FixedPointFunctor>
std::unique_ptr<column> unary_op_with(column_view const& input,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using Type                     = device_storage_type_t<T>;
  using FixedPointUnaryOpFunctor = FixedPointFunctor<Type>;

  // When scale is >= 0 and unary_operator is CEIL or FLOOR, the unary_operation is a no-op
  if (input.type().scale() >= 0 &&
      (std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_ceil<Type>> ||
       std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_floor<Type>>))
    return std::make_unique<cudf::column>(input, stream, mr);

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();

  Type n = 10;
  for (int i = 1; i < -input.type().scale(); ++i) {
    n *= 10;
  }

  thrust::transform(rmm::exec_policy(stream),
                    input.begin<Type>(),
                    input.end<Type>(),
                    out_view.begin<Type>(),
                    FixedPointUnaryOpFunctor{n});

  result->set_null_count(input.null_count());

  return result;
}

template <typename OutputType, typename UFN, typename InputIterator>
std::unique_ptr<cudf::column> transform_fn(InputIterator begin,
                                           InputIterator end,
                                           rmm::device_buffer&& null_mask,
                                           size_type null_count,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const size = cudf::distance(begin, end);

  std::unique_ptr<cudf::column> output =
    make_fixed_width_column(data_type{type_to_id<OutputType>()},
                            size,
                            std::forward<rmm::device_buffer>(null_mask),
                            null_count,
                            stream,
                            mr);
  if (size == 0) return output;

  auto output_view = output->mutable_view();
  thrust::transform(rmm::exec_policy(stream), begin, end, output_view.begin<OutputType>(), UFN{});
  output->set_null_count(null_count);
  return output;
}

template <typename T, typename UFN>
std::unique_ptr<cudf::column> transform_fn(cudf::dictionary_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto dictionary_view = cudf::column_device_view::create(input.parent(), stream);
  auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
  auto default_mr      = cudf::get_current_device_resource_ref();
  // call unary-op using temporary output buffer
  auto output = transform_fn<T, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     detail::copy_bitmask(input.parent(), stream, default_mr),
                                     input.null_count(),
                                     stream,
                                     default_mr);
  return cudf::dictionary::detail::encode(
    output->view(), dictionary::detail::get_indices_type_for_size(output->size()), stream, mr);
}

template <typename UFN>
struct MathOpDispatcher {
  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return transform_fn<T, UFN>(input.begin<T>(),
                                input.end<T>(),
                                cudf::detail::copy_bitmask(input, stream, mr),
                                input.null_count(),
                                stream,
                                mr);
  }

  struct dictionary_dispatch {
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
    {
      return transform_fn<T, UFN>(input, stream, mr);
    }

    template <typename T, typename... Args>
    std::enable_if_t<!std::is_arithmetic_v<T>, std::unique_ptr<cudf::column>> operator()(Args&&...)
    {
      CUDF_FAIL("dictionary keys must be numeric for this operation");
    }
  };

  template <
    typename T,
    std::enable_if_t<!std::is_arithmetic_v<T> and std::is_same_v<T, dictionary32>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    if (input.is_empty()) return empty_like(input);
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic_v<T> and !std::is_same_v<T, dictionary32>,
                   std::unique_ptr<cudf::column>>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported data type for operation");
  }
};

template <typename UFN>
struct NegateOpDispatcher {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_signed_v<T> || cudf::is_duration<T>();
  }

  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return transform_fn<T, UFN>(input.begin<T>(),
                                input.end<T>(),
                                cudf::detail::copy_bitmask(input, stream, mr),
                                input.null_count(),
                                stream,
                                mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<cudf::column>> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported data type for negate operation");
  }
};

template <typename UFN>
struct BitwiseOpDispatcher {
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return transform_fn<T, UFN>(input.begin<T>(),
                                input.end<T>(),
                                cudf::detail::copy_bitmask(input, stream, mr),
                                input.null_count(),
                                stream,
                                mr);
  }

  struct dictionary_dispatch {
    template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
    {
      return transform_fn<T, UFN>(input, stream, mr);
    }

    template <typename T, typename... Args>
    std::enable_if_t<!std::is_integral_v<T>, std::unique_ptr<cudf::column>> operator()(Args&&...)
    {
      CUDF_FAIL("dictionary keys type not supported for this operation");
    }
  };

  template <typename T,
            std::enable_if_t<!std::is_integral_v<T> and std::is_same_v<T, dictionary32>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    if (input.is_empty()) return empty_like(input);
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_integral_v<T> and !std::is_same_v<T, dictionary32>,
                   std::unique_ptr<cudf::column>>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

template <typename UFN>
struct LogicalOpDispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic_v<T> || std::is_same_v<T, bool>;
  }

 public:
  template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return transform_fn<bool, UFN>(input.begin<T>(),
                                   input.end<T>(),
                                   cudf::detail::copy_bitmask(input, stream, mr),
                                   input.null_count(),

                                   stream,
                                   mr);
  }

  struct dictionary_dispatch {
    template <typename T, std::enable_if_t<is_supported<T>()>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
    {
      auto dictionary_view = cudf::column_device_view::create(input.parent(), stream);
      auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
      return transform_fn<bool, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
    }

    template <typename T, typename... Args>
    std::enable_if_t<!is_supported<T>(), std::unique_ptr<cudf::column>> operator()(Args&&...)
    {
      CUDF_FAIL("dictionary keys type not supported for this operation");
    }
  };

  template <typename T,
            std::enable_if_t<!is_supported<T>() and std::is_same_v<T, dictionary32>>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    if (input.is_empty()) return make_empty_column(cudf::data_type{cudf::type_id::BOOL8});
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, stream, mr);
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>() and !std::is_same_v<T, dictionary32>,
                   std::unique_ptr<cudf::column>>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

struct FixedPointOpDispatcher {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_fixed_point<T>(), std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("FixedPointOpDispatcher only for fixed_point");
  }

  template <typename T>
  std::enable_if_t<cudf::is_fixed_point<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    cudf::unary_operator op,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // clang-format off
    switch (op) {
      case cudf::unary_operator::CEIL:   return unary_op_with<T, fixed_point_ceil>(input, stream, mr);
      case cudf::unary_operator::FLOOR:  return unary_op_with<T, fixed_point_floor>(input, stream, mr);
      case cudf::unary_operator::ABS:    return unary_op_with<T, fixed_point_abs>(input, stream, mr);
      case cudf::unary_operator::NEGATE: return unary_op_with<T, fixed_point_negate>(input, stream, mr);
      default: CUDF_FAIL("Unsupported fixed_point unary operation");
    }
    // clang-format on
  }
};

}  // namespace

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  if (cudf::is_fixed_point(input.type()))
    return type_dispatcher(input.type(), detail::FixedPointOpDispatcher{}, input, op, stream, mr);

  switch (op) {
    case cudf::unary_operator::SIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSin>{}, input, stream, mr);
    case cudf::unary_operator::COS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCos>{}, input, stream, mr);
    case cudf::unary_operator::TAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTan>{}, input, stream, mr);
    case cudf::unary_operator::ARCSIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSin>{}, input, stream, mr);
    case cudf::unary_operator::ARCCOS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCos>{}, input, stream, mr);
    case cudf::unary_operator::ARCTAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTan>{}, input, stream, mr);
    case cudf::unary_operator::SINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSinH>{}, input, stream, mr);
    case cudf::unary_operator::COSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCosH>{}, input, stream, mr);
    case cudf::unary_operator::TANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTanH>{}, input, stream, mr);
    case cudf::unary_operator::ARCSINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSinH>{}, input, stream, mr);
    case cudf::unary_operator::ARCCOSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCosH>{}, input, stream, mr);
    case cudf::unary_operator::ARCTANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTanH>{}, input, stream, mr);
    case cudf::unary_operator::EXP:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceExp>{}, input, stream, mr);
    case cudf::unary_operator::LOG:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceLog>{}, input, stream, mr);
    case cudf::unary_operator::SQRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSqrt>{}, input, stream, mr);
    case cudf::unary_operator::CBRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCbrt>{}, input, stream, mr);
    case cudf::unary_operator::CEIL:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCeil>{}, input, stream, mr);
    case cudf::unary_operator::FLOOR:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceFloor>{}, input, stream, mr);
    case cudf::unary_operator::ABS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceAbs>{}, input, stream, mr);
    case cudf::unary_operator::RINT:
      CUDF_EXPECTS(
        (input.type().id() == type_id::FLOAT32) or (input.type().id() == type_id::FLOAT64),
        "rint expects floating point values");
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceRInt>{}, input, stream, mr);
    case cudf::unary_operator::BIT_INVERT:
      return cudf::type_dispatcher(
        input.type(), detail::BitwiseOpDispatcher<detail::DeviceInvert>{}, input, stream, mr);
    case cudf::unary_operator::NOT:
      return cudf::type_dispatcher(
        input.type(), detail::LogicalOpDispatcher<detail::DeviceNot>{}, input, stream, mr);
    case cudf::unary_operator::NEGATE:
      return cudf::type_dispatcher(
        input.type(), detail::NegateOpDispatcher<detail::DeviceNegate>{}, input, stream, mr);
    default: CUDF_FAIL("Undefined unary operation");
  }
}

}  // namespace detail

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::unary_operation(input, op, stream, mr);
}

}  // namespace cudf
