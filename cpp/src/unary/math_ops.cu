/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

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

template <typename OutputType, typename UFN, typename InputIterator>
std::unique_ptr<cudf::column> transform_fn(InputIterator begin,
                                           InputIterator end,
                                           rmm::device_buffer&& null_mask,
                                           size_type null_count,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
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
  thrust::transform(
    rmm::exec_policy(stream)->on(stream), begin, end, output_view.begin<OutputType>(), UFN{});
  return output;
}

template <typename T, typename UFN>
std::unique_ptr<cudf::column> transform_fn(cudf::dictionary_column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
{
  auto dictionary_view = cudf::column_device_view::create(input.parent(), stream);
  auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
  auto default_mr      = rmm::mr::get_current_device_resource();
  // call unary-op using temporary output buffer
  auto output = transform_fn<T, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     copy_bitmask(input.parent(), stream, default_mr),
                                     input.null_count(),
                                     default_mr,
                                     stream);
  return cudf::dictionary::detail::encode(
    output->view(), dictionary::detail::get_indices_type_for_size(output->size()), mr, stream);
}

template <typename UFN>
struct MathOpDispatcher {
  template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return transform_fn<T, UFN>(input.begin<T>(),
                                input.end<T>(),
                                copy_bitmask(input, stream, mr),
                                input.null_count(),
                                mr,
                                stream);
  }

  struct dictionary_dispatch {
    template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      return transform_fn<T, UFN>(input, mr, stream);
    }

    template <typename T, typename std::enable_if_t<!std::is_arithmetic<T>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      CUDF_FAIL("dictionary keys must be numeric for this operation");
    }
  };

  template <typename T,
            typename std::enable_if_t<!std::is_arithmetic<T>::value and
                                      std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    if (input.is_empty()) return empty_like(input);
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, mr, stream);
  }

  template <typename T,
            typename std::enable_if_t<!std::is_arithmetic<T>::value and
                                      !std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported data type for operation");
  }
};

template <typename UFN>
struct BitwiseOpDispatcher {
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return transform_fn<T, UFN>(input.begin<T>(),
                                input.end<T>(),
                                copy_bitmask(input, stream, mr),
                                input.null_count(),
                                mr,
                                stream);
  }

  struct dictionary_dispatch {
    template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      return transform_fn<T, UFN>(input, mr, stream);
    }

    template <typename T, typename std::enable_if_t<!std::is_integral<T>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      CUDF_FAIL("dictionary keys type not supported for this operation");
    }
  };

  template <typename T,
            typename std::enable_if_t<!std::is_integral<T>::value and
                                      std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    if (input.is_empty()) return empty_like(input);
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, mr, stream);
  }

  template <typename T,
            typename std::enable_if_t<!std::is_integral<T>::value and
                                      !std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
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
    return std::is_arithmetic<T>::value || std::is_same<T, bool>::value;
  }

 public:
  template <typename T, typename std::enable_if_t<is_supported<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    return transform_fn<bool, UFN>(input.begin<T>(),
                                   input.end<T>(),
                                   copy_bitmask(input, stream, mr),
                                   input.null_count(),
                                   mr,
                                   stream);
  }

  struct dictionary_dispatch {
    template <typename T, typename std::enable_if_t<is_supported<T>()>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      auto dictionary_view = cudf::column_device_view::create(input.parent(), stream);
      auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
      return transform_fn<bool, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     mr,
                                     stream);
    }

    template <typename T, typename std::enable_if_t<!is_supported<T>()>* = nullptr>
    std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
    {
      CUDF_FAIL("dictionary keys type not supported for this operation");
    }
  };

  template <typename T,
            typename std::enable_if_t<!is_supported<T>() and
                                      std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    if (input.is_empty()) return make_empty_column(cudf::data_type{cudf::type_id::BOOL8});
    auto dictionary_col = dictionary_column_view(input);
    return type_dispatcher(
      dictionary_col.keys().type(), dictionary_dispatch{}, dictionary_col, mr, stream);
  }

  // template <typename T, typename std::enable_if_t<!is_supported<T>()>* = nullptr>
  template <typename T,
            typename std::enable_if_t<!is_supported<T>() and
                                      !std::is_same<T, dictionary32>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

}  // namespace

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_op op,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream)
{
  switch (op) {
    case cudf::unary_op::SIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSin>{}, input, mr, stream);
    case cudf::unary_op::COS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCos>{}, input, mr, stream);
    case cudf::unary_op::TAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTan>{}, input, mr, stream);
    case cudf::unary_op::ARCSIN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSin>{}, input, mr, stream);
    case cudf::unary_op::ARCCOS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCos>{}, input, mr, stream);
    case cudf::unary_op::ARCTAN:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTan>{}, input, mr, stream);
    case cudf::unary_op::SINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSinH>{}, input, mr, stream);
    case cudf::unary_op::COSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCosH>{}, input, mr, stream);
    case cudf::unary_op::TANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceTanH>{}, input, mr, stream);
    case cudf::unary_op::ARCSINH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcSinH>{}, input, mr, stream);
    case cudf::unary_op::ARCCOSH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcCosH>{}, input, mr, stream);
    case cudf::unary_op::ARCTANH:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceArcTanH>{}, input, mr, stream);
    case cudf::unary_op::EXP:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceExp>{}, input, mr, stream);
    case cudf::unary_op::LOG:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceLog>{}, input, mr, stream);
    case cudf::unary_op::SQRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceSqrt>{}, input, mr, stream);
    case cudf::unary_op::CBRT:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCbrt>{}, input, mr, stream);
    case cudf::unary_op::CEIL:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceCeil>{}, input, mr, stream);
    case cudf::unary_op::FLOOR:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceFloor>{}, input, mr, stream);
    case cudf::unary_op::ABS:
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceAbs>{}, input, mr, stream);
    case cudf::unary_op::RINT:
      CUDF_EXPECTS(
        (input.type().id() == type_id::FLOAT32) or (input.type().id() == type_id::FLOAT64),
        "rint expects floating point values");
      return cudf::type_dispatcher(
        input.type(), detail::MathOpDispatcher<detail::DeviceRInt>{}, input, mr, stream);
    case cudf::unary_op::BIT_INVERT:
      return cudf::type_dispatcher(
        input.type(), detail::BitwiseOpDispatcher<detail::DeviceInvert>{}, input, mr, stream);
    case cudf::unary_op::NOT:
      return cudf::type_dispatcher(
        input.type(), detail::LogicalOpDispatcher<detail::DeviceNot>{}, input, mr, stream);
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
