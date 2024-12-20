/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/round.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/temporary.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <type_traits>

namespace cudf {
namespace detail {
namespace {  // anonymous

inline float __device__ generic_round(float f) { return roundf(f); }
inline double __device__ generic_round(double d) { return ::round(d); }

inline float __device__ generic_round_half_even(float f) { return rintf(f); }
inline double __device__ generic_round_half_even(double d) { return rint(d); }

inline float __device__ generic_modf(float a, float* b) { return modff(a, b); }
inline double __device__ generic_modf(double a, double* b) { return modf(a, b); }

template <typename T, std::enable_if_t<cuda::std::is_signed_v<T>>* = nullptr>
T __device__ generic_abs(T value)
{
  return numeric::detail::abs(value);
}

template <typename T, std::enable_if_t<not cuda::std::is_signed_v<T>>* = nullptr>
T __device__ generic_abs(T value)
{
  return value;
}

template <typename T, std::enable_if_t<cuda::std::is_signed_v<T>>* = nullptr>
int16_t __device__ generic_sign(T value)
{
  return value < 0 ? -1 : 1;
}

// this is needed to suppress warning: pointless comparison of unsigned integer with zero
template <typename T, std::enable_if_t<not cuda::std::is_signed_v<T>>* = nullptr>
int16_t __device__ generic_sign(T)
{
  return 1;
}

template <typename T>
constexpr inline auto is_supported_round_type()
{
  return (cudf::is_numeric<T>() && not std::is_same_v<T, bool>) || cudf::is_fixed_point<T>();
}

template <typename T>
struct half_up_zero {
  T n;  // unused in the decimal_places = 0 case
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round(e);
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct half_up_positive {
  T n;
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round(fractional_part * n) / n;
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct half_up_negative {
  T n;
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round(e / n) * n;
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U e)
  {
    auto const down = (e / n) * n;  // result from rounding down
    return down + generic_sign(e) * (generic_abs(e - down) >= n / 2 ? n : 0);
  }
};

template <typename T>
struct half_even_zero {
  T n;  // unused in the decimal_places = 0 case
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round_half_even(e);
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct half_even_positive {
  T n;
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round_half_even(fractional_part * n) / n;
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct half_even_negative {
  T n;
  template <typename U = T, std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round_half_even(e / n) * n;
  }

  template <typename U = T, std::enable_if_t<cuda::std::is_integral_v<U>>* = nullptr>
  __device__ U operator()(U e)
  {
    auto const down_over_n = e / n;            // use this to determine HALF_EVEN case
    auto const down        = down_over_n * n;  // result from rounding down
    auto const diff        = generic_abs(e - down);
    auto const adjustment =
      (diff > n / 2) or (diff == n / 2 && generic_abs(down_over_n) % 2 == 1) ? n : 0;
    return down + generic_sign(e) * adjustment;
  }
};

template <typename T>
struct half_up_fixed_point {
  T n;
  __device__ T operator()(T e) { return half_up_negative<T>{n}(e) / n; }
};

template <typename T>
struct half_even_fixed_point {
  T n;
  __device__ T operator()(T e) { return half_even_negative<T>{n}(e) / n; }
};

template <typename T,
          template <typename>
          typename RoundFunctor,
          std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
std::unique_ptr<column> round_with(column_view const& input,
                                   int32_t decimal_places,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  using Functor = RoundFunctor<T>;

  if (decimal_places >= 0 && std::is_integral_v<T>)
    return std::make_unique<cudf::column>(input, stream, mr);

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();
  T const n     = std::pow(10, std::abs(decimal_places));

  thrust::transform(
    rmm::exec_policy(stream), input.begin<T>(), input.end<T>(), out_view.begin<T>(), Functor{n});

  result->set_null_count(input.null_count());

  return result;
}

template <typename T,
          template <typename>
          typename RoundFunctor,
          std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
std::unique_ptr<column> round_with(column_view const& input,
                                   int32_t decimal_places,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  using namespace numeric;
  using Type                   = device_storage_type_t<T>;
  using FixedPointRoundFunctor = RoundFunctor<Type>;

  if (input.type().scale() == -decimal_places)
    return std::make_unique<cudf::column>(input, stream, mr);

  auto const result_type = data_type{input.type().id(), scale_type{-decimal_places}};

  // if rounding to more precision than fixed_point is capable of, just need to rescale
  // note: decimal_places has the opposite sign of numeric::scale_type (therefore have to negate)
  if (input.type().scale() > -decimal_places)
    return cudf::detail::cast(input, result_type, stream, mr);

  auto result = cudf::make_fixed_width_column(result_type,
                                              input.size(),
                                              detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();

  auto const scale_movement = -decimal_places - input.type().scale();
  // If scale_movement is larger than max precision of current type, the pow operation will
  // overflow. Under this circumstance, we can simply output a zero column because no digits can
  // survive such a large scale movement.
  if (scale_movement > cuda::std::numeric_limits<Type>::digits10) {
    thrust::uninitialized_fill(rmm::exec_policy(stream),
                               out_view.template begin<Type>(),
                               out_view.template end<Type>(),
                               static_cast<Type>(0));
  } else {
    Type n = 10;
    for (int i = 1; i < scale_movement; ++i) {
      n *= 10;
    }
    thrust::transform(rmm::exec_policy(stream),
                      input.begin<Type>(),
                      input.end<Type>(),
                      out_view.begin<Type>(),
                      FixedPointRoundFunctor{n});
  }

  result->set_null_count(input.null_count());

  return result;
}

struct round_type_dispatcher {
  template <typename T, typename... Args>
  std::enable_if_t<not is_supported_round_type<T>(), std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Type not support for cudf::round");
  }

  template <typename T>
  std::enable_if_t<is_supported_round_type<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // clang-format off
    switch (method) {
      case cudf::rounding_method::HALF_UP:
        if      (is_fixed_point<T>()) return round_with<T, half_up_fixed_point>(input, decimal_places, stream, mr);
        else if (decimal_places == 0) return round_with<T, half_up_zero       >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_up_positive   >(input, decimal_places, stream, mr);
        else                          return round_with<T, half_up_negative   >(input, decimal_places, stream, mr);
      case cudf::rounding_method::HALF_EVEN:
        if      (is_fixed_point<T>()) return round_with<T, half_even_fixed_point>(input, decimal_places, stream, mr);
        else if (decimal_places == 0) return round_with<T, half_even_zero       >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_even_positive   >(input, decimal_places, stream, mr);
        else                          return round_with<T, half_even_negative   >(input, decimal_places, stream, mr);
      default: CUDF_FAIL("Undefined rounding method");
    }
    // clang-format on
  }
};

}  // anonymous namespace

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              cudf::rounding_method method,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_numeric(input.type()) || cudf::is_fixed_point(input.type()),
               "Only integral/floating point/fixed point currently supported.");

  if (input.is_empty()) {
    if (is_fixed_point(input.type())) {
      auto const type = data_type{input.type().id(), numeric::scale_type{-decimal_places}};
      return make_empty_column(type);
    }
    return empty_like(input);
  }

  return type_dispatcher(
    input.type(), round_type_dispatcher{}, input, decimal_places, method, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              rounding_method method,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::round(input, decimal_places, method, stream, mr);
}

}  // namespace cudf
