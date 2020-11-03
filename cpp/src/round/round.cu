/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/round.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>

namespace cudf {
namespace detail {
namespace {  // anonymous

float __device__ generic_round(float f) { return roundf(f); }
double __device__ generic_round(double d) { return ::round(d); }

float __device__ generic_round_half_even(float f) { return rintf(f); }
double __device__ generic_round_half_even(double d) { return ::rint(d); }

float __device__ generic_modf(float a, float* b) { return modff(a, b); }
double __device__ generic_modf(double a, double* b) { return modf(a, b); }

template <typename T, typename std::enable_if_t<std::is_signed<T>::value>* = nullptr>
T __device__ generic_abs(T value)
{
  return abs(value);
}

template <typename T, typename std::enable_if_t<not std::is_signed<T>::value>* = nullptr>
T __device__ generic_abs(T value)
{
  return value;
}

template <typename T, typename std::enable_if_t<std::is_signed<T>::value>* = nullptr>
bool __device__ is_negative(T value)
{
  return value < 0;
}

// this is needed to suppress warning: pointless comparison of unsigned integer with zero
template <typename T, typename std::enable_if_t<not std::is_signed<T>::value>* = nullptr>
bool __device__ is_negative(T)
{
  return false;
}

template <typename T>
struct HalfUpZero {
  T n;  // unused in the decimal_places = 0 case
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round(e);
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct HalfUpPositive {
  T n;
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round(fractional_part * n) / n;
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct HalfUpNegative {
  T n;
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round(e / n) * n;
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    auto const down = (e / n) * n;  // result from rounding down
    auto const sign = is_negative(e) ? -1 : 1;
    return down + sign * (generic_abs(e - down) >= n / 2 ? n : 0);
  }
};

template <typename T>
struct HalfEvenZero {
  T n;  // unused in the decimal_places = 0 case
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round_half_even(e);
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct HalfEvenPositive {
  T n;
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round_half_even(fractional_part * n) / n;
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    assert(false);  // Should never get here. Just for compilation
    return U{};
  }
};

template <typename T>
struct HalfEvenNegative {
  T n;
  template <typename U = T, typename std::enable_if_t<cudf::is_floating_point<U>()>* = nullptr>
  __device__ U operator()(U e)
  {
    return generic_round_half_even(e / n) * n;
  }

  template <typename U = T, typename std::enable_if_t<std::is_integral<U>::value>* = nullptr>
  __device__ U operator()(U e)
  {
    assert(false);  // TODO support
    return U{};
  }
};

template <typename T, typename RoundFunctor>
std::unique_ptr<column> round_with(column_view const& input,
                                   int32_t decimal_places,
                                   cudaStream_t stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (decimal_places >= 0 && std::is_integral<T>::value)
    return std::make_unique<cudf::column>(input, stream, mr);

  auto result = cudf::make_fixed_width_column(input.type(),  //
                                              input.size(),
                                              copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();
  T const n     = std::pow(10, std::abs(decimal_places));

  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    input.begin<T>(),
                    input.end<T>(),
                    out_view.begin<T>(),
                    RoundFunctor{n});

  return result;
}

struct round_type_dispatcher {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_numeric<T>(), std::unique_ptr<column>> operator()(Args&&... args)
  {
    CUDF_FAIL("Type not support for cudf::round");
  }

  template <typename T>
  std::enable_if_t<cudf::is_numeric<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
  {
    // clang-format off
    switch (method) {
      case cudf::rounding_method::HALF_UP:
        if      (decimal_places == 0) return round_with<T, HalfUpZero    <T>>(input, decimal_places, stream, mr);
        else if (decimal_places  > 0) return round_with<T, HalfUpPositive<T>>(input, decimal_places, stream, mr);
        else                          return round_with<T, HalfUpNegative<T>>(input, decimal_places, stream, mr);
      case cudf::rounding_method::HALF_EVEN:
        if      (decimal_places == 0) return round_with<T, HalfEvenZero    <T>>(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, HalfEvenPositive<T>>(input, decimal_places, stream, mr);
        else                          return round_with<T, HalfEvenNegative<T>>(input, decimal_places, stream, mr);
      default: CUDF_FAIL("Undefined rounding method");
    }
    // clang-format on
  }
};

};  // anonymous namespace

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              cudf::rounding_method method,
                              cudaStream_t stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_numeric(input.type()), "Only integral/floating point currently supported.");

  // TODO when fixed_point supported, have to adjust type
  if (input.size() == 0) return empty_like(input);

  return type_dispatcher(
    input.type(), round_type_dispatcher{}, input, decimal_places, method, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              rounding_method method,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::round(input, decimal_places, method, 0, mr);
}

}  // namespace cudf
