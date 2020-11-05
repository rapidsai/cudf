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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/round.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {  // anonymous

float __device__ generic_round(float f) { return roundf(f); }
double __device__ generic_round(double d) { return ::round(d); }

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
bool __device__ is_negative(T value)
{
  return false;
}

struct round_fn {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_numeric<T>(), std::unique_ptr<column>> operator()(Args&&... args)
  {
    CUDF_FAIL("Type not support for cudf::round");
  }

  template <typename T>
  std::enable_if_t<cudf::is_floating_point<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto result = cudf::make_fixed_width_column(input.type(),  //
                                                input.size(),
                                                copy_bitmask(input, stream, mr),
                                                input.null_count(),
                                                stream,
                                                mr);

    auto out_view = result->mutable_view();
    T const n     = std::pow(10, std::abs(decimal_places));

    if (decimal_places == 0)
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [] __device__(T e) -> T { return generic_round(e); });
    else if (decimal_places > 0)
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [n] __device__(T e) -> T {
                          T integer_part;
                          T const fractional_part = generic_modf(e, &integer_part);
                          return integer_part + generic_round(fractional_part * n) / n;
                        });
    else  // decimal_places < 0
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [n] __device__(T e) -> T { return generic_round(e / n) * n; });

    return result;
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr)
  {
    // only need to handle the case where decimal_places is < zero
    // integers by definition have no fractional part, so result of "rounding" is a no-op
    if (decimal_places >= 0) return std::make_unique<cudf::column>(input, stream, mr);

    auto result = cudf::make_fixed_width_column(input.type(),  //
                                                input.size(),
                                                detail::copy_bitmask(input, stream, mr),
                                                input.null_count(),
                                                stream,
                                                mr);

    auto out_view = result->mutable_view();
    auto const n  = static_cast<T>(std::pow(10, -decimal_places));

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      input.begin<T>(),
                      input.end<T>(),
                      out_view.begin<T>(),
                      [n] __device__(T e) -> T {
                        auto const down = (e / n) * n;  // result from rounding down
                        auto const sign = is_negative(e) ? -1 : 1;
                        return down + sign * (generic_abs(e - down) >= n / 2 ? n : 0);
                      });

    return result;
  }
};

}  // anonymous namespace

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              cudf::rounding_method method,
                              cudaStream_t stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_numeric(input.type()), "Only integral/floating point currently supported.");

  // TODO when fixed_point supported, have to adjust type
  if (input.size() == 0) return empty_like(input);

  return type_dispatcher(input.type(), round_fn{}, input, decimal_places, method, stream, mr);
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
