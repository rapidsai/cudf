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
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

namespace detail {

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
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    auto result   = cudf::make_fixed_width_column(input.type(), input.size());
    auto out_view = result->mutable_view();
    auto const n  = std::pow(10, std::abs(decimal_places));

    if (decimal_places == 0)
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [] __device__(T e) -> T { return ::round(e); });
    else if (decimal_places > 0)
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [n] __device__(T e) -> T { return ::round(e * n) / n; });
    else  // decimal_places < 0
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [n] __device__(T e) -> T { return ::round(e / n) * n; });

    return result;
  }

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, std::unique_ptr<column>> operator()(
    column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    auto result = cudf::make_fixed_width_column(input.type(), input.size());

    // only need to handle the case where decimal_places is < zero
    // integers by definition have no fractional part, so result of "rounding" is a no-op
    if (decimal_places < 0) {
      auto out_view = result->mutable_view();
      auto const n  = static_cast<int64_t>(std::pow(10, -decimal_places));
      auto const m  = n / 10;  // need 10 ^ (decimal_places - 1) to isolate rounding_digit

      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        input.begin<T>(),
                        input.end<T>(),
                        out_view.begin<T>(),
                        [n, m] __device__(T e) -> T {
                          auto const rounding_digit = static_cast<int64_t>(fabs((e / m) % 10));
                          auto const digits         = e / n;
                          auto const adjust         = e > 0 ? 1 : -1;
                          return rounding_digit < 5 ? digits * n : (digits + adjust) * n;
                        });
    }

    return result;
  }
};

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              cudf::rounding_method method,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_EXPECTS(cudf::is_numeric(input.type()), "Only integral/floating point currently supported.");

  // TODO when fixed_point supported, have to adjust type
  if (input.size() == 0) return empty_like(input);

  return type_dispatcher(input.type(), round_fn{}, input, decimal_places, method, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              rounding_method method,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::round(input, decimal_places, method, mr);
}

}  // namespace cudf
