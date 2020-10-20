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
// #include <cudf/scalar/scalar_factories.hpp>
// #include <cudf/utilities/error.hpp>
// #include <cudf/utilities/traits.hpp>
// #include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

namespace detail {

struct round_fn {
  template <typename T, typename... Args>
  std::enable_if_t<not cudf::is_floating_point<T>(), std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("fail for the moment");
  }

  template <typename T, typename... Args>
  std::enable_if_t<cudf::is_floating_point<T>(), std::unique_ptr<column>> operator()(
    column_view const& col,
    int32_t scale,
    cudf::round_option round,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    auto result       = cudf::make_fixed_width_column(col.type(), col.size());
    auto out_view     = result->mutable_view();
    auto const factor = static_cast<int32_t>(std::pow(10, scale));

    thrust::transform(
      rmm::exec_policy(stream)->on(stream),
      col.begin<T>(),
      col.end<T>(),
      out_view.begin<T>(),
      [factor] __device__(auto e) { return static_cast<T>(std::roundf(e * factor) / factor); });

    return result;
  }
};

// TODO docs
std::unique_ptr<column> round(column_view const& col,
                              int32_t scale,
                              cudf::round_option round,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_EXPECTS(round == round_option::HALF_UP, "HALF_EVEN currently not supported.");
  CUDF_EXPECTS(scale >= 0, "Only positive scales currently supported.");
  CUDF_EXPECTS(cudf::is_floating_point(col.type()), "Only floating point currently supported.");

  // TODO when fixed_point supported, have to adjust type
  if (col.size() == 0) return empty_like(col);

  return type_dispatcher(col.type(), round_fn{}, col, scale, round, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> round(column_view const& col,
                              int32_t scale,
                              round_option round,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::round(col, scale, round, mr);
}

}  // namespace cudf
