/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

struct nunique_scalar_fn {
  template <typename T>
    requires(cudf::is_numeric_not_bool<T>())
  std::unique_ptr<cudf::scalar> operator()(size_type count,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto const value = static_cast<T>(count);
    return cudf::make_fixed_width_scalar<T>(value, stream, mr);
  }

  template <typename T>
    requires(not cudf::is_numeric_not_bool<T>())
  std::unique_ptr<cudf::scalar> operator()(size_type,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("NUNIQUE is not supported for boolean or non-numeric types", std::invalid_argument);
  }
};
}  // namespace

std::unique_ptr<cudf::scalar> nunique(column_view const& col,
                                      cudf::null_policy null_handling,
                                      cudf::data_type const output_dtype,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  size_type count =
    cudf::detail::distinct_count(col, null_handling, nan_policy::NAN_IS_VALID, stream);
  return cudf::type_dispatcher(output_dtype, nunique_scalar_fn{}, count, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
