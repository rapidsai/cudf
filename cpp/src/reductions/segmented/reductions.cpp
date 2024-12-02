/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

struct segmented_reduce_dispatch_functor {
  column_view const& col;
  device_span<size_type const> offsets;
  data_type output_dtype;
  null_policy null_handling;
  std::optional<std::reference_wrapper<scalar const>> init;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  segmented_reduce_dispatch_functor(column_view const& segmented_values,
                                    device_span<size_type const> offsets,
                                    data_type output_dtype,
                                    null_policy null_handling,
                                    std::optional<std::reference_wrapper<scalar const>> init,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
    : col(segmented_values),
      offsets(offsets),
      output_dtype(output_dtype),
      null_handling(null_handling),
      init(init),
      stream(stream),
      mr(mr)
  {
  }

  segmented_reduce_dispatch_functor(column_view const& segmented_values,
                                    device_span<size_type const> offsets,
                                    data_type output_dtype,
                                    null_policy null_handling,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
    : segmented_reduce_dispatch_functor(
        segmented_values, offsets, output_dtype, null_handling, std::nullopt, stream, mr)
  {
  }

  template <segmented_reduce_aggregation::Kind k>
  std::unique_ptr<column> operator()(segmented_reduce_aggregation const& agg)
  {
    switch (k) {
      case segmented_reduce_aggregation::SUM:
        return segmented_sum(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::PRODUCT:
        return segmented_product(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::MIN:
        return segmented_min(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::MAX:
        return segmented_max(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::ANY:
        return segmented_any(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::ALL:
        return segmented_all(col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::SUM_OF_SQUARES:
        return segmented_sum_of_squares(col, offsets, output_dtype, null_handling, stream, mr);
      case segmented_reduce_aggregation::MEAN:
        return segmented_mean(col, offsets, output_dtype, null_handling, stream, mr);
      case segmented_reduce_aggregation::VARIANCE: {
        auto var_agg = static_cast<cudf::detail::var_aggregation const&>(agg);
        return segmented_variance(
          col, offsets, output_dtype, null_handling, var_agg._ddof, stream, mr);
      }
      case segmented_reduce_aggregation::STD: {
        auto var_agg = static_cast<cudf::detail::std_aggregation const&>(agg);
        return segmented_standard_deviation(
          col, offsets, output_dtype, null_handling, var_agg._ddof, stream, mr);
      }
      case segmented_reduce_aggregation::NUNIQUE:
        return segmented_nunique(col, offsets, null_handling, stream, mr);
      default: CUDF_FAIL("Unsupported aggregation type.");
    }
  }
};

std::unique_ptr<column> segmented_reduce(column_view const& segmented_values,
                                         device_span<size_type const> offsets,
                                         segmented_reduce_aggregation const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!init.has_value() || cudf::have_same_types(segmented_values, init.value().get()),
               "column and initial value must be the same type",
               cudf::data_type_error);
  if (init.has_value() && !(agg.kind == aggregation::SUM || agg.kind == aggregation::PRODUCT ||
                            agg.kind == aggregation::MIN || agg.kind == aggregation::MAX ||
                            agg.kind == aggregation::ANY || agg.kind == aggregation::ALL)) {
    CUDF_FAIL(
      "Initial value is only supported for SUM, PRODUCT, MIN, MAX, ANY, and ALL aggregation types");
  }
  CUDF_EXPECTS(offsets.size() > 0, "`offsets` should have at least 1 element.");

  return cudf::detail::aggregation_dispatcher(
    agg.kind,
    segmented_reduce_dispatch_functor{
      segmented_values, offsets, output_dtype, null_handling, init, stream, mr},
    agg);
}
}  // namespace
}  // namespace detail
}  // namespace reduction

std::unique_ptr<column> segmented_reduce(column_view const& segmented_values,
                                         device_span<size_type const> offsets,
                                         segmented_reduce_aggregation const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::segmented_reduce(
    segmented_values, offsets, agg, output_dtype, null_handling, std::nullopt, stream, mr);
}

std::unique_ptr<column> segmented_reduce(column_view const& segmented_values,
                                         device_span<size_type const> offsets,
                                         segmented_reduce_aggregation const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::segmented_reduce(
    segmented_values, offsets, agg, output_dtype, null_handling, init, stream, mr);
}

}  // namespace cudf
