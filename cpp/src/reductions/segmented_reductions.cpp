/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
struct segmented_reduce_dispatch_functor {
  column_view const& col;
  device_span<size_type const> offsets;
  data_type output_dtype;
  null_policy null_handling;
  std::optional<std::reference_wrapper<scalar const>> init;
  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;

  segmented_reduce_dispatch_functor(column_view const& segmented_values,
                                    device_span<size_type const> offsets,
                                    data_type output_dtype,
                                    null_policy null_handling,
                                    std::optional<std::reference_wrapper<scalar const>> init,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
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
                                    rmm::mr::device_memory_resource* mr)
    : segmented_reduce_dispatch_functor(
        segmented_values, offsets, output_dtype, null_handling, std::nullopt, stream, mr)
  {
  }

  template <segmented_reduce_aggregation::Kind k>
  std::unique_ptr<column> operator()()
  {
    switch (k) {
      case segmented_reduce_aggregation::SUM:
        return reduction::segmented_sum(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::PRODUCT:
        return reduction::segmented_product(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::MIN:
        return reduction::segmented_min(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::MAX:
        return reduction::segmented_max(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::ANY:
        return reduction::segmented_any(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      case segmented_reduce_aggregation::ALL:
        return reduction::segmented_all(
          col, offsets, output_dtype, null_handling, init, stream, mr);
      default:
        CUDF_FAIL("Unsupported aggregation type.");
        // TODO: Add support for compound_ops
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
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!init.has_value() || segmented_values.type() == init.value().get().type(),
               "column and initial value must be the same type");
  if (init.has_value() && !(agg.kind == aggregation::SUM || agg.kind == aggregation::PRODUCT ||
                            agg.kind == aggregation::MIN || agg.kind == aggregation::MAX ||
                            agg.kind == aggregation::ANY || agg.kind == aggregation::ALL)) {
    CUDF_FAIL(
      "Initial value is only supported for SUM, PRODUCT, MIN, MAX, ANY, and ALL aggregation types");
  }
  CUDF_EXPECTS(offsets.size() > 0, "`offsets` should have at least 1 element.");

  return aggregation_dispatcher(
    agg.kind,
    segmented_reduce_dispatch_functor{
      segmented_values, offsets, output_dtype, null_handling, init, stream, mr});
}
}  // namespace detail

std::unique_ptr<column> segmented_reduce(column_view const& segmented_values,
                                         device_span<size_type const> offsets,
                                         segmented_reduce_aggregation const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_reduce(segmented_values,
                                  offsets,
                                  agg,
                                  output_dtype,
                                  null_handling,
                                  std::nullopt,
                                  cudf::get_default_stream(),
                                  mr);
}

std::unique_ptr<column> segmented_reduce(column_view const& segmented_values,
                                         device_span<size_type const> offsets,
                                         segmented_reduce_aggregation const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         std::optional<std::reference_wrapper<scalar const>> init,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_reduce(segmented_values,
                                  offsets,
                                  agg,
                                  output_dtype,
                                  null_handling,
                                  init,
                                  cudf::get_default_stream(),
                                  mr);
}

}  // namespace cudf
