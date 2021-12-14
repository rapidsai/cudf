/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "cudf/types.hpp"
#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/quantiles.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/structs/structs_column_view.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
struct segmented_reduce_dispatch_functor {
  column_view const col;
  column_view const offsets;
  data_type output_dtype;
  null_policy null_handling;
  rmm::mr::device_memory_resource* mr;
  rmm::cuda_stream_view stream;

  segmented_reduce_dispatch_functor(column_view const& col,
                                    column_view const& offsets,
                                    data_type output_dtype,
                                    null_policy null_handling,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
    : col(col),
      offsets(offsets),
      output_dtype(output_dtype),
      null_handling(null_handling),
      mr(mr),
      stream(stream)
  {
  }

  template <aggregation::Kind k>
  std::unique_ptr<column> operator()(std::unique_ptr<aggregation> const& agg)
  {
    switch (k) {
      case aggregation::SUM:
        return reduction::segmented_sum(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      case aggregation::PRODUCT:
        return reduction::segmented_product(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      case aggregation::MIN:
        return reduction::segmented_min(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      case aggregation::MAX:
        return reduction::segmented_max(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      case aggregation::ANY:
        return reduction::segmented_any(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      case aggregation::ALL:
        return reduction::segmented_all(col, offsets, output_dtype, null_handling, stream, mr);
        break;
      default: CUDF_FAIL("Unsupported aggregation type.");
    }
  }
};

std::unique_ptr<column> segmented_reduce(
  column_view const& col,
  column_view const& offsets,
  std::unique_ptr<aggregation> const& agg,
  data_type output_dtype,
  null_policy null_handling,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // TODO: handle invalid inputs.

  return aggregation_dispatcher(
    agg->kind,
    segmented_reduce_dispatch_functor{col, offsets, output_dtype, null_handling, stream, mr},
    agg);
}
}  // namespace detail

std::unique_ptr<column> segmented_reduce(column_view const& col,
                                         column_view const& offsets,
                                         std::unique_ptr<aggregation> const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_reduce(
    col, offsets, agg, output_dtype, null_handling, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
