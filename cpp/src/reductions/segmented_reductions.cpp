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

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/quantiles.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
struct segmented_reduce_dispatch_functor {
  column_view const col;
  device_span<size_type const> offsets;
  data_type output_dtype;
  null_policy null_handling;
  rmm::mr::device_memory_resource* mr;
  rmm::cuda_stream_view stream;

  segmented_reduce_dispatch_functor(column_view const& col,
                                    device_span<size_type const> offsets,
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
      // TODO: Support compound_ops for segmented reductions
      default: CUDF_FAIL("Unsupported aggregation type.");
    }
  }
};

std::unique_ptr<column> segmented_reduce(lists_column_view const& col,
                                         std::unique_ptr<aggregation> const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  auto const child   = col.child();
  auto const offsets = col.offsets();
  auto const offsets_device_span =
    device_span<size_type const>(offsets.data<size_type const>(), offsets.size());
  auto result =
    aggregation_dispatcher(agg->kind,
                           segmented_reduce_dispatch_functor{
                             child, offsets_device_span, output_dtype, null_handling, stream, mr},
                           agg);

  if (col.has_nulls()) {
    // Compute the bitmask-and of the reduced result with lists column's parent null mask.
    auto size         = result->size();
    auto result_mview = result->mutable_view();
    std::vector<bitmask_type const*> mask{result_mview.null_mask(), col.null_mask()};
    std::vector<size_type> begin_bits{0, 0};
    size_type valid_count = cudf::detail::inplace_bitmask_and(
      device_span<bitmask_type>(result_mview.null_mask(), num_bitmask_words(size)),
      mask,
      begin_bits,
      size,
      stream,
      mr);
    result_mview.set_null_count(size - valid_count);
  }

  return result;
}
}  // namespace detail

std::unique_ptr<column> segmented_reduce(lists_column_view const& col,
                                         std::unique_ptr<aggregation> const& agg,
                                         data_type output_dtype,
                                         null_policy null_handling,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_reduce(
    col, agg, output_dtype, null_handling, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
