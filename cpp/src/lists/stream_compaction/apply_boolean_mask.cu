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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(boolean_mask.child().type().id() == type_id::BOOL8, "Mask must be of type BOOL8.");
  CUDF_EXPECTS(input.size() == boolean_mask.size(),
               "Boolean masks column must have same number of rows as input.");
  auto const num_rows = input.size();

  if (num_rows == 0) { return cudf::empty_like(input.parent()); }

  auto constexpr offset_data_type = data_type{type_id::INT32};

  auto const boolean_mask_sliced_child = boolean_mask.get_sliced_child(stream);

  auto const make_filtered_child = [&] {
    auto filtered =
      cudf::detail::apply_boolean_mask(
        cudf::table_view{{input.get_sliced_child(stream)}}, boolean_mask_sliced_child, stream, mr)
        ->release();
    return std::move(filtered.front());
  };

  auto const make_output_offsets = [&] {
    auto boolean_mask_sliced_offsets =
      cudf::detail::slice(
        boolean_mask.offsets(), {boolean_mask.offset(), boolean_mask.size() + 1}, stream)
        .front();
    auto const sizes       = cudf::reduction::segmented_sum(boolean_mask_sliced_child,
                                                      boolean_mask_sliced_offsets,
                                                      offset_data_type,
                                                      null_policy::EXCLUDE,
                                                      std::nullopt,
                                                      stream);
    auto const d_sizes     = column_device_view::create(*sizes, stream);
    auto const sizes_begin = cudf::detail::make_null_replacement_iterator(*d_sizes, offset_type{0});
    auto const sizes_end   = sizes_begin + sizes->size();
    auto output_offsets    = cudf::make_numeric_column(
      offset_data_type, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    auto output_offsets_view = output_offsets->mutable_view();

    // Could have attempted an exclusive_scan(), but it would not compute the last entry.
    // Instead, inclusive_scan(), followed by writing `0` to the head of the offsets column.
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           sizes_begin,
                           sizes_end,
                           output_offsets_view.begin<offset_type>() + 1);
    CUDF_CUDA_TRY(cudaMemsetAsync(
      output_offsets_view.begin<offset_type>(), 0, sizeof(offset_type), stream.value()));
    return output_offsets;
  };

  return cudf::make_lists_column(input.size(),
                                 make_output_offsets(),
                                 make_filtered_child(),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}
}  // namespace detail

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_boolean_mask(input, boolean_mask, cudf::get_default_stream(), mr);
}

}  // namespace cudf::lists
