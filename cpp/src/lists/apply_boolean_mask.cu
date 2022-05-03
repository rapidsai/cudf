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
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/reduce.h>

namespace cudf::lists {
namespace detail {
namespace {

class get_list_size {
 public:
  explicit get_list_size(lists_column_view const& lcv)
    : num_rows{lcv.size()},
      offsets{lcv.offsets().begin<offset_type>() + lcv.offset()},
      bitmask{lcv.null_mask()}
  {
  }

  size_type __device__ operator()(size_type i) const
  {
    return bit_value_or(bitmask, i, true) ? (offsets[i + 1] - offsets[i]) : 0;
  }

 private:
  size_type num_rows;
  offset_type const* offsets;
  bitmask_type const* bitmask;
};

void assert_same_list_sizes(lists_column_view const& input,
                            lists_column_view const& boolean_mask,
                            rmm::cuda_stream_view stream)
{
  auto const begin = cudf::detail::make_counting_transform_iterator(
    0,
    [get_list_size = get_list_size{input}, get_mask_size = get_list_size{boolean_mask}] __device__(
      size_type i) -> size_type { return get_list_size(i) != get_mask_size(i); });

  CUDF_EXPECTS(thrust::reduce(rmm::exec_policy(stream), begin, begin + input.size()) == 0,
               "Each list row must match the corresponding boolean mask row in size.");
}
}  // namespace

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() == boolean_mask.size(),
               "Boolean masks column must have same number of rows as input.");

  auto const num_rows = input.size();

  if (num_rows == 0) { return cudf::empty_like(input.parent()); }
  // Note: This assert guarantees that no elements are gathered
  // from nominally NULL input list rows.
  assert_same_list_sizes(input, boolean_mask, stream);

  auto constexpr offset_data_type = data_type{type_id::INT32};

  auto filtered_child = [&] {
    std::unique_ptr<cudf::table> tbl =
      cudf::detail::apply_boolean_mask(cudf::table_view{{input.get_sliced_child(stream)}},
                                       boolean_mask.get_sliced_child(stream),
                                       stream,
                                       mr);
    std::vector<std::unique_ptr<cudf::column>> columns = tbl->release();
    return std::move(columns.front());
  };

  auto output_offsets = [&] {
    auto boolean_mask_sliced_offsets =
      cudf::detail::slice(
        boolean_mask.offsets(), {boolean_mask.offset(), boolean_mask.size() + 1}, stream)
        .front();

    auto const sizes         = cudf::reduction::segmented_sum(boolean_mask.get_sliced_child(stream),
                                                      boolean_mask_sliced_offsets,
                                                      offset_data_type,
                                                      null_policy::EXCLUDE,
                                                      stream);
    auto const scalar_0      = cudf::numeric_scalar<offset_type>{0, true, stream};
    auto const no_null_sizes = cudf::detail::replace_nulls(*sizes, scalar_0, stream);

    auto offsets = cudf::make_numeric_column(
      offset_data_type, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           no_null_sizes->view().begin<offset_type>(),
                           no_null_sizes->view().end<offset_type>(),
                           offsets->mutable_view().begin<offset_type>() + 1);
    CUDF_CUDA_TRY(cudaMemsetAsync(
      offsets->mutable_view().begin<offset_type>(), 0, sizeof(offset_type), stream.value()));
    return offsets;
  };

  return cudf::make_lists_column(input.size(),
                                 output_offsets(),
                                 filtered_child(),
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
  return detail::apply_boolean_mask(input, boolean_mask, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
