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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <memory>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Function object for exploding a column.
 */
struct explode_functor {
  template <typename T>
  std::unique_ptr<table> operator()(table_view const& input_table,
                                    size_type explode_column_idx,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr) const
  {
    CUDF_FAIL("Unsupported non-list column");

    return std::make_unique<table>();
  }
};

template <>
std::unique_ptr<table> explode_functor::operator()<list_view>(
  table_view const& input_table,
  size_type explode_column_idx,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  /* we explode by building a gather map that includes the number of entries in each list inside
   the column for each index. Interestingly, this can be done with lower_bound across the offsets
   as values between the offsets will all map down to the index below. We have some off-by-one
   manipulations we need to do with the output, but it's almost our gather map by itself. Once we
   build the gather map we need to remove the explode column from the table and run gather on it.
   Next we build the explode column, which turns out is simply lifting the child column out of the
   explode column. This unrolls the top level of lists. Then we need to insert the explode column
   back into the table and return it. */
  lists_column_view lc{input_table.column(explode_column_idx)};
  auto sliced_child = lc.get_sliced_child(stream);
  rmm::device_uvector<size_type> gather_map_indices(sliced_child.size(), stream, mr);

  // sliced columns can make this a little tricky. We have to start iterating at the start of the
  // offsets for this column, which could be > 0. Then we also have to handle rebasing the offsets
  // as we go.
  auto offsets           = lc.offsets().begin<size_type>() + lc.offset();
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets, [offsets] __device__(auto i) { return (i - offsets[0]) - 1; });
  auto counting_iter = thrust::make_counting_iterator(0);

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. It was pointed out that
  // this can be accomplished by simply skipping the first entry and using the result of
  // `lower_bound` directly.
  thrust::lower_bound(rmm::exec_policy(stream),
                      offsets_minus_one + 1,
                      offsets_minus_one + lc.size() + 1,
                      counting_iter,
                      counting_iter + gather_map_indices.size(),
                      gather_map_indices.begin());

  auto select_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [explode_column_idx](size_type i) { return i >= explode_column_idx ? i + 1 : i; });
  std::vector<size_type> selected_columns(select_iter, select_iter + input_table.num_columns() - 1);

  auto gathered_table = cudf::detail::gather(
    input_table.select(selected_columns),
    column_view(data_type(type_to_id<size_type>()), sliced_child.size(), gather_map_indices.data()),
    cudf::out_of_bounds_policy::DONT_CHECK,
    cudf::detail::negative_index_policy::ALLOWED,
    stream,
    mr);

  std::vector<std::unique_ptr<column>> columns = gathered_table.release()->release();

  columns.insert(columns.begin() + explode_column_idx,
                 std::make_unique<column>(column(sliced_child, stream, mr)));

  return std::make_unique<table>(std::move(columns));
}
}  // namespace

/**
 * @copydoc
 * cudf::explode(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> explode(table_view const& input_table,
                               size_type explode_column_idx,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(input_table.column(explode_column_idx).type(),
                         explode_functor{},
                         input_table,
                         explode_column_idx,
                         stream,
                         mr);
}

}  // namespace detail

/**
 * @copydoc cudf::explode(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode(table_view const& input_table,
                               size_type explode_column_idx,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::explode(input_table, explode_column_idx, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
