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
#include <cudf/detail/gather.cuh>
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
  /**
   * @brief Function object for exploding a column.
   */
  template <typename T>
  std::unique_ptr<table> operator()(table_view const& input_table,
                                    size_type const explode_column_idx,
                                    bool include_pos,
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
  size_type const explode_column_idx,
  bool include_pos,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  lists_column_view lc{input_table.column(explode_column_idx)};
  auto sliced_child = lc.get_sliced_child(stream);
  rmm::device_uvector<size_type> gather_map_indices(sliced_child.size(), stream);

  // Sliced columns may require rebasing of the offsets.
  auto offsets = lc.offsets_begin();
  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets + 1, [offsets] __device__(auto i) { return (i - offsets[0]) - 1; });
  auto counting_iter = thrust::make_counting_iterator(0);

  rmm::device_uvector<size_type> pos(include_pos ? sliced_child.size() : 0, stream, mr);

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. This can be accomplished by
  // skipping the first entry and using the result of `lower_bound` directly.
  if (include_pos) {
    thrust::transform(
      rmm::exec_policy(stream),
      counting_iter,
      counting_iter + gather_map_indices.size(),
      gather_map_indices.begin(),
      [position_array = pos.data(), offsets_minus_one, offsets, offset_size = lc.size()] __device__(
        auto idx) -> size_type {
        auto lb_idx = thrust::lower_bound(
                        thrust::seq, offsets_minus_one, offsets_minus_one + offset_size, idx) -
                      offsets_minus_one;
        position_array[idx] = idx - (offsets[lb_idx] - offsets[0]);
        return lb_idx;
      });
  } else {
    thrust::lower_bound(rmm::exec_policy(stream),
                        offsets_minus_one,
                        offsets_minus_one + lc.size(),
                        counting_iter,
                        counting_iter + gather_map_indices.size(),
                        gather_map_indices.begin());
  }

  auto select_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [explode_column_idx](size_type i) { return i >= explode_column_idx ? i + 1 : i; });
  std::vector<size_type> selected_columns(select_iter, select_iter + input_table.num_columns() - 1);

  auto gathered_table = cudf::detail::gather(input_table.select(selected_columns),
                                             gather_map_indices.begin(),
                                             gather_map_indices.end(),
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr);

  std::vector<std::unique_ptr<column>> columns = gathered_table.release()->release();

  columns.insert(columns.begin() + explode_column_idx,
                 std::make_unique<column>(sliced_child, stream, mr));

  if (include_pos) {
    columns.insert(columns.begin() + explode_column_idx,
                   std::make_unique<column>(
                     data_type(type_to_id<size_type>()), sliced_child.size(), pos.release()));
  }

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
                               bool include_pos,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(input_table.column(explode_column_idx).type(),
                         explode_functor{},
                         input_table,
                         explode_column_idx,
                         include_pos,
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
  return detail::explode(input_table, explode_column_idx, false, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc cudf::explode_position(input_table,explode_column_idx,rmm::mr::device_memory_resource)
 */
std::unique_ptr<table> explode_position(table_view const& input_table,
                                        size_type explode_column_idx,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::explode(input_table, explode_column_idx, true, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
