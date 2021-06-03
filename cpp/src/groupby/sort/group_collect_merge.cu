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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

// TODO: Reorganize
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/combine.hpp>

//#include <cudf_test/column_utilities.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_collect_merge(column_view const& values,
                                            cudf::device_span<size_type const> group_offsets,
                                            size_type num_groups,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  //  printf("line: %d\n", __LINE__);
  //  cudf::test::print(values);
  //  auto child = lists_column_view(values).get_sliced_child(stream);
  //  printf("line: %d\n", __LINE__);
  //  cudf::test::print(child);
  /*

  auto const keys1  = keys_col{1, 2};
  auto const keys2  = keys_col{1, 3};
  auto const keys3  = keys_col{2, 3, 4};
  auto const lists1 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists2 = lists_col{{10, 11}, {11, 12}};
  auto const lists3 = lists_col{{20, 21, 22}, {23, 24, 25}, {24, 25, 26}};

  auto const keys  = cudf::concatenate(table_views({keys1, keys2, keys3}));
  auto const lists = cudf::concatenate(table_views({lists1, lists2, lists3}));

  auto const expected_keys = keys_col{1, 2, 3, 4};
  auto const expected_lists =
    lists_col{{1, 2, 3, 10, 11}, {4, 5, 6, 20, 21, 22}, {11, 12, 23, 24, 25}, {24, 25, 26}};





line: 44
List<int8_t>:
Length : 7
Offsets : 0, 3, 5, 8, 11, 13, 16, 19
   1, 2, 3, 10, 11, 4, 5, 6, 20, 21, 22, 11, 12, 23, 24, 25, 24, 25, 26


line: 47
    1,2,3,10,11,4,5,6,20,21,22,11,12,23,24,25,24,25,26


   */

  CUDF_EXPECTS(!values.nullable(), "Input to `group_collect_merge` must not be nullable.");

  // Generate offsets just by copying from the provided offsets.
  auto offsets_column = make_numeric_column(
    data_type(type_to_id<offset_type>()), num_groups + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<offset_type>(0),
    thrust::make_counting_iterator<offset_type>(group_offsets.size()),
    offsets_column->mutable_view().template begin<offset_type>(),
    [group_offsets = group_offsets.begin(),
     list_offsets  = lists_column_view(values).offsets_begin()] __device__(auto const idx)
      -> offset_type { return list_offsets[group_offsets[idx]]; });

  // The child column of the output lists column is just copied from the input column.
  // (grandchild of input will become child of output)
  auto child_column = std::make_unique<column>(lists_column_view(values).get_sliced_child(stream));

  //  printf("line: %d\n", __LINE__);

  return make_lists_column(num_groups,
                           std::move(offsets_column),
                           std::move(child_column),
                           0,
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
