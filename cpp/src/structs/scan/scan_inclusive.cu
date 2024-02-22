/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "reductions/nested_type_minmax_util.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <vector>

namespace cudf {
namespace structs {
namespace detail {
namespace {

}  // namespace

template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  // Create a gather map containing indices of the prefix min/max elements.
  auto gather_map = rmm::device_uvector<size_type>(input.size(), stream);
  auto const binop_generator =
    cudf::reduction::detail::comparison_binop_generator::create<Op>(input, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         thrust::counting_iterator<size_type>(0),
                         thrust::counting_iterator<size_type>(input.size()),
                         gather_map.begin(),
                         binop_generator.binop());

  // Gather the children columns of the input column. Must use `get_sliced_child` to properly
  // handle input in case it is a sliced view.
  auto const input_children = [&] {
    auto const it = cudf::detail::make_counting_transform_iterator(
      0, [structs_view = structs_column_view{input}, &stream](auto const child_idx) {
        return structs_view.get_sliced_child(child_idx, stream);
      });
    return std::vector<column_view>(it, it + input.num_children());
  }();

  // Gather the children elements of the prefix min/max struct elements for the output.
  auto scanned_children = cudf::detail::gather(table_view{input_children},
                                               gather_map,
                                               cudf::out_of_bounds_policy::DONT_CHECK,
                                               cudf::detail::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr)
                            ->release();

  // Don't need to set a null mask because that will be handled at the caller.
  return make_structs_column(
    input.size(), std::move(scanned_children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
}

template std::unique_ptr<column> scan_inclusive<DeviceMin>(column_view const& input_view,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr);

template std::unique_ptr<column> scan_inclusive<DeviceMax>(column_view const& input_view,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace structs
}  // namespace cudf
