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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace lists {
namespace detail {
/**
 * @brief Returns a numeric column containing lengths of each element.
 *
 * @param input Input lists column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New INT32 column with lengths.
 */
std::unique_ptr<column> count_elements(lists_column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto device_column = cudf::column_device_view::create(input.parent(), stream);
  auto d_column      = *device_column;
  // create output column
  auto output = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                        input.size(),
                                        copy_bitmask(input.parent()),
                                        input.null_count(),
                                        stream,
                                        mr);

  // fill in the sizes
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(input.size()),
                    output->mutable_view().begin<size_type>(),
                    list_size_functor{d_column});

  output->set_null_count(input.null_count());  // reset null count
  return output;
}

}  // namespace detail

// external APIS

std::unique_ptr<column> count_elements(lists_column_view const& input,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_elements(input, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
