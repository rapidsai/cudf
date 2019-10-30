/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#if 0  // TODO: enable once the scatter/gather PR is merged
#include <cudf/detail/gather.cuh>
#endif
#include <cudf/detail/repeat.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/mr/device_memory_resource.hpp>

// for gdf_scalar, unnecessary once we switch to cudf::scalar
#include <cudf/types.h>
// for gdf_dtype_of, unnecessary once we switch to cudf::scalar
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

#include <cuda_runtime.h>

#include <memory>

namespace cudf {
namespace experimental {

namespace detail {

std::unique_ptr<table> repeat(table_view const& input_table,
                   column_view const& count, bool check_count,
                   cudaStream_t stream,
                   rmm::mr::device_memory_resource* mr) {
  // TODO: can't this be of any integral type?
  CUDF_EXPECTS(count.type().id() == type_to_id<size_type>(),
               "count column should be of index type");
  CUDF_EXPECTS(input_table.num_rows() == count.size(),
               "in and count must have equal size");
  CUDF_EXPECTS(count.has_nulls(), "count cannot contain nulls");

  if (input_table.num_rows() == 0) {
    return cudf::experimental::empty_like(input_table);
  }
  
  rmm::device_vector<size_type> offsets(input_table.num_rows());
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                         count.begin<size_t>(), count.end<size_t>(), offsets.begin());
  if (check_count == true) {
    CUDF_EXPECTS(thrust::is_sorted(rmm::exec_policy(stream)->on(stream),
                                   offsets.begin(), offsets.end()) == true,
                 "count has negative values or the resulting table has more \
                  rows than size_type's limit.");
  }

  auto output_size = size_type{offsets.back()};
  auto p_indices = make_numeric_column(data_type{type_to_id<size_type>()},
                                       output_size, mask_state{UNALLOCATED},
                                       stream, mr);
  auto indices = p_indices->mutable_view();
  thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                      offsets.begin(), offsets.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(output_size),
                      indices.begin<size_type>());

#if 1  // TODO: placeholder till the scatter/gather PR is merged
  return cudf::experimental::empty_like(input_table);
#else
  return detail::gather(input_table, *p_indices, false, stream, mr);
#endif
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              gdf_scalar const& count,
                              cudaStream_t stream,
                              rmm::mr::device_memory_resource* mr) {
  // TODO: can't this be of any integral type?
  CUDF_EXPECTS(count.dtype == gdf_dtype_of<gdf_size_type>(),
               "count value should be of index type");
  CUDF_EXPECTS(count.is_valid, "count cannot be null");
  auto stride = size_type{count.data.si32};
  CUDF_EXPECTS(stride >= 0, "count value should be non-negative");
  CUDF_EXPECTS(static_cast<int64_t>(input_table.num_rows()) * stride <=
                 std::numeric_limits<size_type>::max(),
               "The resulting table has more rows than size_type's limit.");

  if ((input_table.num_rows() == 0) || (stride == 0)) {
    return cudf::experimental::empty_like(input_table);
  }

  auto output_size = input_table.num_rows() * stride;
  // TODO: no need to create this intermediate buffer if the gather function
  // directly takes thrust iterators.
  auto p_indices = make_numeric_column(data_type{type_to_id<size_type>()},
                                       output_size, mask_state{UNALLOCATED},
                                       stream, mr);
  auto indices = p_indices->mutable_view();
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(output_size),
                    indices.begin<size_type>(),
                    [stride] __device__ (auto i) { return i / stride; });

#if 1  // TODO: placeholder till the scatter/gather PR is merged
  return cudf::experimental::empty_like(input_table);
#else
  return detail::gather(input_table, indices, false, stream, mr);
#endif
}

}  // namespace detail

std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              bool check_count,
                              rmm::mr::device_memory_resource* mr) {
  return detail::repeat(input_table, count, check_count, 0, mr);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              gdf_scalar const& count,
                              rmm::mr::device_memory_resource* mr) {
  return detail::repeat(input_table, count, 0, mr);
}

}  // namespace experimental
}  // namespace cudf
