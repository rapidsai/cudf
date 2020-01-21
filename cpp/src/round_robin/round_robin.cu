/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <type_traits>
#include <cmath> // for std::ceil()

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/utilities/bit.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/gather.cuh>

namespace cudf {
namespace experimental { 
namespace detail {

std::pair<std::unique_ptr<table>,
          std::vector<cudf::size_type>>
round_robin_partition(table_view const& input,
                      cudf::size_type num_partitions,
                      cudf::size_type start_partition = 0,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                      cudaStream_t stream = 0)
{
  auto nrows = input.num_rows();

  CUDF_EXPECTS( num_partitions > 1 && num_partitions < nrows, "Incorrect number of partitions. Must be greater than 1 and less than number of rows." );
  CUDF_EXPECTS( start_partition < num_partitions, "Incorrect start_partition index. Must be less than number of partitions." );
  
  auto num_partitions_max_size = nrows % num_partitions;//# partitions of max size
  cudf::size_type max_partition_size = std::ceil( static_cast<double>(nrows) / static_cast<double>(num_partitions));// max size of partitions
  
  auto total_max_partitions_size = num_partitions_max_size * max_partition_size;
  auto num_partitions_min_size = num_partitions - num_partitions_max_size;

  //delta is the number of positions to rotate right
  //the original range [0,1,...,n-1]
  //and is calculated by accumulating the first
  //`start_partition` partition sizes from the end;
  //i.e.,
  //the partition sizes array (of size p) being:
  //[m,m,...,m,(m-1),...,(m-1)]
  //(with num_partitions_max_size sizes `m` at the beginning;
  //and (p-num_partitions_max_size) sizes `(m-1)` at the end)
  //we accumulate the 1st `start_partition` entries from the end:
  //
  auto delta = (start_partition > num_partitions_min_size?
                num_partitions_min_size*(max_partition_size-1) + (start_partition - num_partitions_min_size)*max_partition_size :
                start_partition*(max_partition_size-1));

  
  auto iter_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                    [nrows, num_partitions, max_partition_size, num_partitions_max_size, total_max_partitions_size, delta] __device__ (auto index0){
                                      //rotate original index right by delta positions;
                                      //this is the effect of applying start_partition:
                                      //
                                      auto rotated_index = (index0 + nrows - delta) % nrows;

                                      //using rotated_index = given index0, rotated;
                                      //the algorithm below calculates the src round-robin row,
                                      //by calculating the partition_index and the index_within_partition:
                                      //
                                      auto index_within_partition = (rotated_index <= total_max_partitions_size ? rotated_index % max_partition_size: (rotated_index - total_max_partitions_size) % (max_partition_size-1) );
                                      auto partition_index = (rotated_index <= total_max_partitions_size ? rotated_index / max_partition_size: num_partitions_max_size + (rotated_index - total_max_partitions_size) / (max_partition_size-1) );
                                      return num_partitions * index_within_partition + partition_index;
                                    });

  auto uniq_tbl = cudf::experimental::detail::gather(input,
                                                     iter_begin, iter_begin + nrows,
                                                     false, false, false,
                                                     mr,
                                                     stream);
  auto ret_pair =
    std::make_pair(std::move(uniq_tbl), std::vector<cudf::size_type>(num_partitions));

  //this has the effect of rotating the set of partition sizes
  //right by start_partition positions:
  //
  auto rotated_iter_begin =
    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                    [num_partitions, start_partition, max_partition_size, num_partitions_max_size] (auto index){
                                      return ((index + num_partitions - start_partition) % num_partitions < num_partitions_max_size? max_partition_size : max_partition_size-1);
                                    });

  //then exclusive_scan on the resulting
  //rotated partition sizes to get the partition offsets
  //corresponding to start_partition:
  //Since:
  //"num_partitions is usually going to be relatively small
  //(<1,000), as such, it's probably more expensive to do this on the device.
  //Instead, do it on the host directly into the std::vector and avoid the memcpy." - JH
  //
  thrust::exclusive_scan(thrust::host,
                         rotated_iter_begin, rotated_iter_begin + num_partitions,
                         ret_pair.second.begin());

  return ret_pair;
}
  
}  // namespace detail

std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>>
round_robin_partition(table_view const& input,
                      cudf::size_type num_partitions,
                      cudf::size_type start_partition = 0,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()) {
  
  return cudf::experimental::detail::round_robin_partition(input, num_partitions, start_partition, mr);
}
  
}  // namespace experimental
}  // namespace cudf
