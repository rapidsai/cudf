/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/shuffle.h>

namespace cudf {
namespace detail {

std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement,
                              int64_t const seed,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(n >= 0, "expected number of samples should be non-negative");
  auto const num_rows = input.num_rows();

  if ((n > num_rows) and (replacement == sample_with_replacement::FALSE)) {
    CUDF_FAIL("If n > number of rows, then multiple sampling of the same row should be allowed");
  }

  if (n == 0) return cudf::empty_like(input);

  if (replacement == sample_with_replacement::TRUE) {
    auto RandomGen = [seed, num_rows] __device__(auto i) {
      thrust::default_random_engine rng(seed);
      thrust::uniform_int_distribution<size_type> dist{0, num_rows - 1};
      rng.discard(i);
      return dist(rng);
    };

    auto begin = cudf::detail::make_counting_transform_iterator(0, RandomGen);

    return detail::gather(input, begin, begin + n, out_of_bounds_policy::DONT_CHECK, stream, mr);
  } else {
    auto gather_map =
      make_numeric_column(data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED, stream);
    auto gather_map_mutable_view = gather_map->mutable_view();
    // Shuffle all the row indices
    thrust::shuffle_copy(rmm::exec_policy(stream),
                         thrust::counting_iterator<size_type>(0),
                         thrust::counting_iterator<size_type>(num_rows),
                         gather_map_mutable_view.begin<size_type>(),
                         thrust::default_random_engine(seed));

    auto gather_map_view =
      (n == num_rows) ? gather_map->view() : cudf::slice(gather_map->view(), {0, n})[0];
    return detail::gather(input,
                          gather_map_view.begin<size_type>(),
                          gather_map_view.end<size_type>(),
                          out_of_bounds_policy::DONT_CHECK,
                          stream,
                          mr);
  }
}

}  // namespace detail

std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement,
                              int64_t const seed,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  return detail::sample(input, n, replacement, seed, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
