/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cudf/detail/gather.cuh>

namespace cudf {
namespace {
struct tile_functor {
  size_type count;
  size_type __device__ operator()(size_type i) { return i % count; }
};

}  // anonymous namespace

namespace detail {
std::unique_ptr<table> tile(const table_view &in,
                            size_type count,
                            cudaStream_t stream,
                            rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(count >= 0, "Count cannot be negative");

  auto in_num_rows = in.num_rows();

  if (count == 0 or in_num_rows == 0) { return empty_like(in); }

  auto out_num_rows = in_num_rows * count;
  auto counting_it  = thrust::make_counting_iterator<size_type>(0);
  auto tiled_it     = thrust::make_transform_iterator(counting_it, tile_functor{in_num_rows});

  return detail::gather(in, tiled_it, tiled_it + out_num_rows, false, mr, stream);
}
}  // namespace detail

std::unique_ptr<table> tile(const table_view &in,
                            size_type count,
                            rmm::mr::device_memory_resource *mr)
{
  CUDF_FUNC_RANGE();
  return detail::tile(in, count, 0, mr);
}

}  // namespace cudf
