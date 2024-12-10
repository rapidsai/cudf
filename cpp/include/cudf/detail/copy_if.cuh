/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

/**
 * @brief Filters `input` using a Filter function object
 *
 * @p filter must be a functor or lambda with the following signature:
 * __device__ bool operator()(cudf::size_type i);
 * It will return true if element i of @p input should be copied,
 * false otherwise.
 *
 * @tparam Filter the filter functor type
 * @param input The table_view to filter
 * @param filter A function object that takes an index and returns a bool
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used for allocating the returned memory
 * @return The table generated from filtered `input`
 */
template <typename Filter>
std::unique_ptr<table> copy_if(table_view const& input,
                               Filter filter,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (0 == input.num_rows() || 0 == input.num_columns()) { return empty_like(input); }

  auto indices     = rmm::device_uvector<size_type>(input.num_rows(), stream);
  auto const begin = thrust::counting_iterator<size_type>(0);
  auto const end   = begin + input.num_rows();
  auto const indices_end =
    thrust::copy_if(rmm::exec_policy(stream), begin, end, indices.begin(), filter);

  auto const output_size = static_cast<size_type>(thrust::distance(indices.begin(), indices_end));

  // nothing selected
  if (output_size == 0) { return empty_like(input); }
  // everything selected
  if (output_size == input.num_rows()) { return std::make_unique<table>(input, stream, mr); }

  auto const map = device_span<size_type const>(indices.data(), output_size);
  return cudf::detail::gather(
    input, map, out_of_bounds_policy::DONT_CHECK, negative_index_policy::NOT_ALLOWED, stream, mr);
}

}  // namespace detail
}  // namespace cudf
