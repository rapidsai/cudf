/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>

namespace cudf::detail {

/**
 * @copydoc cudf::purge_nonempty_nulls(structs_column_view const&, rmm::mr::device_memory_resource*)
 *
 * @tparam ColumnViewT View type (lists_column_view, strings_column_view, or strings_column_view)
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename ColumnViewT>
std::unique_ptr<cudf::column> purge_nonempty_nulls(ColumnViewT const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  // Implement via identity gather.
  auto const input_column = input.parent();
  auto const gather_begin = thrust::counting_iterator<cudf::size_type>(0);
  auto const gather_end   = gather_begin + input_column.size();

  auto gathered_table = cudf::detail::gather(table_view{{input_column}},
                                             gather_begin,
                                             gather_end,
                                             out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr);
  return std::move(gathered_table->release()[0]);
}

}  // namespace cudf::detail
