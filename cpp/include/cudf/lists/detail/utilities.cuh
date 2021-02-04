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
#pragma once

#include <cudf/column/column_view.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Fetch the number of rows in a lists column's child given its offsets column.
 *
 * @param[in] list_offsets Offsets child of a lists column
 * @param[in] stream The cuda-stream to synchronize on, when reading from device memory
 * @return cudf::size_type The number of child rows in the lists column
 */
static cudf::size_type get_num_child_rows(cudf::column_view const& list_offsets,
                                          rmm::cuda_stream_view stream)
{
  // Number of rows in child-column == last offset value.
  cudf::size_type num_child_rows{};
  CUDA_TRY(cudaMemcpyAsync(&num_child_rows,
                           list_offsets.data<cudf::size_type>() + list_offsets.size() - 1,
                           sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  stream.synchronize();
  return num_child_rows;
}

}  // namespace detail
}  // namespace cudf
