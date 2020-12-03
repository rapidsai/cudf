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

#include <cudf/column/column.hpp>
#include <cudf/detail/merge.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace dictionary {
namespace detail {

/**
 * @brief Merges two dictionary columns.
 *
 * The keys of both dictionary columns are expected to be already matched.
 * Otherwise, the result is undefined behavior.
 *
 * Caller must set the validity mask in the output column.
 *
 * @param lcol First column.
 * @param rcol Second column.
 * @param row_order Indexes for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> merge(dictionary_column_view const& lcol,
                              dictionary_column_view const& rcol,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
