/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <numbers>

struct DLManagedTensor;
struct DLManagedTensorVersioned;

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Export column as DLManagedTensorVersioned
 *
 * This function copies the data only if `to_cpu` is true or it is flagged.
 * Note that the signature is currently designed to be called from pylibcudf.
 * (Otherwise, it should for example signal the DLPack version.)
 *
 * @param col Column view to export.
 * @param copy Whether to copy the data (must be set if `to_cpu` is true).
 * If copy is false, a `delete_func` must be provided if the capsule may outlive
 * the column view.  WARNING: This function is not called on error!
 * @param to_cpu Whether to copy to the CPU rather than making a device copy.
 * @param sync_stream If passed, an event is put on this stream to ensure
 * data is available for use on it.  If `nullopt`, the stream is synchronized.
 * @param stream The cudf stream to operate on (i.e. to copy the data safely).
 * @param mr Memory resource to use for the new allocation (if on GPU).
 * @param delete_func Custom delete function, needed to avoid Python API use.
 * On error, the delete function is not called.
 * @param delete_ctx Context for custom delete (i.e. Python owner of data).
 */
DLManagedTensorVersioned* to_dlpack_versioned(
  column_view col,
  bool copy,
  bool to_cpu,
  rmm::cuda_stream_view sync_stream,
  void (*delete_func)(void*),
  void* delete_ctx,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @copydoc cudf::to_dlpack
 *
 * @note Prefer the column-based overload for new code. This overload should be
 * deprecated eventually, but is public API. (Or updated if we need it.)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

/**
 * @brief Return a maximum precision for a given type.
 *
 * @tparam T the type to get the maximum precision for
 */
template <typename T>
constexpr std::size_t max_precision()
{
  auto constexpr num_bits = sizeof(T) * 8;
  return std::floor(num_bits * std::numbers::ln2 / std::numbers::ln10);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
