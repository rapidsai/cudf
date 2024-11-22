/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <nanoarrow/nanoarrow.h>

namespace cudf {
namespace detail {

/**
 * @brief constants for buffer indexes of Arrow arrays
 *
 */
static constexpr int validity_buffer_idx         = 0;
static constexpr int fixed_width_data_buffer_idx = 1;

/**
 * @brief Map ArrowType id to cudf column type id
 *
 * @param arrow_view SchemaView to pull the logical and storage types from
 * @return Column type id
 */
data_type arrow_to_cudf_type(ArrowSchemaView const* arrow_view);

/**
 * @brief Map cudf column type id to ArrowType id
 *
 * @param id Column type id
 * @return ArrowType id
 */
ArrowType id_to_arrow_type(cudf::type_id id);

/**
 * @brief Map cudf column type id to the storage type for Arrow
 *
 * Specifically this is for handling the underlying storage type of
 * timestamps and durations.
 *
 * @param id column type id
 * @return ArrowType storage type
 */
ArrowType id_to_arrow_storage_type(cudf::type_id id);

/**
 * @brief Helper to initialize ArrowArray struct
 *
 * @param arr Pointer to ArrowArray to initialize
 * @param storage_type The type to initialize with
 * @param column view for column to get the length and null count from
 * @return nanoarrow status code, should be NANOARROW_OK if there are no errors
 */
int initialize_array(ArrowArray* arr, ArrowType storage_type, cudf::column_view column);

/**
 * @brief Helper to convert decimal values to 128-bit versions for Arrow compatibility
 *
 * The template parameter should be the underlying type of the data (e.g. int32_t for
 * 32-bit decimal and int64_t for 64-bit decimal).
 *
 * @param input column_view of the data
 * @param stream cuda stream to perform the operations on
 * @param mr memory resource to allocate the returned device_uvector with
 * @return unique_ptr to a device_buffer containing the upcasted data
 */
template <typename DeviceType>
std::unique_ptr<rmm::device_buffer> decimals_to_arrow(cudf::column_view input,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

template <typename T, class Enable = void>
struct DeviceType {
  using type = T;
};

template <typename T>
struct DeviceType<T, typename std::enable_if_t<is_fixed_point<T>(), void>> {
  using type = typename T::rep;
};

}  // namespace detail
}  // namespace cudf
