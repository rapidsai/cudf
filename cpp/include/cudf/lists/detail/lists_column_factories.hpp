/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief Internal API to construct a lists column from a `list_scalar`, for public
 * use, use `cudf::make_column_from_scalar`.
 *
 * @param[in] value The `list_scalar` to construct from
 * @param[in] size The number of rows for the output column.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<cudf::column> make_lists_column_from_scalar(list_scalar const& value,
                                                            size_type size,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty lists column.
 *
 * A list column requires a child type and so cannot be created with `make_empty_column`.
 *
 * @param child_type The type used for the empty child column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_empty_lists_column(data_type child_type,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @brief Create a lists column with all null rows.
 *
 * @param size Size of the output lists column
 * @param child_type The type used for the empty child column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_all_nulls_lists_column(size_type size,
                                                    data_type child_type,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace lists
}  // namespace cudf
