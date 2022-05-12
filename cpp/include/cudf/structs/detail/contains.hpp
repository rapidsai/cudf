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

#include <cudf/column/column.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace structs {
namespace detail {

/**
 * @brief Returns true if the scalar is found in the column.
 *
 * @param haystack Column to search against
 * @param needle Scalar to search for
 * @param stream  CUDA stream used for device memory operations and kernel launches.
 * @return True if the `needle` is found in `haystack`
 */
bool contains(structs_column_view const& haystack,
              scalar const& needle,
              rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace structs
}  // namespace cudf
