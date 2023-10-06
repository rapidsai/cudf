/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/detail/normalizing_iterator.cuh>

namespace cudf {
namespace detail {

/**
 * @brief The offsets normalizing input iterator
 *
 * This is an iterator that can be used for offsets where the underlying
 * type may be int32_t or int64_t.
 *
 * Use the offsetalator_factory to create an appropriate input iterator
 * from an offsets column_view.
 */
using input_offsetalator = input_normalator<int64_t>;

/**
 * @brief The offsets normalizing output iterator
 *
 * This is an iterator that can be used for storing offsets values
 * where the underlying type may be either int32_t or int64_t.
 *
 * Use the offsetalator_factory to create an appropriate output iterator
 * from a mutable_column_view.
 *
 */
using output_offsetalator = output_normalator<int64_t>;

}  // namespace detail
}  // namespace cudf
