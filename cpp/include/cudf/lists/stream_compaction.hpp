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
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::lists {

/**
 * @brief Filters elements in each row of `input` LIST column using `boolean_mask`
 * LIST of booleans as a mask.
 *
 * Given an input `LIST` column and a list-of-bools column, the function produces
 * a new `LIST` column of the same type as `input`, where each element is copied
 * from the input row *only* if the corresponding `boolean_mask` is non-null and `true`.
 *
 * E.g.
 * @code{.pseudo}
 * auto const input    = lcw<int32_t>{ {0,1,2}, {3,4}, {5,6,7}, {8,9} };
 * auto const selector = lcw<bool>   { {0,1,1}, {1,0}, {1,1,1}, {0,0} };
 * auto const results  = apply_boolean_mask(lists_column_view{input}, lists_column_view{selector});
 * results             == { {1,2}, {3}, {5,6,7}, {} };
 * @endcode
 *
 * `input` and `boolean_mask` must have the same number of rows.
 * The output column has the same number of rows as the input column.
 * An element is copied to an output row *only* if the corresponding bool selector is `true`.
 * An output row is invalid only if the input row is invalid.
 *
 * @param input The input list column view to be filtered
 * @param boolean_mask A nullable list of bools column used to filter `input` elements
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return List column of the same type as `input`, containing filtered list rows
 */
std::unique_ptr<column> apply_boolean_mask(
  lists_column_view const& input,
  lists_column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cudf::lists
