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

#include <cudf/column/column.hpp>
#include <cudf/hashing.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

namespace nvtext {
/**
 * @addtogroup nvtext_jaccard
 * @{
 * @file
 */

/**
 * @brief Computes the jaccard similarity between individual rows
 * in two strings columns
 *
 * The similarity is calculated between strings in corresponding rows
 * such that `output[row] = J(input1[row],input2[row])`.
 *
 * @throw std::invalid_argument if the width < 2
 *
 * @param input1 Strings column to compare with input2
 * @param input2 Strings column to compare with input1
 * @param width The character width used for apply substrings;
 *              Default is 5 characters.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Index calculation values
 */
std::unique_ptr<cudf::column> jaccard_index(
  cudf::strings_column_view const& input1,
  cudf::strings_column_view const& input2,
  cudf::size_type width               = 5,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
