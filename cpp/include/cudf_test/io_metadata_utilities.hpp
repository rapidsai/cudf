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

#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

void expect_metadata_equal(cudf::io::table_input_metadata in_meta,
                           cudf::io::table_metadata out_meta);

/**
 * @brief Ensures that the metadata of two tables matches for the root columns as well as for all
 * descendents (recursively)
 */
void expect_metadata_equal(cudf::io::table_metadata lhs_meta, cudf::io::table_metadata rhs_meta);

}  // namespace test
}  // namespace CUDF_EXPORT cudf
