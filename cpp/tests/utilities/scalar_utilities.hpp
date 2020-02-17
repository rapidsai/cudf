/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace test {

/**
 * @brief Verifies the equality of two scalars.
 *
 * Treats invalid scalars as equivalent.
 *
 * @param lhs                   The first scalar
 * @param rhs                   The second scalar
 *---------------------------------------------------------------------------**/
void expect_scalars_equal(cudf::scalar const& lhs,
                          cudf::scalar const& rhs);

} // namespace test
} // namespace cudf
