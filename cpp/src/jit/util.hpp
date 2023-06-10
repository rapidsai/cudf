/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <string>

namespace cudf {
namespace jit {
/**
 * @brief Get the raw pointer to data in a (mutable_)column_view
 */
void const* get_data_ptr(column_view const& view);

/**
 * @brief Get the raw pointer to data in a scalar
 */
void const* get_data_ptr(scalar const& s);

}  // namespace jit
}  // namespace cudf
