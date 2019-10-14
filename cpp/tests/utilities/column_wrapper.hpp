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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <tests/utilities/cudf_gtest.hpp>

namespace cudf {
namespace test {

class column_wrapper {
  operator column_view() const { return col.view(); }

  operator mutable_column_view() { return col.mutable_view(); }

 protected:
  cudf::column col;
};

template <typename T>
class fixed_width_column_wrapper : public column_wrapper {
  static_assert(cudf::is_fixed_width<T>(), "Unexpected non-fixed width type.");
};

class strings_column_wrapper : public column_wrapper {};

}  // namespace test
}  // namespace cudf