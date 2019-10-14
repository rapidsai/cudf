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
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <iterator>
#include <memory>

namespace cudf {
namespace test {

class column_wrapper {
  operator column_view() const { return col->view(); }

  operator mutable_column_view() { return col->mutable_view(); }

 protected:
  std::unique_ptr<cudf::column> col{};
};

template <typename T>
class fixed_width_column_wrapper : public column_wrapper {
  static_assert(cudf::is_fixed_width<T>(), "Unexpected non-fixed width type.");

  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end)
      : column_wrapper{} {
    std::vector<T> elements(begin, end);
    rmm::device_buffer d_elements{elements.data(), elements.size() * sizeof(T)};
    col.reset(cudf::data_type{cudf::experimental::type_to_id<T>()},
              elements.size(), std::move(d_elements));
  }

  fixed_width_column_wrapper(std::initializer_list<T> element_list)
      : fixed_width_column_wrapper{std::cbegin(element_list),
                                   std::cend(element_list)} {}

  template <typename InputIterator, typename NullIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end,
                             NullIterator bit_initializer)
      : column_wrapper{} {}
};

class strings_column_wrapper : public column_wrapper {};

}  // namespace test
}  // namespace cudf