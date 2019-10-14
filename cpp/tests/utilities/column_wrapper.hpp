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
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <iterator>
#include <memory>

namespace cudf {
namespace test {

template <typename UnaryFunction>
auto make_counting_transform_iterator(cudf::size_type start, UnaryFunction f) {
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start),
                                         f);
}

class column_wrapper {
 public:
  operator column_view() const { return col->view(); }

  operator mutable_column_view() { return col->mutable_view(); }

 protected:
  std::unique_ptr<cudf::column> col{};
};

template <typename T>
class fixed_width_column_wrapper : public column_wrapper {
  static_assert(cudf::is_fixed_width<T>(), "Unexpected non-fixed width type.");

 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a new fixed width column wrapper object
   *
   * @tparam InputIterator
   * @tparam ValidInitializer
   * @param begin
   * @param end
   * @param v
   *---------------------------------------------------------------------------**/
  template <typename InputIterator, typename ValidInitializer>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end,
                             ValidInitializer v)
      : column_wrapper{} {
    std::vector<T> elements(begin, end);
    rmm::device_buffer d_elements{elements.data(), elements.size() * sizeof(T)};

    std::vector<uint8_t> null_mask(
        cudf::bitmask_allocation_size_bytes(elements.size()), 0);
    for (auto i = 0; i < elements.size(); ++i) {
      if (v[i] == true) {
        set_bit_unsafe(reinterpret_cast<cudf::bitmask_type*>(null_mask.data()),
                       i);
      }
    }

    rmm::device_buffer d_null_mask{null_mask.data(),
                                   null_mask.size() * sizeof(uint8_t)};

    col.reset(new cudf::column{
        cudf::data_type{cudf::experimental::type_to_id<T>()},
        static_cast<cudf::size_type>(elements.size()), std::move(d_elements),
        std::move(d_null_mask), cudf::UNKNOWN_NULL_COUNT});
  }

  /**---------------------------------------------------------------------------*
   * @brief
   *
   * @tparam InputIterator
   * @param begin
   * @param end
   *---------------------------------------------------------------------------**/
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end)
      : column_wrapper{} {
    std::vector<T> elements(begin, end);
    rmm::device_buffer d_elements{elements.data(), elements.size() * sizeof(T)};
    col.reset(new cudf::column{
        cudf::data_type{cudf::experimental::type_to_id<T>()},
        static_cast<cudf::size_type>(elements.size()), std::move(d_elements)});
  }

  /**---------------------------------------------------------------------------*
   * @brief
   *
   * @param element_list
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper(std::initializer_list<T> elements)
      : fixed_width_column_wrapper{std::cbegin(elements), std::cend(elements)} {
  }

  /**---------------------------------------------------------------------------*
   * @brief
   *
   * @param elements
   * @param validity
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper(std::initializer_list<T> elements,
                             std::initializer_list<bool> validity)
      : fixed_width_column_wrapper{std::cbegin(elements), std::cend(elements),
                                   std::cbegin(validity)} {}

  /**---------------------------------------------------------------------------*
   * @brief
   *
   * @param element_list
   *---------------------------------------------------------------------------**/
  template <typename ValidInitializer>
  fixed_width_column_wrapper(std::initializer_list<T> element_list,
                             ValidInitializer v)
      : fixed_width_column_wrapper{std::cbegin(element_list),
                                   std::cend(element_list), v} {}
};

class strings_column_wrapper : public column_wrapper {};

}  // namespace test
}  // namespace cudf