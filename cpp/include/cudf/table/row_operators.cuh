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

#include <cudf/column/column_device_view.cuh>
#include <cudf/sorting.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utils/traits.hpp>
#include <utilities/release_assert.cuh>

namespace cudf {

namespace exp {

enum class comparison_state { LESS, EQUAL, GREATER };

template <bool has_nulls = true>
struct element_relational_comparator {
  template <typename Element, std::enable_if_t<cudf::is_relationally_comparable<
                                  Element, Element>>* = nullptr>
  __device__ bool operator()(column_device_view lhs,
                             size_type lhs_element_index,
                             column_device_view rhs,
                             size_type rhs_element_index) {
    if (has_nulls) {
      bool const lhs_valid{not lhs.nullable() or
                           lhs.is_valid(lhs_element_index)};

      bool const rhs_valid{not rhs.nullable() or
                           rhs.is_valid(rhs_element_index)};
    }

    Element const lhs_element = lhs.data<Element>(lhs_element_index);
    Element const rhs_element = rhs.data<Element>(rhs_element_index);

    if (lhs_element < rhs_element) {
      return true;
    } else if (rhs_element < lhs_element) {
      return false;
    }
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_relationally_comparable<
                Element, Element>>* = nullptr>
  __device__ bool operator()(column_device_view lhs,
                             size_type lhs_element_index,
                             column_device_view rhs,
                             size_type rhs_element_index) {
    release_assert(false &&
                   "Attempted to compare elements of uncomparable types.");
  }
};

template <bool has_nulls = true>
class row_lexicographic_comparator {
 public:
  row_lexicographic_comparator(table_device_view lhs, table_device_view rhs,
                               null_size size_of_nulls = null_size::LOWEST,
                               order* column_order = nullptr)
      : _lhs{lhs},
        _rhs{rhs},
        _size_of_nulls{size_of_nulls},
        _column_order{column_order} {}

  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const
      noexcept {}

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  null_size _size_of_nulls{null_size::LOWEST};
  order const* _column_order{};
};

}  // namespace exp

}  // namespace cudf
