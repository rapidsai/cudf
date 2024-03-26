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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

namespace cudf {
namespace {

struct columns_equal_fn {
  template <typename T>
  bool operator()(column_view const& lhs, column_view const& rhs)
  {
    return lhs.type() == rhs.type();
  }
};

template <>
bool columns_equal_fn::operator()<dictionary32>(column_view const& lhs, column_view const& rhs)
{
  if (not cudf::is_dictionary(rhs.type())) { return false; }
  auto const kidx = dictionary_column_view::keys_column_index;
  return lhs.num_children() > 0 and rhs.num_children() > 0
           ? lhs.child(kidx).type() == rhs.child(kidx).type()
           : lhs.is_empty() and rhs.is_empty();
}

template <>
bool columns_equal_fn::operator()<list_view>(column_view const& lhs, column_view const& rhs)
{
  if (rhs.type().id() != type_id::LIST) { return false; }
  auto const& ci = lists_column_view::child_column_index;
  return column_types_equal(lhs.child(ci), rhs.child(ci));
}

template <>
bool columns_equal_fn::operator()<struct_view>(column_view const& lhs, column_view const& rhs)
{
  if (rhs.type().id() != type_id::STRUCT) { return false; }
  return lhs.num_children() == rhs.num_children() and
         std::all_of(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(lhs.num_children()),
                     [&](auto i) { return column_types_equal(lhs.child(i), rhs.child(i)); });
}

struct column_scalar_equal_fn {
  template <typename T>
  bool operator()(column_view const& col, scalar const& slr)
  {
    return col.type() == slr.type();
  }
};

template <>
bool column_scalar_equal_fn::operator()<dictionary32>(column_view const& col, scalar const& slr)
{
  // It is not possible to have a scalar dictionary, so compare the dictionary
  // column keys type to the scalar type.
  auto col_keys = cudf::dictionary_column_view(col).keys();
  return column_scalar_types_equal(col_keys, slr);
}

template <>
bool column_scalar_equal_fn::operator()<list_view>(column_view const& col, scalar const& slr)
{
  if (slr.type().id() != type_id::LIST) { return false; }
  auto const& ci      = lists_column_view::child_column_index;
  auto const list_slr = static_cast<list_scalar const*>(&slr);
  return column_types_equal(col.child(ci), list_slr->view());
}

template <>
bool column_scalar_equal_fn::operator()<struct_view>(column_view const& col, scalar const& slr)
{
  if (slr.type().id() != type_id::STRUCT) { return false; }
  auto const struct_slr = static_cast<struct_scalar const*>(&slr);
  auto const slr_tbl    = struct_slr->view();
  return col.num_children() == slr_tbl.num_columns() and
         std::all_of(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(col.num_children()),
                     [&](auto i) { return column_types_equal(col.child(i), slr_tbl.column(i)); });
}

};  // namespace

// Implementation note: avoid using double dispatch for this function
// as it increases code paths to NxN for N types.
bool column_types_equal(column_view const& lhs, column_view const& rhs)
{
  return type_dispatcher(lhs.type(), columns_equal_fn{}, lhs, rhs);
}

bool column_scalar_types_equal(column_view const& col, scalar const& slr)
{
  return type_dispatcher(col.type(), column_scalar_equal_fn{}, col, slr);
}

bool column_types_equivalent(column_view const& lhs, column_view const& rhs)
{
  // Check if the columns have fixed point types. This is the only case where
  // type equality and equivalence differ.
  if (lhs.type().id() == type_id::DECIMAL32 || lhs.type().id() == type_id::DECIMAL64 ||
      lhs.type().id() == type_id::DECIMAL128) {
    return lhs.type().id() == rhs.type().id();
  }
  return column_types_equal(lhs, rhs);
}

}  // namespace cudf
