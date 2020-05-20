/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <initializer_list>
#include <tests/utilities/column_wrapper.hpp>

namespace cudf {
namespace test {

/**
 * @brief Initialize as a nested list column composed of other list composed of other list columns.
 *
 * This function handles a special case.  For convenience of declaration, we want to treat these two
 * cases as equivalent
 *
 * List<int>      = { 0, 1 }
 * List<int>      = { {0, 1} }
 *
 * while at the same time, allowing further nesting
 * List<List<int> = { {{0, 1}} }
 *
 * @param c Input column to be wrapped
 *
 */
void lists_column_wrapper::build_from_nested(std::initializer_list<lists_column_wrapper> elements,
                                             std::vector<bool> const& v)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [&v](auto i) { return v.size() <= 0 ? true : v[i]; });

  // preprocess the incoming lists. unwrap any "root" lists and just use their
  // underlying non-list data.
  // also, sanity check everything to make sure the types of all the columns are the same
  std::vector<column_view> cols;
  type_id child_id = EMPTY;
  std::transform(
    elements.begin(), elements.end(), std::back_inserter(cols), [&](lists_column_wrapper const& l) {
      // potentially unwrap
      cudf::column_view col = l.root ? lists_column_view(*l.wrapped).child() : *l.wrapped;

      // verify all children are of the same type (C++ allows you to use initializer
      // lists that could construct an invalid list column type)
      if (child_id == EMPTY) {
        child_id = col.type().id();
      } else {
        CUDF_EXPECTS(child_id == col.type().id(), "Mismatched list types");
      }

      return col;
    });

  // generate offsets column and do some type checking to make sure the user hasn't passed an
  // invalid initializer list
  size_type count = 0;
  std::vector<size_type> offsetv;
  std::transform(cols.begin(),
                 cols.end(),
                 valids,
                 std::back_inserter(offsetv),
                 [&](cudf::column_view const& col, bool valid) {
                   // nulls are represented as a repeated offset
                   size_type ret = count;
                   if (valid) { count += col.size(); }
                   return ret;
                 });
  // add the final offset
  offsetv.push_back(count);
  auto offsets =
    cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

  // concatenate them together, skipping data for children that are null
  std::vector<column_view> children;
  for (int idx = 0; idx < cols.size(); idx++) {
    if (valids[idx]) { children.push_back(cols[idx]); }
  }
  auto data = concatenate(children);

  // construct the list column
  wrapped = make_lists_column(
    cols.size(),
    std::move(offsets),
    std::move(data),
    v.size() <= 0 ? 0 : cudf::UNKNOWN_NULL_COUNT,
    v.size() <= 0 ? rmm::device_buffer{0} : detail::make_null_mask(v.begin(), v.end()));
}

/**
 * @brief Initialize as a "root" list column from a non-list input column.  Root columns
 * will be "unwrapped" when used in the nesting (list of lists) case.
 *
 * @param c Input column to be wrapped
 *
 */
void lists_column_wrapper::build_from_non_nested(std::unique_ptr<column> c)
{
  CUDF_EXPECTS(!cudf::is_nested(c->type()), "Unexpected nested type");

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  // since the incoming column is just a non-nested column, we'll turn it into a single list
  std::vector<size_type> offsetv{0, c->size()};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

  // construct the list column. mark this as a root
  root    = true;
  wrapped = make_lists_column(1, std::move(offsets), std::move(c), 0, rmm::device_buffer{0});
}

}  // namespace test
}  // namespace cudf
