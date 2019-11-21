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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/strings/copying.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/copying.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cudf {

// Copy constructor
column::column(column const &other)
    : _type{other._type},
      _size{other._size},
      _data{other._data},
      _null_mask{other._null_mask},
      _null_count{other._null_count} {
  _children.reserve(other.num_children());
  for (auto const &c : other._children) {
    _children.emplace_back(std::make_unique<column>(*c));
  }
}

// Copy ctor w/ explicit stream/mr
column::column(column const &other, cudaStream_t stream,
               rmm::mr::device_memory_resource *mr)
    : _type{other._type},
      _size{other._size},
      _data{other._data, stream, mr},
      _null_mask{other._null_mask, stream, mr},
      _null_count{other._null_count} {
  _children.reserve(other.num_children());
  for (auto const &c : other._children) {
    _children.emplace_back(std::make_unique<column>(*c, stream, mr));
  }
}

// Move constructor
column::column(column &&other) noexcept
    : _type{other._type},
      _size{other._size},
      _data{std::move(other._data)},
      _null_mask{std::move(other._null_mask)},
      _null_count{other._null_count},
      _children{std::move(other._children)} {
  other._size = 0;
  other._null_count = 0;
  other._type = data_type{EMPTY};
}

// Release contents
column::contents column::release() noexcept {
  _size = 0;
  _null_count = 0;
  _type = data_type{EMPTY};
  return column::contents{
      std::make_unique<rmm::device_buffer>(std::move(_data)),
      std::make_unique<rmm::device_buffer>(std::move(_null_mask)),
      std::move(_children)};
}

// Create immutable view
column_view column::view() const {
  // Create views of children
  std::vector<column_view> child_views;
  child_views.reserve(_children.size());
  for (auto const &c : _children) {
    child_views.emplace_back(*c);
  }

  return column_view{
      type(),       size(),
      _data.data(), static_cast<bitmask_type const *>(_null_mask.data()),
      null_count(), 0,
      child_views};
}

// Create mutable view
mutable_column_view column::mutable_view() {
  // create views of children
  std::vector<mutable_column_view> child_views;
  child_views.reserve(_children.size());
  for (auto const &c : _children) {
    child_views.emplace_back(*c);
  }

  // Store the old null count
  auto current_null_count = null_count();

  // The elements of a column could be changed through a `mutable_column_view`,
  // therefore the existing `null_count` is no longer valid. Reset it to
  // `UNKNOWN_NULL_COUNT` forcing it to be recomputed on the next invocation of
  // `null_count()`.
  set_null_count(cudf::UNKNOWN_NULL_COUNT);

  return mutable_column_view{type(),
                             size(),
                             _data.data(),
                             static_cast<bitmask_type *>(_null_mask.data()),
                             current_null_count,
                             0,
                             child_views};
}

// If the null count is known, return it. Else, compute and return it
size_type column::null_count() const {
  if (_null_count <= cudf::UNKNOWN_NULL_COUNT) {
    _null_count = cudf::count_unset_bits(
        static_cast<bitmask_type const *>(_null_mask.data()), 0, size());
  }
  return _null_count;
}

void column::set_null_count(size_type new_null_count) {
  if (new_null_count > 0) {
    CUDF_EXPECTS(nullable(), "Invalid null count.");
  }
  _null_count = new_null_count;
}

struct create_column_from_view {
  cudf::column_view view;
  cudaStream_t stream;
  rmm::mr::device_memory_resource *mr;

 template <typename ColumnType,
           std::enable_if_t<std::is_same<ColumnType, cudf::string_view>::value>* = nullptr>
 std::unique_ptr<column> operator()() {
   cudf::strings_column_view sview(view);
   auto col = cudf::strings::detail::slice(sview, view.offset(), -1, 1, stream, mr);
   return col;
 }

 template <typename ColumnType,
           std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
 std::unique_ptr<column> operator()() {

   std::vector<std::unique_ptr<column>> children;
   for (size_type i = 0; i < view.num_children(); ++i) {
     children.emplace_back(std::make_unique<column>(view.child(i), stream, mr));
   }

   auto col = std::make_unique<column>(view.type(), view.size(),
       rmm::device_buffer{
       static_cast<const char*>(view.head()) +
       (view.offset() * cudf::size_of(view.type())),
       view.size() * cudf::size_of(view.type()), stream, mr},
       cudf::copy_bitmask(view, stream, mr),
       view.null_count(), std::move(children));

   return col;
 }

};

struct create_column_from_view_vector {
  std::vector<cudf::column_view> views;
  cudaStream_t stream;
  rmm::mr::device_memory_resource *mr;

 template <typename ColumnType,
           std::enable_if_t<std::is_same<ColumnType, cudf::string_view>::value>* = nullptr>
 std::unique_ptr<column> operator()() {
   std::vector<cudf::strings_column_view> sviews;
   sviews.reserve(views.size());
   for (auto &v : views) { sviews.emplace_back(v); }

   auto col = cudf::strings::detail::concatenate(sviews, mr, stream);

   //If concatenated string column is nullable, proceed to calculate it
   if (col->nullable()) {
     cudf::detail::concatenate_masks(views,
         (col->mutable_view()).null_mask(), stream);
   }

   return col;
 }

 template <typename ColumnType,
           std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
 std::unique_ptr<column> operator()() {

   auto type = views.front().type();
   size_type total_element_count =
     std::accumulate(views.begin(), views.end(), 0,
         [](auto accumulator, auto const& v) { return accumulator + v.size(); });

   bool has_nulls = std::any_of(views.begin(), views.end(),
                      [](const column_view col) { return col.has_nulls(); });
   using mask_policy = cudf::experimental::mask_allocation_policy;

   mask_policy policy{mask_policy::NEVER};
   if (has_nulls) { policy = mask_policy::ALWAYS; }

   auto col = cudf::experimental::allocate_like(views.front(),
       total_element_count, policy, mr);

   auto m_view = col->mutable_view();
   auto count = 0;
   // TODO replace loop with a single kernel https://github.com/rapidsai/cudf/issues/2881
   for (auto &v : views) {
     thrust::copy(rmm::exec_policy()->on(stream),
         v.begin<ColumnType>(),
         v.end<ColumnType>(),
         m_view.begin<ColumnType>() + count);
     count += v.size();
   }

   //If concatenated column is nullable, proceed to calculate it
   if (col->nullable()) {
     cudf::detail::concatenate_masks(views,
         (col->mutable_view()).null_mask(), stream);
   }

   return col;
 }

};

// Copy from a view
column::column(column_view view, cudaStream_t stream,
               rmm::mr::device_memory_resource *mr) :
column( *cudf::experimental::type_dispatcher(view.type(), create_column_from_view{view, stream, mr})) {}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  if (columns_to_concat.empty()) { return std::make_unique<column>(); }

  data_type type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
        [type](auto const& c) { return c.type() == type; }),
      "Type mismatch in columns to concatenate.");
  return cudf::experimental::type_dispatcher(type,
      create_column_from_view_vector{columns_to_concat, stream, mr});
}

}  // namespace cudf
