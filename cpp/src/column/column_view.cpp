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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <exception>
#include <numeric>
#include <vector>

namespace cudf {
namespace detail {
column_view_base::column_view_base(data_type type,
                                   size_type size,
                                   void const* data,
                                   bitmask_type const* null_mask,
                                   size_type null_count,
                                   size_type offset)
  : _type{type},
    _size{size},
    _data{data},
    _null_mask{null_mask},
    _null_count{null_count},
    _offset{offset}
{
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  if (type.id() == type_id::EMPTY) {
    _null_count = size;
    CUDF_EXPECTS(nullptr == data, "EMPTY column should have no data.");
    CUDF_EXPECTS(nullptr == null_mask, "EMPTY column should have no null mask.");
  } else if (is_compound(type)) {
    CUDF_EXPECTS(nullptr == data, "Compound (parent) columns cannot have data");
  } else if (size > 0) {
    CUDF_EXPECTS(nullptr != data, "Null data pointer.");
  }

  CUDF_EXPECTS(offset >= 0, "Invalid offset.");

  if ((null_count > 0) and (type.id() != type_id::EMPTY)) {
    CUDF_EXPECTS(nullptr != null_mask, "Invalid null mask for non-zero null count.");
  }
}

size_type column_view_base::null_count(size_type begin, size_type end) const
{
  CUDF_EXPECTS((begin >= 0) && (end <= size()) && (begin <= end), "Range is out of bounds.");
  return (null_count() == 0)
           ? 0
           : cudf::detail::null_count(
               null_mask(), offset() + begin, offset() + end, cudf::get_default_stream());
}

// Struct to use custom hash combine and fold expression
struct HashValue {
  std::size_t hash;
  explicit HashValue(std::size_t h) : hash{h} {}
  HashValue operator^(HashValue const& other) const
  {
    return HashValue{cudf::hashing::detail::hash_combine(hash, other.hash)};
  }
};

template <typename... Ts>
constexpr auto hash(Ts&&... ts)
{
  return (... ^ HashValue(std::hash<Ts>{}(ts))).hash;
}

std::size_t shallow_hash_impl(column_view const& c, bool is_parent_empty = false)
{
  std::size_t const init = (is_parent_empty or c.is_empty())
                             ? hash(c.type(), 0)
                             : hash(c.type(), c.size(), c.head(), c.null_mask(), c.offset());
  return std::accumulate(c.child_begin(),
                         c.child_end(),
                         init,
                         [&c, is_parent_empty](std::size_t hash, auto const& child) {
                           return cudf::hashing::detail::hash_combine(
                             hash, shallow_hash_impl(child, c.is_empty() or is_parent_empty));
                         });
}

std::size_t shallow_hash(column_view const& input) { return shallow_hash_impl(input); }

bool shallow_equivalent_impl(column_view const& lhs,
                             column_view const& rhs,
                             bool is_parent_empty = false)
{
  bool const is_empty = (lhs.is_empty() and rhs.is_empty()) or is_parent_empty;
  return (lhs.type() == rhs.type()) and
         (is_empty or ((lhs.size() == rhs.size()) and (lhs.head() == rhs.head()) and
                       (lhs.null_mask() == rhs.null_mask()) and (lhs.offset() == rhs.offset()))) and
         std::equal(lhs.child_begin(),
                    lhs.child_end(),
                    rhs.child_begin(),
                    rhs.child_end(),
                    [is_empty](auto const& lhs_child, auto const& rhs_child) {
                      return shallow_equivalent_impl(lhs_child, rhs_child, is_empty);
                    });
}
bool is_shallow_equivalent(column_view const& lhs, column_view const& rhs)
{
  return shallow_equivalent_impl(lhs, rhs);
}
}  // namespace detail

// Immutable view constructor
column_view::column_view(data_type type,
                         size_type size,
                         void const* data,
                         bitmask_type const* null_mask,
                         size_type null_count,
                         size_type offset,
                         std::vector<column_view> const& children)
  : detail::column_view_base{type, size, data, null_mask, null_count, offset}, _children{children}
{
  if (type.id() == type_id::EMPTY) {
    CUDF_EXPECTS(num_children() == 0, "EMPTY column cannot have children.");
  }
}

// Mutable view constructor
mutable_column_view::mutable_column_view(data_type type,
                                         size_type size,
                                         void* data,
                                         bitmask_type* null_mask,
                                         size_type null_count,
                                         size_type offset,
                                         std::vector<mutable_column_view> const& children)
  : detail::column_view_base{type, size, data, null_mask, null_count, offset},
    mutable_children{children}
{
  if (type.id() == type_id::EMPTY) {
    CUDF_EXPECTS(num_children() == 0, "EMPTY column cannot have children.");
  }
}

// Update the null count
void mutable_column_view::set_null_count(size_type new_null_count)
{
  if (new_null_count > 0) { CUDF_EXPECTS(nullable(), "Invalid null count."); }
  _null_count = new_null_count;
}

// Conversion from mutable to immutable
mutable_column_view::operator column_view() const
{
  // Convert children to immutable views
  std::vector<column_view> child_views(num_children());
  std::copy(std::cbegin(mutable_children), std::cend(mutable_children), std::begin(child_views));
  return column_view{_type, _size, _data, _null_mask, _null_count, _offset, std::move(child_views)};
}

size_type count_descendants(column_view parent)
{
  auto descendants = [](auto const& child) { return count_descendants(child); };
  auto begin       = thrust::make_transform_iterator(parent.child_begin(), descendants);
  return std::accumulate(begin, begin + parent.num_children(), size_type{parent.num_children()});
}

column_view bit_cast(column_view const& input, data_type type)
{
  CUDF_EXPECTS(is_bit_castable(input._type, type), "types are not bit-castable");
  return column_view{type,
                     input._size,
                     input._data,
                     input._null_mask,
                     input._null_count,
                     input._offset,
                     input._children};
}

mutable_column_view bit_cast(mutable_column_view const& input, data_type type)
{
  CUDF_EXPECTS(is_bit_castable(input._type, type), "types are not bit-castable");
  return mutable_column_view{type,
                             input._size,
                             const_cast<void*>(input._data),
                             const_cast<cudf::bitmask_type*>(input._null_mask),
                             input._null_count,
                             input._offset,
                             input.mutable_children};
}

}  // namespace cudf
