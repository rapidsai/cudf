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
#include <cudf/utilities/bit.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <algorithm>
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
column::column(column &&other)
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
      null_count(),  0,
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

/**---------------------------------------------------------------------------*
 * @brief Copies the bits starting at the specified offset from a source
 * bitmask into the destination bitmask.
 *
 * Bit `i` in `destination` will be equal to bit `i + offset` from `source`.
 *
 * @param destination The mask to copy into
 * @param source The mask to copy from
 * @param bit_offset The offset into `source` from which to begin the copy
 * @param number_of_bits The number of bits to copy
 *---------------------------------------------------------------------------**/
__global__ void copy_offset_bitmask(bitmask_type *__restrict__ destination,
                                    bitmask_type const *__restrict__ source,
                                    size_type bit_offset,
                                    size_type number_of_bits) {
  constexpr size_type warp_size{32};
  size_type destination_bit_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask =
      __ballot_sync(0xFFFF'FFFF, destination_bit_index < number_of_bits);

  while (destination_bit_index < number_of_bits) {
    bitmask_type const new_word = __ballot_sync(
        active_mask, bit_is_set(source, bit_offset + destination_bit_index));

    if (threadIdx.x % warp_size == 0) {
      destination[word_index(destination_bit_index)] = new_word;
    }

    destination_bit_index += blockDim.x * gridDim.x;
    active_mask =
        __ballot_sync(active_mask, destination_bit_index < number_of_bits);
  }
}

// Copy from a view
column::column(column_view view, cudaStream_t stream,
               rmm::mr::device_memory_resource *mr) {
  CUDF_FAIL("Copying from a view is not supported yet.");
}
/*
: _type{view.type()},
_size{view.size()},
// TODO: Fix for variable-width types
_data{view.head() + (view.offset() * cudf::size_of(view.type())),
view.size() * cudf::size_of(view.type()), stream, mr},
_null_count{view.null_count()} {

if (view.nullable()) {
// If there's no offset, do a simple copy
if (view.offset() == 0) {
_null_mask =
rmm::device_buffer{static_cast<void const *>(view.null_mask()),
              bitmask_allocation_size_bytes(size()), stream, mr},
} else {
CUDF_EXPECTS(view.offset() > 0, "Invalid view offset.");
// If there's a non-zero offset, need to handle offset bitmask elements
_null_mask =
rmm::device_buffer{bitmask_allocation_size_bytes(size()), stream, mr};
cudf::util::cuda::grid_config_1d config(view.size(), 256);
copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0,
             stream>>>(
static_cast<bitmask_type *>(_null_mask.data()), view.null_mask(),
view.offset(), view.size());
CHECK_STREAM(stream);
}
}

// Implicitly invokes conversion of the view's child views to `column`s
for (size_type i = 0; i < view.num_children(); ++i) {
_children.emplace_back(view.child(i), stream, mr);
}
}
*/

}  // namespace cudf
