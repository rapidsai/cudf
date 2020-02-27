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
#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/strings/copying.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/copying.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/binary_search.h>

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

void column::set_null_mask(rmm::device_buffer&& new_null_mask,
                   size_type new_null_count) {
  if(new_null_count > 0){
    CUDF_EXPECTS(new_null_mask.size() >=
                   cudf::bitmask_allocation_size_bytes(this->size()),
                 "Column with null values must be nullable and the null mask \
                  buffer size should match the size of the column.");
    }
    _null_mask = std::move(new_null_mask);  // move
    _null_count = new_null_count;
}

void column::set_null_mask(rmm::device_buffer const& new_null_mask,
                   size_type new_null_count) {
  if(new_null_count > 0){
    CUDF_EXPECTS(new_null_mask.size() >=
                   cudf::bitmask_allocation_size_bytes(this->size()),
                 "Column with null values must be nullable and the null mask \
                  buffer size should match the size of the column.");
    }
    _null_mask = new_null_mask;  // copy
    _null_count = new_null_count;
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
   return cudf::strings::detail::slice(sview, 0, view.size(), 1, stream, mr);
 }

 template <typename ColumnType,
           std::enable_if_t<std::is_same<ColumnType, cudf::dictionary32>::value>* = nullptr>
 std::unique_ptr<column> operator()() {
   CUDF_FAIL("dictionary not supported yet");
 }
 
 template <typename ColumnType,
           std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
 std::unique_ptr<column> operator()() {

   std::vector<std::unique_ptr<column>> children;
   for (size_type i = 0; i < view.num_children(); ++i) {
     children.emplace_back(std::make_unique<column>(view.child(i), stream, mr));
   }

   return std::make_unique<column>(view.type(), view.size(),
       rmm::device_buffer{
       static_cast<const char*>(view.head()) +
       (view.offset() * cudf::size_of(view.type())),
       view.size() * cudf::size_of(view.type()), stream, mr},
       cudf::copy_bitmask(view, stream, mr),
       view.null_count(), std::move(children));
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
           std::enable_if_t<std::is_same<ColumnType, cudf::dictionary32>::value>* = nullptr>
 std::unique_ptr<column> operator()() {
   CUDF_FAIL("dictionary not supported yet");
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

template <typename T>
struct partition_map_fn {
  size_type const* offsets_ptr;
  size_type const* partition_ptr;
  T const* const* data_ptr;

  T __device__ operator()(size_type i) {
    auto const partition = partition_ptr[i];
    auto const offset = offsets_ptr[partition];
    return data_ptr[partition][i - offset];
  }
};

template <typename T>
struct binary_search_fn {
  size_type const* offsets_ptr;
  size_t const input_size;
  T const* const* data_ptr;

  T __device__ operator()(size_type i) {
    // Look up the current index in the offsets table to find the partition
    auto const offset_it = thrust::prev(thrust::upper_bound(thrust::seq,
        offsets_ptr, offsets_ptr + input_size, i));
    auto const partition = offset_it - offsets_ptr;
    auto const offset = *offset_it;
    return data_ptr[partition][i - offset];
  }
};

auto compute_partition_map(rmm::device_vector<size_type> const& d_offsets,
    size_t output_size, cudaStream_t stream) {

  auto d_partition_map = rmm::device_vector<size_type>(output_size);

  // Scatter 1s at the start of each partition, then scan to fill the map
  auto const const_it = thrust::make_constant_iterator(size_type{1});
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
      const_it, const_it + (d_offsets.size() - 2), // skip first and last offset
      std::next(d_offsets.cbegin()), // skip first offset, leaving it set to 0
      d_partition_map.begin());

  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
      d_partition_map.cbegin(), d_partition_map.cend(),
      d_partition_map.begin());

  return d_partition_map;
}

// Allow strategy switching at runtime for easier benchmarking
// TODO remove when done
static concatenate_mode current_mode = concatenate_mode::UNOPTIMIZED;
void temp_set_concatenate_mode(concatenate_mode mode) {
  current_mode = mode;
}

struct optimized_concatenate {
  template <typename T,
      std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    using mask_policy = cudf::experimental::mask_allocation_policy;

    // Compute the partition offsets
    auto offsets = thrust::host_vector<size_type>(views.size() + 1);
    // TODO This should be thrust::transform_inclusive_scan, but getting
    // error related to https://github.com/rapidsai/rmm/pull/312
    thrust::transform(views.cbegin(), views.cend(), std::next(offsets.begin()),
        [](column_view const& col) {
          return col.size();
        });
    thrust::inclusive_scan(std::next(offsets.cbegin()), offsets.cend(),
        std::next(offsets.begin()));
    auto const d_offsets = rmm::device_vector<size_type>(offsets);
    auto const output_size = offsets.back();

    // Transform views to array of data pointers
    auto data_ptrs = thrust::host_vector<T const*>(views.size());
    std::transform(views.begin(), views.end(), data_ptrs.begin(),
        [](column_view const& col) {
          return col.data<T>();
        });
    auto const d_data_ptrs = rmm::device_vector<T const*>(data_ptrs);

    // Allocate output column, with null mask if any columns have nulls
    bool const has_nulls = std::any_of(views.begin(), views.end(),
        [](auto const& col) { return col.has_nulls(); });
    auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
    auto out_col = experimental::detail::allocate_like(views.front(),
        output_size, policy, mr, stream);
    auto out_view = out_col->mutable_view();

    // Initialize each output row from its corresponding input view
    // NOTE device lambdas were giving weird errors, so use functor instead
    auto const* offsets_ptr = d_offsets.data().get();
    T const* const* data_ptr = d_data_ptrs.data().get();
    if (current_mode == concatenate_mode::PARTITION_MAP) {
      auto const d_partition_map = compute_partition_map(d_offsets, output_size, stream);
      auto const* partition_ptr = d_partition_map.data().get();
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          out_view.begin<T>(), out_view.end<T>(),
          partition_map_fn<T>{offsets_ptr, partition_ptr, data_ptr});
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          out_view.begin<T>(), out_view.end<T>(),
          binary_search_fn<T>{offsets_ptr, views.size(), data_ptr});
    }

    // If concatenated column is nullable, proceed to calculate it
    // TODO try to fuse this with the data kernel so they share binary search
    if (has_nulls) {
      cudf::detail::concatenate_masks(views, out_view.null_mask(), stream);
    }

    return out_col;
  }

  template <typename ColumnType,
      std::enable_if_t<not is_fixed_width<ColumnType>()>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    CUDF_FAIL("non-fixed-width types not yet supported");
  }
};

// Copy from a view
column::column(column_view view, cudaStream_t stream,
               rmm::mr::device_memory_resource *mr) :
  // Move is needed here because the dereference operator of unique_ptr returns
  // an lvalue reference, which would otherwise dispatch to the copy constructor
  column{std::move(*experimental::type_dispatcher(view.type(),
                    create_column_from_view{view, stream, mr}))} {}

struct nvtx_raii {
  nvtx_raii(char const* name, nvtx::color color) { nvtx::range_push(name, color); }
  ~nvtx_raii() { nvtx::range_pop(); }
};

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
//#if defined(BUILD_BENCHMARKS)
// TODO this doesn't seem to work, so remove after testing
// This shows additional information for profiling in NVTX ranges,
// but at the expense of some extra computation
#if true // for testing, print [num_cols][rows_per_col]
  auto const num_cols = columns_to_concat.size();
  // This should be thrust::transform_reduce, but getting
  // error related to https://github.com/rapidsai/rmm/pull/312
  auto col_sizes = std::vector<size_type>(num_cols);
  thrust::transform(
      columns_to_concat.cbegin(), columns_to_concat.cend(),
      col_sizes.begin(),
      [] __host__ (auto const& col) {
        return col.size();
      });
  auto const rows_per_col = std::accumulate(
      col_sizes.begin(), col_sizes.end(), size_type{0});
  auto const message = std::string("cudf::concatenate[")
      + std::to_string(num_cols) + "]["
      + std::to_string(rows_per_col / num_cols) + "]";
  nvtx_raii range(message.c_str(), nvtx::color::DARK_GREEN);
#else
  nvtx_raii range("cudf::concatenate", nvtx::color::DARK_GREEN);
#endif

  if (columns_to_concat.empty()) { return std::make_unique<column>(); }

  data_type type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
        [type](auto const& c) { return c.type() == type; }),
      "Type mismatch in columns to concatenate.");

  switch (current_mode) {
    case concatenate_mode::UNOPTIMIZED:
      return experimental::type_dispatcher(type,
          create_column_from_view_vector{columns_to_concat, stream, mr});
    case concatenate_mode::PARTITION_MAP:
    case concatenate_mode::BINARY_SEARCH:
      return experimental::type_dispatcher(type,
          optimized_concatenate{}, columns_to_concat, mr, stream);
    default:
      CUDF_FAIL("Invalid concatenate mode");
  }
}

}  // namespace cudf
