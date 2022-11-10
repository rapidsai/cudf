/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cudf/detail/concatenate.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/structs/detail/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace cudf {
namespace detail {

// From benchmark data, the fused kernel optimization appears to perform better
// when there are more than a trivial number of columns, or when the null mask
// can also be computed at the same time
constexpr bool use_fused_kernel_heuristic(bool const has_nulls, size_t const num_columns)
{
  return has_nulls || num_columns > 4;
}

auto create_device_views(host_span<column_view const> views, rmm::cuda_stream_view stream)
{
  // Create device views for each input view
  using CDViewPtr = decltype(
    column_device_view::create(std::declval<column_view>(), std::declval<rmm::cuda_stream_view>()));
  auto device_view_owners = std::vector<CDViewPtr>(views.size());
  std::transform(views.begin(), views.end(), device_view_owners.begin(), [stream](auto const& col) {
    return column_device_view::create(col, stream);
  });

  // Assemble contiguous array of device views
  auto device_views = thrust::host_vector<column_device_view>();
  device_views.reserve(views.size());
  std::transform(device_view_owners.cbegin(),
                 device_view_owners.cend(),
                 std::back_inserter(device_views),
                 [](auto const& col) { return *col; });

  auto d_views = make_device_uvector_async(device_views, stream);

  // Compute the partition offsets
  auto offsets = thrust::host_vector<size_t>(views.size() + 1);
  thrust::transform_inclusive_scan(
    thrust::host,
    device_views.cbegin(),
    device_views.cend(),
    std::next(offsets.begin()),
    [](auto const& col) { return col.size(); },
    thrust::plus{});
  auto d_offsets         = make_device_uvector_async(offsets, stream);
  auto const output_size = offsets.back();

  return std::make_tuple(
    std::move(device_view_owners), std::move(d_views), std::move(d_offsets), output_size);
}

/**
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @param views Array of column_device_view
 * @param output_offsets Prefix sum of sizes of elements of `views`
 * @param number_of_views Size of `views` array
 * @param dest_mask The output buffer to copy null masks into
 * @param number_of_mask_bits The total number of null masks bits that are being
 * copied
 */
__global__ void concatenate_masks_kernel(column_device_view const* views,
                                         size_t const* output_offsets,
                                         size_type number_of_views,
                                         bitmask_type* dest_mask,
                                         size_type number_of_mask_bits)
{
  size_type mask_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask = __ballot_sync(0xFFFF'FFFFu, mask_index < number_of_mask_bits);

  while (mask_index < number_of_mask_bits) {
    size_type const source_view_index =
      thrust::upper_bound(
        thrust::seq, output_offsets, output_offsets + number_of_views, mask_index) -
      output_offsets - 1;
    bool bit_is_set = true;
    if (source_view_index < number_of_views) {
      size_type const column_element_index = mask_index - output_offsets[source_view_index];
      bit_is_set = views[source_view_index].is_valid(column_element_index);
    }
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    if (threadIdx.x % detail::warp_size == 0) { dest_mask[word_index(mask_index)] = new_word; }

    mask_index += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, mask_index < number_of_mask_bits);
  }
}

void concatenate_masks(device_span<column_device_view const> d_views,
                       device_span<size_t const> d_offsets,
                       bitmask_type* dest_mask,
                       size_type output_size,
                       rmm::cuda_stream_view stream)
{
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  concatenate_masks_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    d_views.data(),
    d_offsets.data(),
    static_cast<size_type>(d_views.size()),
    dest_mask,
    output_size);
}

void concatenate_masks(host_span<column_view const> views,
                       bitmask_type* dest_mask,
                       rmm::cuda_stream_view stream)
{
  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  concatenate_masks(d_views, d_offsets, dest_mask, output_size, stream);
}

template <typename T, size_type block_size, bool Nullable>
__global__ void fused_concatenate_kernel(column_device_view const* input_views,
                                         size_t const* input_offsets,
                                         size_type num_input_views,
                                         mutable_column_device_view output_view,
                                         size_type* out_valid_count)
{
  auto const output_size = output_view.size();
  auto* output_data      = output_view.data<T>();

  int64_t output_index       = threadIdx.x + blockIdx.x * blockDim.x;
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) { active_mask = __ballot_sync(0xFFFF'FFFFu, output_index < output_size); }
  while (output_index < output_size) {
    // Lookup input index by searching for output index in offsets
    auto const offset_it            = thrust::prev(thrust::upper_bound(
      thrust::seq, input_offsets, input_offsets + num_input_views, output_index));
    size_type const partition_index = offset_it - input_offsets;

    // Copy input data to output
    auto const offset_index   = output_index - *offset_it;
    auto const& input_view    = input_views[partition_index];
    auto const* input_data    = input_view.data<T>();
    output_data[output_index] = input_data[offset_index];

    if (Nullable) {
      bool const bit_is_set       = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % detail::warp_size == 0) {
        output_view.null_mask()[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += blockDim.x * gridDim.x;
    if (Nullable) { active_mask = __ballot_sync(active_mask, output_index < output_size); }
  }

  if (Nullable) {
    using detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
  }
}

template <typename T>
std::unique_ptr<column> fused_concatenate(host_span<column_view const> views,
                                          bool const has_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  using mask_policy = cudf::mask_allocation_policy;

  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  CUDF_EXPECTS(output_size <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds size_type range");

  // Allocate output
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto out_col      = detail::allocate_like(views.front(), output_size, policy, stream, mr);
  auto out_view     = out_col->mutable_view();
  auto d_out_view   = mutable_column_device_view::create(out_view, stream);

  rmm::device_scalar<size_type> d_valid_count(0, stream);

  // Launch kernel
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  auto const kernel = has_nulls ? fused_concatenate_kernel<T, block_size, true>
                                : fused_concatenate_kernel<T, block_size, false>;
  kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    d_views.data(),
    d_offsets.data(),
    static_cast<size_type>(d_views.size()),
    *d_out_view,
    d_valid_count.data());

  if (has_nulls) {
    out_col->set_null_count(output_size - d_valid_count.value(stream));
  } else {
    out_col->set_null_count(0);  // prevent null count from being materialized
  }

  return out_col;
}

template <typename T>
std::unique_ptr<column> for_each_concatenate(host_span<column_view const> views,
                                             bool const has_nulls,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  size_type const total_element_count =
    std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
      return accumulator + v.size();
    });

  using mask_policy = cudf::mask_allocation_policy;
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto col = cudf::detail::allocate_like(views.front(), total_element_count, policy, stream, mr);

  auto m_view = col->mutable_view();

  auto count = 0;
  for (auto& v : views) {
    thrust::copy(rmm::exec_policy(stream), v.begin<T>(), v.end<T>(), m_view.begin<T>() + count);
    count += v.size();
  }

  // If concatenated column is nullable, proceed to calculate it
  if (has_nulls) {
    cudf::detail::concatenate_masks(views, (col->mutable_view()).null_mask(), stream);
  } else {
    col->set_null_count(0);  // prevent null count from being materialized
  }

  return col;
}

struct concatenate_dispatch {
  host_span<column_view const> views;
  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;

  // fixed width
  template <typename T>
  std::unique_ptr<column> operator()()
  {
    bool const has_nulls =
      std::any_of(views.begin(), views.end(), [](auto const& col) { return col.has_nulls(); });

    // Use a heuristic to guess when the fused kernel will be faster
    if (use_fused_kernel_heuristic(has_nulls, views.size())) {
      return fused_concatenate<T>(views, has_nulls, stream, mr);
    } else {
      return for_each_concatenate<T>(views, has_nulls, stream, mr);
    }
  }
};

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::dictionary32>()
{
  return cudf::dictionary::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::string_view>()
{
  return cudf::strings::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::list_view>()
{
  return cudf::lists::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::struct_view>()
{
  return cudf::structs::detail::concatenate(views, stream, mr);
}

namespace {

void bounds_and_type_check(host_span<column_view const> cols, rmm::cuda_stream_view stream);

/**
 * @brief Functor for traversing child columns and recursively verifying concatenation
 * bounds and types.
 */
class traverse_children {
 public:
  // nothing to do for simple types.
  template <typename T>
  void operator()(host_span<column_view const>, rmm::cuda_stream_view)
  {
  }

 private:
  // verify length of concatenated offsets.
  void check_offsets_size(host_span<column_view const> cols)
  {
    // offsets.  we can't just add up the total sizes of all offset child columns because each one
    // has an extra value, regardless of the # of parent rows.  So we have to add up the total # of
    // rows in the base column and add 1 at the end
    size_t const total_offset_count =
      std::accumulate(cols.begin(),
                      cols.end(),
                      std::size_t{},
                      [](size_t a, auto const& b) -> size_t { return a + b.size(); }) +
      1;
    // note:  output text must include "exceeds size_type range" for python error handling
    CUDF_EXPECTS(total_offset_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
                 "Total number of concatenated offsets exceeds size_type range");
  }
};

template <>
void traverse_children::operator()<cudf::string_view>(host_span<column_view const> cols,
                                                      rmm::cuda_stream_view stream)
{
  // verify offsets
  check_offsets_size(cols);

  // chars
  size_t const total_char_count = std::accumulate(
    cols.begin(), cols.end(), std::size_t{}, [stream](size_t a, auto const& b) -> size_t {
      strings_column_view scv(b);
      return a + (scv.is_empty() ? 0
                  // if the column is unsliced, skip the offset retrieval.
                  : scv.offset() > 0
                    ? cudf::detail::get_value<offset_type>(
                        scv.offsets(), scv.offset() + scv.size(), stream) -
                        cudf::detail::get_value<offset_type>(scv.offsets(), scv.offset(), stream)
                  // if the offset() is 0, it can still be sliced to a shorter length. in this case
                  // we only need to read a single offset. otherwise just return the full length
                  // (chars_size())
                  : scv.size() + 1 == scv.offsets().size()
                    ? scv.chars_size()
                    : cudf::detail::get_value<offset_type>(scv.offsets(), scv.size(), stream));
    });
  // note:  output text must include "exceeds size_type range" for python error handling
  CUDF_EXPECTS(total_char_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated chars exceeds size_type range");
}

template <>
void traverse_children::operator()<cudf::struct_view>(host_span<column_view const> cols,
                                                      rmm::cuda_stream_view stream)
{
  // march each child
  auto child_iter         = thrust::make_counting_iterator(0);
  auto const num_children = cols.front().num_children();
  std::vector<column_view> nth_children;
  nth_children.reserve(cols.size());
  std::for_each(child_iter, child_iter + num_children, [&](auto child_index) {
    std::transform(cols.begin(),
                   cols.end(),
                   std::back_inserter(nth_children),
                   [child_index, stream](column_view const& col) {
                     structs_column_view scv(col);
                     return scv.get_sliced_child(child_index);
                   });

    bounds_and_type_check(nth_children, stream);
    nth_children.clear();
  });
}

template <>
void traverse_children::operator()<cudf::list_view>(host_span<column_view const> cols,
                                                    rmm::cuda_stream_view stream)
{
  // verify offsets
  check_offsets_size(cols);

  // recurse into the child columns
  std::vector<column_view> nth_children;
  nth_children.reserve(cols.size());
  std::transform(
    cols.begin(), cols.end(), std::back_inserter(nth_children), [stream](column_view const& col) {
      lists_column_view lcv(col);
      return lcv.get_sliced_child(stream);
    });
  bounds_and_type_check(nth_children, stream);
}

/**
 * @brief Verifies that the sum of the sizes of all the columns to be concatenated
 * will not exceed the max value of size_type, and verifies all column types match
 *
 * @param columns_to_concat Span of columns to check
 *
 * @throws cudf::logic_error if the total length of the concatenated columns would
 * exceed the max value of size_type
 *
 * @throws cudf::logic_error if all of the input column types don't match
 */
void bounds_and_type_check(host_span<column_view const> cols, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(std::all_of(cols.begin(),
                           cols.end(),
                           [expected_type = cols.front().type()](auto const& c) {
                             return c.type() == expected_type;
                           }),
               "Type mismatch in columns to concatenate.");

  // total size of all concatenated rows
  size_t const total_row_count =
    std::accumulate(cols.begin(), cols.end(), std::size_t{}, [](size_t a, auto const& b) {
      return a + static_cast<size_t>(b.size());
    });
  // note:  output text must include "exceeds size_type range" for python error handling
  CUDF_EXPECTS(total_row_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds size_type range");

  // traverse children
  cudf::type_dispatcher(cols.front().type(), traverse_children{}, cols, stream);
}

}  // anonymous namespace

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(host_span<column_view const> columns_to_concat,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  // verify all types match and that we won't overflow size_type in output size
  bounds_and_type_check(columns_to_concat, stream);

  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(), [](column_view const& c) {
        return c.is_empty();
      })) {
    return empty_like(columns_to_concat.front());
  }

  return type_dispatcher<dispatch_storage_type>(
    columns_to_concat.front().type(), concatenate_dispatch{columns_to_concat, stream, mr});
}

std::unique_ptr<table> concatenate(host_span<table_view const> tables_to_concat,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(),
                           tables_to_concat.end(),
                           [&first_table](auto const& t) {
                             return t.num_columns() == first_table.num_columns();
                           }),
               "Mismatch in table columns to concatenate.");

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < first_table.num_columns(); ++i) {
    std::vector<column_view> cols;
    std::transform(tables_to_concat.begin(),
                   tables_to_concat.end(),
                   std::back_inserter(cols),
                   [i](auto const& t) { return t.column(i); });

    // verify all types match and that we won't overflow size_type in output size
    bounds_and_type_check(cols, stream);
    concat_columns.emplace_back(detail::concatenate(cols, stream, mr));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

rmm::device_buffer concatenate_masks(host_span<column_view const> views,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  bool const has_nulls =
    std::any_of(views.begin(), views.end(), [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
    size_type const total_element_count =
      std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
        return accumulator + v.size();
      });

    rmm::device_buffer null_mask =
      create_null_mask(total_element_count, mask_state::UNINITIALIZED, mr);

    detail::concatenate_masks(views, static_cast<bitmask_type*>(null_mask.data()), stream);

    return null_mask;
  }
  // no nulls, so return an empty device buffer
  return rmm::device_buffer{0, stream, mr};
}

}  // namespace detail

rmm::device_buffer concatenate_masks(host_span<column_view const> views,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_masks(views, cudf::get_default_stream(), mr);
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(host_span<column_view const> columns_to_concat,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(columns_to_concat, cudf::get_default_stream(), mr);
}

std::unique_ptr<table> concatenate(host_span<table_view const> tables_to_concat,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(tables_to_concat, cudf::get_default_stream(), mr);
}

}  // namespace cudf
