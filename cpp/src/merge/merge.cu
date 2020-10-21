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
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/copying.hpp>
#include <cudf/detail/merge.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/merge.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/tuple.h>

#include <queue>
#include <vector>

namespace {  // anonym.

using namespace cudf;

using detail::side;
using index_type = detail::index_type;

/**
 * @brief Merges the bits of two validity bitmasks.
 *
 * Merges the bits from two column_device_views into the destination column_device_view
 * according to `merged_indices` map such that bit `i` in `out_col`
 * will be equal to bit `thrust::get<1>(merged_indices[i])` from `left_dcol`
 * if `thrust::get<0>(merged_indices[i])` equals `side::LEFT`; otherwise,
 * from `right_dcol`.
 *
 * `left_dcol`, `right_dcol` and `out_dcol` must not
 * overlap.
 *
 * @tparam left_have_valids Indicates whether left_dcol mask is unallocated (hence, ALL_VALID)
 * @tparam right_have_valids Indicates whether right_dcol mask is unallocated (hence ALL_VALID)
 * @param[in] left_dcol The left column_device_view whose bits will be merged
 * @param[in] right_dcol The right column_device_view whose bits will be merged
 * @param[out] out_dcol The output mutable_column_device_view after merging the left and right
 * @param[in] num_destination_rows The number of rows in the out_dcol
 * @param[in] merged_indices The map that indicates the source of the input and index
 * to be copied to the output. Length must be equal to `num_destination_rows`
 */
template <bool left_have_valids, bool right_have_valids>
__global__ void materialize_merged_bitmask_kernel(
  column_device_view left_dcol,
  column_device_view right_dcol,
  mutable_column_device_view out_dcol,
  size_type const num_destination_rows,
  index_type const* const __restrict__ merged_indices)
{
  size_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads = __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    index_type const& merged_idx = merged_indices[destination_row];
    side const src_side          = thrust::get<0>(merged_idx);
    size_type const src_row      = thrust::get<1>(merged_idx);
    bool const from_left{src_side == side::LEFT};
    bool source_bit_is_valid{true};
    if (left_have_valids && from_left) {
      source_bit_is_valid = left_dcol.is_valid_nocheck(src_row);
    } else if (right_have_valids && !from_left) {
      source_bit_is_valid = right_dcol.is_valid_nocheck(src_row);
    }

    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    bitmask_type const result_mask{__ballot_sync(active_threads, source_bit_is_valid)};

    size_type const output_element = word_index(destination_row);

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) { out_dcol.set_mask_word(output_element, result_mask); }

    destination_row += blockDim.x * gridDim.x;
    active_threads = __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

void materialize_bitmask(column_view const& left_col,
                         column_view const& right_col,
                         mutable_column_view& out_col,
                         index_type const* merged_indices,
                         cudaStream_t stream)
{
  constexpr size_type BLOCK_SIZE{256};
  detail::grid_1d grid_config{out_col.size(), BLOCK_SIZE};

  auto p_left_dcol  = column_device_view::create(left_col);
  auto p_right_dcol = column_device_view::create(right_col);
  auto p_out_dcol   = mutable_column_device_view::create(out_col);

  auto left_valid  = *p_left_dcol;
  auto right_valid = *p_right_dcol;
  auto out_valid   = *p_out_dcol;

  if (left_col.has_nulls()) {
    if (right_col.has_nulls()) {
      materialize_merged_bitmask_kernel<true, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>(
          left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      materialize_merged_bitmask_kernel<true, false>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>(
          left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    }
  } else {
    if (right_col.has_nulls()) {
      materialize_merged_bitmask_kernel<false, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>(
          left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      CUDF_FAIL("materialize_merged_bitmask_kernel<false, false>() should never be called.");
    }
  }

  CHECK_CUDA(stream);
}

/**
 * @brief Generates the row indices and source side (left or right) in accordance with the index
 * columns.
 *
 *
 * @tparam index_type Indicates the type to be used to collect index and side information;
 * @param[in] left_table The left table_view to be merged
 * @param[in] right_table The right table_view to be merged
 * @param[in] column_order Sort order types of index columns
 * @param[in] null_precedence Array indicating the order of nulls with respect to non-nulls for the
 * index columns
 * @param[in] nullable Flag indicating if at least one of the table_view arguments has nulls
 * (defaults to true)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A vector of merged indices
 */
rmm::device_vector<index_type> generate_merged_indices(
  table_view const& left_table,
  table_view const& right_table,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  bool nullable       = true,
  cudaStream_t stream = nullptr)
{
  const size_type left_size  = left_table.num_rows();
  const size_type right_size = right_table.num_rows();
  const size_type total_size = left_size + right_size;

  thrust::constant_iterator<side> left_side(side::LEFT);
  thrust::constant_iterator<side> right_side(side::RIGHT);

  auto left_indices  = thrust::make_counting_iterator(static_cast<size_type>(0));
  auto right_indices = thrust::make_counting_iterator(static_cast<size_type>(0));

  auto left_begin_zip_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
  auto right_begin_zip_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

  auto left_end_zip_iterator =
    thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
  auto right_end_zip_iterator = thrust::make_zip_iterator(
    thrust::make_tuple(right_side + right_size, right_indices + right_size));

  rmm::device_vector<index_type> merged_indices(total_size);

  auto lhs_device_view = table_device_view::create(left_table, stream);
  auto rhs_device_view = table_device_view::create(right_table, stream);

  rmm::device_vector<order> d_column_order(column_order);

  auto exec_pol = rmm::exec_policy(stream);
  if (nullable) {
    rmm::device_vector<null_order> d_null_precedence(null_precedence);

    auto ineq_op =
      detail::row_lexicographic_tagged_comparator<true>(*lhs_device_view,
                                                        *rhs_device_view,
                                                        d_column_order.data().get(),
                                                        d_null_precedence.data().get());
    thrust::merge(exec_pol->on(stream),
                  left_begin_zip_iterator,
                  left_end_zip_iterator,
                  right_begin_zip_iterator,
                  right_end_zip_iterator,
                  merged_indices.begin(),
                  ineq_op);
  } else {
    auto ineq_op = detail::row_lexicographic_tagged_comparator<false>(
      *lhs_device_view, *rhs_device_view, d_column_order.data().get());
    thrust::merge(exec_pol->on(stream),
                  left_begin_zip_iterator,
                  left_end_zip_iterator,
                  right_begin_zip_iterator,
                  right_end_zip_iterator,
                  merged_indices.begin(),
                  ineq_op);
  }

  CHECK_CUDA(stream);

  return merged_indices;
}

}  // namespace

namespace cudf {
namespace detail {
// generate merged column
// given row order of merged tables
//(ordered according to indices of key_cols)
// and the 2 columns to merge
//
struct column_merger {
  using index_vector = rmm::device_vector<index_type>;
  explicit column_merger(
    index_vector const& row_order,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
    cudaStream_t stream                 = nullptr)
    : row_order_(row_order), mr_(mr), stream_(stream)
  {
  }

  // column merger operator;
  //
  template <typename Element>  // required: column type
  std::unique_ptr<column> operator()(column_view const& lcol, column_view const& rcol) const
  {
    auto lsz         = lcol.size();
    auto merged_size = lsz + rcol.size();
    auto type        = lcol.type();
    auto merged_col  = lcol.has_nulls() ? cudf::allocate_like(lcol, merged_size)
                                       : cudf::allocate_like(rcol, merged_size);

    //"gather" data from lcol, rcol according to row_order_ "map"
    //(directly calling gather() won't work because
    // lcol, rcol indices overlap!)
    //
    cudf::mutable_column_view merged_view = merged_col->mutable_view();

    // initialize null_mask to all valid:
    //
    // Note: this initialization in conjunction with _conditionally_
    // calling materialize_bitmask() below covers the case
    // materialize_merged_bitmask_kernel<false, false>()
    // which won't be called anymore (because of the _condition_ below)
    //
    cudf::set_null_mask(merged_view.null_mask(), 0, merged_view.size(), true, stream_);

    // set the null count:
    //
    merged_col->set_null_count(lcol.null_count() + rcol.null_count());

    using Type = device_storage_type_t<Element>;

    // to resolve view.data()'s types use: Element
    //
    auto const d_lcol = lcol.data<Type>();
    auto const d_rcol = rcol.data<Type>();

    auto exe_pol = rmm::exec_policy(stream_);

    // capture lcol, rcol
    // and "gather" into merged_view.data()[indx_merged]
    // from lcol or rcol, depending on side;
    //
    thrust::transform(exe_pol->on(stream_),
                      row_order_.begin(),
                      row_order_.end(),
                      merged_view.begin<Type>(),
                      [d_lcol, d_rcol] __device__(index_type const& index_pair) {
                        // When C++17, use structure bindings
                        auto side  = thrust::get<0>(index_pair);
                        auto index = thrust::get<1>(index_pair);
                        return side == side::LEFT ? d_lcol[index] : d_rcol[index];
                      });

    // CAVEAT: conditional call below is erroneous without
    // set_null_mask() call (see TODO above):
    //
    if (lcol.has_nulls() || rcol.has_nulls()) {
      // resolve null mask:
      //
      materialize_bitmask(lcol, rcol, merged_view, row_order_.data().get(), stream_);
    }

    return merged_col;
  }

 private:
  index_vector const& row_order_;
  rmm::mr::device_memory_resource* mr_;
  cudaStream_t stream_;
};

// specialization for strings
template <>
std::unique_ptr<column> column_merger::operator()<cudf::string_view>(column_view const& lcol,
                                                                     column_view const& rcol) const
{
  auto column = strings::detail::merge<index_type>(strings_column_view(lcol),
                                                   strings_column_view(rcol),
                                                   row_order_.begin(),
                                                   row_order_.end(),
                                                   mr_,
                                                   stream_);
  if (lcol.has_nulls() || rcol.has_nulls()) {
    auto merged_view = column->mutable_view();
    materialize_bitmask(lcol, rcol, merged_view, row_order_.data().get(), stream_);
  }
  return column;
}

// specialization for dictionary
template <>
std::unique_ptr<column> column_merger::operator()<cudf::dictionary32>(column_view const& lcol,
                                                                      column_view const& rcol) const
{
  CUDF_FAIL("dictionary not supported yet");
}

using table_ptr_type = std::unique_ptr<cudf::table>;

namespace {
table_ptr_type merge(cudf::table_view const& left_table,
                     cudf::table_view const& right_table,
                     std::vector<cudf::size_type> const& key_cols,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream = 0)
{
  // collect index columns for lhs, rhs, resp.
  //
  cudf::table_view index_left_view{left_table.select(key_cols)};
  cudf::table_view index_right_view{right_table.select(key_cols)};
  bool const nullable = cudf::has_nulls(index_left_view) || cudf::has_nulls(index_right_view);

  // extract merged row order according to indices:
  //
  rmm::device_vector<index_type> merged_indices = generate_merged_indices(
    index_left_view, index_right_view, column_order, null_precedence, nullable);

  // create merged table:
  //
  auto const n_cols = left_table.num_columns();
  std::vector<std::unique_ptr<column>> merged_cols;
  merged_cols.reserve(n_cols);

  column_merger merger{merged_indices, mr, stream};
  transform(left_table.begin(),
            left_table.end(),
            right_table.begin(),
            std::back_inserter(merged_cols),
            [&](auto const& left_col, auto const& right_col) {
              return cudf::type_dispatcher(left_col.type(), merger, left_col, right_col);
            });

  return std::make_unique<cudf::table>(std::move(merged_cols));
}

struct merge_queue_item {
  table_view view;
  table_ptr_type table;
  // Priority is a separate member to ensure that moving from an object
  // does not change its priority (which would ruin the queue invariant)
  cudf::size_type priority = 0;

  merge_queue_item(table_view const& view, table_ptr_type&& table)
    : view{view}, table{std::move(table)}, priority{-view.num_rows()}
  {
  }

  bool operator<(merge_queue_item const& other) const { return priority < other.priority; }
};

// Helper function to ensure that moving out of the priority_queue is "atomic"
template <typename T>
T top_and_pop(std::priority_queue<T>& q)
{
  auto moved = std::move(const_cast<T&>(q.top()));
  q.pop();
  return moved;
}

}  // namespace

table_ptr_type merge(std::vector<table_view> const& tables_to_merge,
                     std::vector<cudf::size_type> const& key_cols,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     rmm::mr::device_memory_resource* mr,
                     cudaStream_t stream = 0)
{
  if (tables_to_merge.empty()) { return std::make_unique<cudf::table>(); }

  auto const& first_table = tables_to_merge.front();
  auto const n_cols       = first_table.num_columns();

  CUDF_EXPECTS(std::all_of(tables_to_merge.cbegin(),
                           tables_to_merge.cend(),
                           [n_cols](auto const& tbl) { return n_cols == tbl.num_columns(); }),
               "Mismatched number of columns");
  CUDF_EXPECTS(
    std::all_of(tables_to_merge.cbegin(),
                tables_to_merge.cend(),
                [&](auto const& tbl) { return cudf::have_same_types(first_table, tbl); }),
    "Mismatched column types");

  CUDF_EXPECTS(!key_cols.empty(), "Empty key_cols");
  CUDF_EXPECTS(key_cols.size() <= static_cast<size_t>(n_cols), "Too many values in key_cols");

  CUDF_EXPECTS(key_cols.size() == column_order.size(),
               "Mismatched size between key_cols and column_order");

  // A queue of (table view, table) pairs
  std::priority_queue<merge_queue_item> merge_queue;
  // The table pointer is null if we do not own the table (input tables)
  std::for_each(tables_to_merge.begin(), tables_to_merge.end(), [&](auto const& table) {
    if (table.num_rows() > 0) merge_queue.emplace(table, table_ptr_type());
  });

  // If there is only one non-empty table_view, return its copy
  if (merge_queue.size() == 1) { return std::make_unique<cudf::table>(merge_queue.top().view); }
  // No inputs have rows, return a table with same columns as the first one
  if (merge_queue.empty()) { return empty_like(first_table); }

  // Pick the two smallest tables and merge them
  // Until there is only one table left in the queue
  while (merge_queue.size() > 1) {
    // To delete the intermediate table at the end of the block
    auto const left_table = top_and_pop(merge_queue);
    // Deallocated at the end of the block
    auto const right_table = top_and_pop(merge_queue);

    // Only use mr for the output table
    auto const& new_tbl_rm = merge_queue.empty() ? mr : rmm::mr::get_current_device_resource();
    auto merged_table      = merge(left_table.view,
                              right_table.view,
                              key_cols,
                              column_order,
                              null_precedence,
                              new_tbl_rm,
                              stream);

    auto const merged_table_view = merged_table->view();
    merge_queue.emplace(merged_table_view, std::move(merged_table));
  }

  return std::move(top_and_pop(merge_queue).table);
}

}  // namespace detail

std::unique_ptr<cudf::table> merge(std::vector<table_view> const& tables_to_merge,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& column_order,
                                   std::vector<cudf::null_order> const& null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::merge(tables_to_merge, key_cols, column_order, null_precedence, mr);
}

}  // namespace cudf
