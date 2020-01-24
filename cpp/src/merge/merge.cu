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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/merge.h>

#include <vector>

#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>

#include <cudf/detail/merge.cuh>
#include <cudf/strings/detail/merge.cuh>

namespace { // anonym.

using namespace cudf;

using experimental::detail::side;
using index_type = experimental::detail::index_type;


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
__global__ void materialize_merged_bitmask_kernel(column_device_view left_dcol,
                                                  column_device_view right_dcol,
                                                  mutable_column_device_view out_dcol,
                                                  size_type const num_destination_rows,
                                                  index_type const* const __restrict__ merged_indices) {
  size_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
    __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    index_type const& merged_idx = merged_indices[destination_row];
    side const src_side = thrust::get<0>(merged_idx);
    size_type const src_row  = thrust::get<1>(merged_idx);
    bool const from_left{src_side == side::LEFT};
    bool source_bit_is_valid{true};
    if (left_have_valids && from_left) {
      source_bit_is_valid = left_dcol.is_valid_nocheck(src_row);
    }
    else if (right_have_valids && !from_left) {
      source_bit_is_valid = right_dcol.is_valid_nocheck(src_row);
    }

    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    bitmask_type const result_mask{
      __ballot_sync(active_threads, source_bit_is_valid)};

    size_type const output_element = word_index(destination_row);

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      out_dcol.set_mask_word(output_element, result_mask);
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
      __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

void materialize_bitmask(column_view const& left_col,
                         column_view const& right_col,
                         mutable_column_view& out_col,
                         index_type const* merged_indices,
                         cudaStream_t stream) {
  constexpr size_type BLOCK_SIZE{256};
  experimental::detail::grid_1d grid_config {out_col.size(), BLOCK_SIZE };

  auto p_left_dcol  = column_device_view::create(left_col);
  auto p_right_dcol = column_device_view::create(right_col);
  auto p_out_dcol   = mutable_column_device_view::create(out_col);

  auto left_valid  = *p_left_dcol;
  auto right_valid = *p_right_dcol;
  auto out_valid   = *p_out_dcol;

  if (p_left_dcol->has_nulls()) {
    if (p_right_dcol->has_nulls()) {
      materialize_merged_bitmask_kernel<true, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      materialize_merged_bitmask_kernel<true, false>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    }
  } else {
    if (p_right_dcol->has_nulls()) {
      materialize_merged_bitmask_kernel<false, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      CUDF_FAIL("materialize_merged_bitmask_kernel<false, false>() should never be called.");
    }
  }

  CHECK_CUDA(stream);
}

/**
 * @brief Generates the row indices and source side (left or right) in accordance with the index columns.
 *
 *
 * @tparam index_type Indicates the type to be used to collect index and side information;
 * @param[in] left_table The left table_view to be merged
 * @param[in] right_tbale The right table_view to be merged
 * @param[in] column_order Sort order types of index columns
 * @param[in] null_precedence Array indicating the order of nulls with respect to non-nulls for the index columns
 * @param[in] nullable Flag indicating if at least one of the table_view arguments has nulls (defaults to true)
 * @param[in] stream CUDA stream (defaults to nullptr)
 *
 * @return A vector of merged indices
 */
rmm::device_vector<index_type>
generate_merged_indices(table_view const& left_table,
                        table_view const& right_table,
                        std::vector<order> const& column_order,
                        std::vector<null_order> const& null_precedence,
                        bool nullable = true,
                        cudaStream_t stream = nullptr) {

    const size_type left_size  = left_table.num_rows();
    const size_type right_size = right_table.num_rows();
    const size_type total_size = left_size + right_size;

    thrust::constant_iterator<side> left_side(side::LEFT);
    thrust::constant_iterator<side> right_side(side::RIGHT);

    auto left_indices = thrust::make_counting_iterator(static_cast<size_type>(0));
    auto right_indices = thrust::make_counting_iterator(static_cast<size_type>(0));

    auto left_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
    auto right_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

    auto left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
    auto right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side + right_size, right_indices + right_size));

    rmm::device_vector<index_type> merged_indices(total_size);

    auto lhs_device_view = table_device_view::create(left_table, stream);
    auto rhs_device_view = table_device_view::create(right_table, stream);

    rmm::device_vector<order> d_column_order(column_order);

    auto exec_pol = rmm::exec_policy(stream);
    if (nullable){
      rmm::device_vector<null_order> d_null_precedence(null_precedence);

      auto ineq_op =
        experimental::detail::row_lexicographic_tagged_comparator<true>(*lhs_device_view,
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
      auto ineq_op =
        experimental::detail::row_lexicographic_tagged_comparator<false>(*lhs_device_view,
                                                                               *rhs_device_view,
                                                                               d_column_order.data().get());
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

} // anonym. namespace

namespace cudf {
namespace experimental {
namespace detail {

//generate merged column
//given row order of merged tables
//(ordered according to indices of key_cols)
//and the 2 columns to merge
//
struct column_merger
{
  using index_vector = rmm::device_vector<index_type>;
  explicit column_merger(index_vector const& row_order,
                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                        cudaStream_t stream = nullptr):
    dv_row_order_(row_order),
    mr_(mr),
    stream_(stream)
  {
  }

  // column merger operator;
  //
  template<typename Element>//required: column type
  std::enable_if_t<cudf::is_fixed_width<Element>(),
                   std::unique_ptr<cudf::column>>
  operator()(cudf::column_view const& lcol, cudf::column_view const& rcol) const
  {
    auto lsz = lcol.size();
    auto merged_size = lsz + rcol.size();
    auto type = lcol.type();

    std::unique_ptr<cudf::column> p_merged_col{nullptr};
    if (lcol.has_nulls())
      p_merged_col = cudf::experimental::allocate_like(lcol, merged_size);
    else
      p_merged_col = cudf::experimental::allocate_like(rcol, merged_size);

    //"gather" data from lcol, rcol according to dv_row_order_ "map"
    //(directly calling gather() won't work because
    // lcol, rcol indices overlap!)
    //
    cudf::mutable_column_view merged_view = p_merged_col->mutable_view();

    //initialize null_mask to all valid:
    //
    //Note: this initialization in conjunction with _conditionally_
    //calling materialze_bitmask() below covers the case
    //materialize_merged_bitmask_kernel<false, false>()
    //which won't be called anymore (because of the _condition_ below)
    //
    cudf::set_null_mask(merged_view.null_mask(),
                        merged_view.size(),
                        true,
                        stream_);

    //set the null count:
    //
    p_merged_col->set_null_count(lcol.null_count() + rcol.null_count());

    //to resolve view.data()'s types use: Element
    //
    Element const* p_d_lcol = lcol.data<Element>();
    Element const* p_d_rcol = rcol.data<Element>();

    auto exe_pol = rmm::exec_policy(stream_);

    //capture lcol, rcol
    //and "gather" into merged_view.data()[indx_merged]
    //from lcol or rcol, depending on side;
    //
    thrust::transform(exe_pol->on(stream_),
                      dv_row_order_.begin(), dv_row_order_.end(),
                      merged_view.begin<Element>(),
                      [p_d_lcol, p_d_rcol] __device__ (index_type const& index_pair){
                       auto side = thrust::get<0>(index_pair);
                       auto index = thrust::get<1>(index_pair);

                       Element val = (side == side::LEFT ? p_d_lcol[index] : p_d_rcol[index]);
                       return val;
                      }
                     );

    //CAVEAT: conditional call below is erroneous without
    //set_null_mask() call (see TODO above):
    //
    if (lcol.has_nulls() || rcol.has_nulls()) {
      //resolve null mask:
      //
      materialize_bitmask(lcol,rcol, merged_view, dv_row_order_.data().get(), stream_);
    }

    return p_merged_col;
  }

  //specialization for strings
  //
  template<typename Element>//required: column type
  std::enable_if_t<not cudf::is_fixed_width<Element>(),
                   std::unique_ptr<cudf::column>>
  operator()(cudf::column_view const& lcol, cudf::column_view const& rcol) const
  {

    auto column = strings::detail::merge<index_type>( strings_column_view(lcol),
                                                      strings_column_view(rcol),
                                                      dv_row_order_.begin(),
                                                      dv_row_order_.end(),
                                                      mr_,
                                                      stream_);

    if (lcol.has_nulls() || rcol.has_nulls())
      {
        auto merged_view = column->mutable_view();
        materialize_bitmask(lcol,
                            rcol,
                            merged_view,
                            dv_row_order_.data().get(),
                            stream_);
      }
    return column;
  }

private:
  index_vector const& dv_row_order_;
  rmm::mr::device_memory_resource* mr_;
  cudaStream_t stream_;
};


std::unique_ptr<cudf::experimental::table> merge(cudf::table_view const& left_table,
                                                 cudf::table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& column_order,
                                                 std::vector<cudf::null_order> const& null_precedence,
                                                 rmm::mr::device_memory_resource* mr,
                                                 cudaStream_t stream = 0) {
    auto n_cols = left_table.num_columns();
    CUDF_EXPECTS( n_cols == right_table.num_columns(), "Mismatched number of columns");
    if (left_table.num_columns() == 0) {
      return cudf::experimental::empty_like(left_table);
    }

    CUDF_EXPECTS(cudf::have_same_types(left_table, right_table), "Mismatched column types");

    auto keys_sz = key_cols.size();
    CUDF_EXPECTS( keys_sz > 0, "Empty key_cols");
    CUDF_EXPECTS( keys_sz <= static_cast<size_t>(left_table.num_columns()), "Too many values in key_cols");

    CUDF_EXPECTS(keys_sz == column_order.size(), "Mismatched size between key_cols and column_order");

    if (not column_order.empty())
      {
        CUDF_EXPECTS(column_order.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in column_order");
      }

    //collect index columns for lhs, rhs, resp.
    //
    cudf::table_view index_left_view{left_table.select(key_cols)};
    cudf::table_view index_right_view{right_table.select(key_cols)};
    bool nullable = cudf::has_nulls(index_left_view) || cudf::has_nulls(index_right_view);

    //extract merged row order according to indices:
    //
    rmm::device_vector<index_type>
      merged_indices = generate_merged_indices(index_left_view,
                                               index_right_view,
                                               column_order,
                                               null_precedence,
                                               nullable);

    //create merged table:
    //
    std::vector<std::unique_ptr<column>> v_merged_cols;
    v_merged_cols.reserve(n_cols);

    column_merger merger{merged_indices, mr, stream};

    for(auto i=0;i<n_cols;++i)
      {
        const auto& left_col = left_table.column(i);
        const auto& right_col= right_table.column(i);

        auto merged = cudf::experimental::type_dispatcher(left_col.type(),
                                                          merger,
                                                          left_col,
                                                          right_col);
        v_merged_cols.emplace_back(std::move(merged));
      }

    return std::make_unique<cudf::experimental::table>(std::move(v_merged_cols));
}

}  // namespace detail

std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                                 table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& column_order,
                                                 std::vector<cudf::null_order> const& null_precedence,
                                                 rmm::mr::device_memory_resource* mr){
  return detail::merge(left_table, right_table, key_cols, column_order, null_precedence, mr);
}

}  // namespace experimental
}  // namespace cudf
