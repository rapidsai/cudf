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

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <type_traits>
#include <nvstrings/NVCategory.h>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cuda_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/null_mask.hpp>

namespace {

/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };
  
using index_type = thrust::tuple<side, cudf::size_type>; // `thrust::get<0>` indicates left/right side, `thrust::get<1>` indicates the row index

using BitMaskT = cudf::bitmask_type;// some confusion around this type (11/11/19)...

/**
 * @brief Merges the bits of two validity bitmasks.
 *
 * Merges the bits from two source bitmask into the destination bitmask
 * according to `merged_indices` map such that bit `i` in `destination_mask`
 * will be equal to bit `thrust::get<1>(merged_indices[i])` from `source_left_mask`
 * if `thrust::get<0>(merged_indices[i])` equals `side::LEFT`; otherwise,
 * from `source_right_mask`.
 *
 * `source_left_mask`, `source_right_mask` and `destination_mask` must not
 * overlap.
 *
 * @tparam left_have_valids Indicates whether source_left_mask is null
 * @tparam right_have_valids Indicates whether source_right_mask is null
 * @param[in] source_left_mask The left mask whose bits will be merged
 * @param[in] source_right_mask The right mask whose bits will be merged
 * @param[out] destination_mask The output mask after merging the left and right masks
 * @param[in] num_destination_rows The number of bits in the destination_mask
 * @param[in] merged_indices The map that indicates from which input mask and which bit
 * will be copied to the output. Length must be equal to `num_destination_rows`
 */
template <bool left_have_valids, bool right_have_valids>
__global__ void materialize_merged_bitmask_kernel(cudf::column_device_view left_dcol,
                                                  cudf::column_device_view right_dcol,
                                                  cudf::mutable_column_device_view out_dcol,
                                                  cudf::size_type const num_destination_rows,
                                                  index_type const* const __restrict__ merged_indices) {
  cudf::size_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  BitMaskT const* const __restrict__ source_left_mask = left_dcol.null_mask();
  BitMaskT const* const __restrict__ source_right_mask= right_dcol.null_mask();
  BitMaskT* const destination_mask = out_dcol.null_mask();
  
  auto active_threads =
    __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    index_type const& merged_idx = merged_indices[destination_row];
    side const src_side = thrust::get<0>(merged_idx);
    cudf::size_type const src_row  = thrust::get<1>(merged_idx);
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
    BitMaskT const result_mask{
      __ballot_sync(active_threads, source_bit_is_valid)};

    cudf::size_type const output_element = cudf::word_index(destination_row);

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask[output_element] = result_mask;
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
      __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

void materialize_bitmask(cudf::column_view const& left_col,
                         cudf::column_view const& right_col,
                         cudf::mutable_column_view& out_col,
                         index_type const* merged_indices,
                         cudaStream_t stream) {
  constexpr cudf::size_type BLOCK_SIZE{256};
  cudf::util::cuda::grid_config_1d grid_config {out_col.size(), BLOCK_SIZE };

  auto p_left_dcol  = cudf::column_device_view::create(left_col);
  auto p_right_dcol = cudf::column_device_view::create(right_col);
  auto p_out_dcol   = cudf::mutable_column_device_view::create(out_col);

  auto left_valid  = *p_left_dcol;
  auto right_valid = *p_right_dcol;
  auto out_valid   = *p_out_dcol;
  
  if (p_left_dcol) {
    if (p_right_dcol) {
      materialize_merged_bitmask_kernel<true, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      materialize_merged_bitmask_kernel<true, false>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    }
  } else {
    if (p_right_dcol) {
      materialize_merged_bitmask_kernel<false, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    } else {
      materialize_merged_bitmask_kernel<false, false>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
        (left_valid, right_valid, out_valid, out_col.size(), merged_indices);
    }
  }

  CHECK_STREAM(stream);
}
  
  
  
rmm::device_vector<index_type>
generate_merged_indices(cudf::table_view const& left_table,
                        cudf::table_view const& right_table,
                        std::vector<cudf::order> const& asc_desc,
                        std::vector<cudf::null_order> const& null_precedence,
                        bool nullable = true,
                        cudaStream_t stream = nullptr) {

    const cudf::size_type left_size  = left_table.num_rows();
    const cudf::size_type right_size = right_table.num_rows();
    const cudf::size_type total_size = left_size + right_size;

    thrust::constant_iterator<side> left_side(side::LEFT);
    thrust::constant_iterator<side> right_side(side::RIGHT);

    auto left_indices = thrust::make_counting_iterator(static_cast<cudf::size_type>(0));
    auto right_indices = thrust::make_counting_iterator(static_cast<cudf::size_type>(0));

    auto left_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
    auto right_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

    auto left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
    auto right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side + right_size, right_indices + right_size));

    rmm::device_vector<index_type> merged_indices(total_size);
    
    auto lhs_tdv = cudf::table_device_view::create(left_table, stream);
    auto rhs_tdv = cudf::table_device_view::create(right_table, stream);

    rmm::device_vector<cudf::order> d_column_order(asc_desc); 
    
    auto exec_pol = rmm::exec_policy(stream);
    if (nullable){
      rmm::device_vector<cudf::null_order> d_null_precedence(null_precedence);
      
      auto ineq_op =
        cudf::experimental::row_lexicographic_comparator<true>(*lhs_tdv,
                                                               *rhs_tdv,
                                                               d_column_order.data().get(),
                                                               d_null_precedence.data().get());
      
        thrust::merge(exec_pol->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    merged_indices.begin(),
                    [=] __device__ (thrust::tuple<side, cudf::size_type> const & right_tuple,
                                    thrust::tuple<side, cudf::size_type> const & left_tuple) {
                        return ineq_op(thrust::get<1>(right_tuple), thrust::get<1>(left_tuple));
                    });			        
    } else {
      auto ineq_op =
        cudf::experimental::row_lexicographic_comparator<false>(*lhs_tdv,
                                                                *rhs_tdv,
                                                                d_column_order.data().get()); 
        thrust::merge(exec_pol->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    merged_indices.begin(),
                    [=] __device__ (thrust::tuple<side, cudf::size_type> const & right_tuple,
                                    thrust::tuple<side, cudf::size_type> const & left_tuple) {
                        return ineq_op(thrust::get<1>(right_tuple), thrust::get<1>(left_tuple));
                    });					        
    }

    CHECK_STREAM(stream);

    return merged_indices;
}

} // namespace

namespace cudf {
namespace experimental { 
namespace detail {

//work-in-progress:
//
//generate merged column
//given row order of merged tables
//(ordered according to indices of key_cols)
//and the 2 columns to merge
//
template<typename VectorI>
struct ColumnMerger
{
  //error: class "cudf::string_view" has no member "type"
  //using StringT = typename cudf::experimental::id_to_type<cudf::STRING>::type;
  
  explicit ColumnMerger(VectorI const& row_order,
                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                        cudaStream_t stream = nullptr):
    dv_row_order_(row_order),
    mr_(mr),
    stream_(stream)
  {
  }

  //raison d'etre for enable_if:
  //need separate versions for primary `Element` != strings
  
  // type_dispatcher() _can_ dispatch host functors:
  //
  template<typename ElemenT>//required: column type
  //std::enable_if_t<!(std::is_same<ElemenT,StringT>::value),
  std::unique_ptr<cudf::column> //>
  operator()(cudf::column_view const& lcol, cudf::column_view const& rcol) const
  {   
    auto merged_sz = lcol.size() + rcol.size();
    auto type = lcol.type();
    
    rmm::device_buffer data{merged_sz * cudf::size_of(type),
        stream_,
        mr_};
    
    rmm::device_buffer mask = cudf::create_null_mask(merged_sz,
                                                     ALL_VALID,
                                                     stream_,
                                                     mr_);

    //state_null_count(state, merged_size);//cudf::mask_state state{ALL_VALID};
    
    std::unique_ptr<cudf::column> p_merged_col{
      new cudf::column{type, merged_sz, data, mask}
    };

    //OR: instead cnstr a merged column_view, then a column out of it?
    //
    return p_merged_col;//for now...
  }

  //specialization for string...?
  //or should use `cudf::string_view` instead?
  //
  // template<typename ElemenT>//required: column type
  // std::enable_if_t<std::is_same<ElemenT, StringT>::value, std::unique_ptr<cudf::column>>
  // operator()(cudf::column_view const& lcol, cudf::column_view const& rcol) const
  // {
  //   return nullptr;//<-TODO
  // }

private:
  VectorI const& dv_row_order_;
  rmm::mr::device_memory_resource* mr_;
  cudaStream_t stream_;
  
  //see `class element_relational_comparator` in `cpp/include/cudf/table/row_operators.cuh` as a model;
};
  

  std::unique_ptr<cudf::experimental::table> merge(cudf::table_view const& left_table,
                                                   cudf::table_view const& right_table,
                                                   std::vector<cudf::size_type> const& key_cols,
                                                   std::vector<cudf::order> const& asc_desc,
                                                   std::vector<cudf::null_order> const& null_precedence) {
    auto n_cols = left_table.num_columns();
    CUDF_EXPECTS( n_cols == right_table.num_columns(), "Mismatched number of columns");
    if (left_table.num_columns() == 0) {
        return nullptr;
    }

    CUDF_EXPECTS(std::equal(left_table.begin(), left_table.end(), right_table.begin(), right_table.end(),
                            [](cudf::column_view const& lcol, cudf::column_view const& rcol){
                              return (lcol.type() == rcol.type());
                            }),
                 "Mismatched column dtypes");
    
    CUDF_EXPECTS(key_cols.size() > 0, "Empty key_cols");
    CUDF_EXPECTS(key_cols.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in key_cols");
    CUDF_EXPECTS(asc_desc.size() > 0, "Empty asc_desc");
    CUDF_EXPECTS(asc_desc.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in asc_desc");
    CUDF_EXPECTS(key_cols.size() == asc_desc.size(), "Mismatched size between key_cols and asc_desc");    

    //collect index columns for lhs, rhs, resp.
    //
    std::vector<cudf::column_view> left_index_cols;
    std::vector<cudf::column_view> right_index_cols;
    bool nullable{false};
    for(auto&& indx: key_cols)
      {
        const cudf::column_view& left_col = left_table.column(indx);
        const cudf::column_view& right_col= right_table.column(indx);

        //for the purpose of generating merged indices, there's
        //no point looking into _all_ table columns for nulls,
        //just the index ones:
        //
        if( left_col.has_nulls() || right_col.has_nulls() )
          nullable = true;
        
        left_index_cols.push_back(left_col);
        right_index_cols.push_back(right_col);
      }
    cudf::table_view index_left_view{left_index_cols};   //table_view move cnstr. would be nice
    cudf::table_view index_right_view{right_index_cols}; //same...

    //extract merged row order according to indices:
    //
    rmm::device_vector<index_type>
      merged_indices = generate_merged_indices(index_left_view,
                                               index_right_view,
                                               asc_desc,
                                               null_precedence,
                                               nullable);

    //create merged table:
    //
    std::vector<std::unique_ptr<column>> v_merged_cols;
    v_merged_cols.reserve(n_cols);

    static_assert(std::is_same<decltype(v_merged_cols), std::vector<std::unique_ptr<cudf::column>> >::value, "ERROR: unexpected type.");

    ColumnMerger<rmm::device_vector<index_type>> merger{merged_indices};
    
    for(auto i=0;i<n_cols;++i)
      {
        const auto& left_col = left_table.column(i);
        const auto& right_col= right_table.column(i);

        //not clear yet what must be done for STRING:
        //
        //if( left_col.type().id() != STRING )
        //  continue;//?

        auto merged = cudf::experimental::type_dispatcher(left_col.type(),
                                                          merger,
                                                          left_col,
                                                          right_col);
        v_merged_cols.emplace_back(std::move(merged));
      }
    
    return std::unique_ptr<cudf::experimental::table>{new cudf::experimental::table(std::move(v_merged_cols))};
}

}  // namespace detail

std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                                 table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& asc_desc,
                                                 std::vector<cudf::null_order> const& null_precedence){
  return detail::merge(left_table, right_table, key_cols, asc_desc, null_precedence);
}

}  // namespace experimental
}  // namespace cudf
