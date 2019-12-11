
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
#include <nvstrings/NVCategory.h>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/table.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include "bitmask/legacy/bit_mask.cuh"
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/legacy/cuda_utils.hpp"

namespace {

/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };

using bit_mask::bit_mask_t;
using index_type = thrust::tuple<side, cudf::size_type>; // `thrust::get<0>` indicates left/right side, `thrust::get<1>` indicates the row index

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
__global__ void materialize_merged_bitmask_kernel(
    bit_mask_t const* const __restrict__ source_left_mask,
    bit_mask_t const* const __restrict__ source_right_mask,
    bit_mask_t* const destination_mask,
    cudf::size_type const num_destination_rows,
    index_type const* const __restrict__ merged_indices) {

  cudf::size_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    index_type const& merged_idx = merged_indices[destination_row];
    side const src_side = thrust::get<0>(merged_idx);
    cudf::size_type const src_row  = thrust::get<1>(merged_idx);
    bool const from_left{src_side == side::LEFT};
    bool source_bit_is_valid{true};
    if (left_have_valids && from_left) {
        source_bit_is_valid = bit_mask::is_valid(source_left_mask, src_row);
    }
    else if (right_have_valids && !from_left) {
        source_bit_is_valid = bit_mask::is_valid(source_right_mask, src_row);
    }
    
    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    bit_mask_t const result_mask{
        __ballot_sync(active_threads, source_bit_is_valid)};

    cudf::size_type const output_element = cudf::util::detail::bit_container_index<bit_mask_t, cudf::size_type>(destination_row);
    
    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask[output_element] = result_mask;
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
        __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

void materialize_bitmask(gdf_column const* left_col,
                        gdf_column const* right_col,
                        gdf_column* out_col,
                        index_type const* merged_indices,
                        cudaStream_t stream) {
    constexpr cudf::size_type BLOCK_SIZE{256};
    cudf::util::cuda::grid_config_1d grid_config {out_col->size, BLOCK_SIZE };

    bit_mask_t* left_valid = reinterpret_cast<bit_mask_t*>(left_col->valid);
    bit_mask_t* right_valid = reinterpret_cast<bit_mask_t*>(right_col->valid);
    bit_mask_t* out_valid = reinterpret_cast<bit_mask_t*>(out_col->valid);
    if (left_valid) {
        if (right_valid) {
            materialize_merged_bitmask_kernel<true, true>
            <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
            (left_valid, right_valid, out_valid, out_col->size, merged_indices);
        } else {
            materialize_merged_bitmask_kernel<true, false>
            <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
            (left_valid, right_valid, out_valid, out_col->size, merged_indices);
        }
    } else {
        if (right_valid) {
            materialize_merged_bitmask_kernel<false, true>
            <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
            (left_valid, right_valid, out_valid, out_col->size, merged_indices);
        } else {
            materialize_merged_bitmask_kernel<false, false>
            <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
            (left_valid, right_valid, out_valid, out_col->size, merged_indices);
        }
    }

    CHECK_CUDA(stream);
}

rmm::device_vector<index_type>
generate_merged_indices(device_table const& left_table,
                        device_table const& right_table,
                        rmm::device_vector<int8_t> const& asc_desc,
                        bool nulls_are_smallest,
                        cudaStream_t stream) {

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
    bool nullable = left_table.has_nulls() || right_table.has_nulls();
    if (nullable){
        auto ineq_op = row_inequality_comparator<true>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
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
        auto ineq_op = row_inequality_comparator<false>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
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

    CHECK_CUDA(stream);

    return merged_indices;
}

} // namespace

namespace cudf {
namespace detail {

table merge(table const& left_table,
            table const& right_table,
            std::vector<cudf::size_type> const& key_cols,
            std::vector<order_by_type> const& asc_desc,
            bool nulls_are_smallest,
            cudaStream_t stream = 0) {
    CUDF_EXPECTS(left_table.num_columns() == right_table.num_columns(), "Mismatched number of columns");
    if (left_table.num_columns() == 0) {
        return cudf::empty_like(left_table);
    }
    
    std::vector<gdf_dtype> left_table_dtypes = cudf::column_dtypes(left_table);
    std::vector<gdf_dtype> right_table_dtypes = cudf::column_dtypes(right_table);
    CUDF_EXPECTS(std::equal(left_table_dtypes.cbegin(), left_table_dtypes.cend(), right_table_dtypes.cbegin(), right_table_dtypes.cend()), "Mismatched column dtypes");
    CUDF_EXPECTS(key_cols.size() > 0, "Empty key_cols");
    CUDF_EXPECTS(key_cols.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in key_cols");
    CUDF_EXPECTS(asc_desc.size() > 0, "Empty asc_desc");
    CUDF_EXPECTS(asc_desc.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in asc_desc");
    CUDF_EXPECTS(key_cols.size() == asc_desc.size(), "Mismatched size between key_cols and asc_desc");


    auto gdf_col_deleter = [](gdf_column *col) {
        gdf_column_free(col);
        delete col;
    };
    using gdf_col_ptr = typename std::unique_ptr<gdf_column, decltype(gdf_col_deleter)>;
    std::vector<gdf_col_ptr> temp_columns_to_free;
    std::vector<gdf_column*> left_cols_sync(const_cast<gdf_column**>(left_table.begin()), const_cast<gdf_column**>(left_table.end()));
    std::vector<gdf_column*> right_cols_sync(const_cast<gdf_column**>(right_table.begin()), const_cast<gdf_column**>(right_table.end()));
    for (cudf::size_type i = 0; i < left_table.num_columns(); i++) {
        gdf_column * left_col = const_cast<gdf_column*>(left_table.get_column(i));
        gdf_column * right_col = const_cast<gdf_column*>(right_table.get_column(i));
        
        if (left_col->dtype != GDF_STRING_CATEGORY){
            continue;
        }

        // If the inputs are nvcategory we need to make the dictionaries comparable

        temp_columns_to_free.push_back(gdf_col_ptr(new gdf_column{}, gdf_col_deleter));
        gdf_column * new_left_column_ptr = temp_columns_to_free.back().get();
        temp_columns_to_free.push_back(gdf_col_ptr(new gdf_column{}, gdf_col_deleter));
        gdf_column * new_right_column_ptr = temp_columns_to_free.back().get();

        *new_left_column_ptr = allocate_like(*left_col, RETAIN, stream);
        if (new_left_column_ptr->valid) {
            CUDA_TRY( cudaMemcpyAsync(new_left_column_ptr->valid, left_col->valid, sizeof(cudf::valid_type)*gdf_num_bitmask_elements(left_col->size), cudaMemcpyDefault, stream) );
            new_left_column_ptr->null_count = left_col->null_count;
        }
        
        *new_right_column_ptr = allocate_like(*right_col, RETAIN, stream);
        if (new_right_column_ptr->valid) {
            CUDA_TRY( cudaMemcpyAsync(new_right_column_ptr->valid, right_col->valid, sizeof(cudf::valid_type)*gdf_num_bitmask_elements(right_col->size), cudaMemcpyDefault, stream) );
            new_right_column_ptr->null_count = right_col->null_count;
        }

        gdf_column * tmp_arr_input[2] = {left_col, right_col};
        gdf_column * tmp_arr_output[2] = {new_left_column_ptr, new_right_column_ptr};
        CUDF_TRY( sync_column_categories(tmp_arr_input, tmp_arr_output, 2) );

        left_cols_sync[i] = new_left_column_ptr;
        right_cols_sync[i] = new_right_column_ptr;
    }

    table left_sync_table(left_cols_sync);
    table right_sync_table(right_cols_sync);

    std::vector<gdf_column*> left_key_cols_vect(key_cols.size());
    std::transform(key_cols.cbegin(), key_cols.cend(), left_key_cols_vect.begin(),
                  [&] (cudf::size_type const index) { return left_sync_table.get_column(index); });
    
    std::vector<gdf_column*> right_key_cols_vect(key_cols.size());
    std::transform(key_cols.cbegin(), key_cols.cend(), right_key_cols_vect.begin(),
                  [&] (cudf::size_type const index) { return right_sync_table.get_column(index); });

    auto left_key_table = device_table::create(left_key_cols_vect.size(), left_key_cols_vect.data());
    auto right_key_table = device_table::create(right_key_cols_vect.size(), right_key_cols_vect.data());
    rmm::device_vector<int8_t> asc_desc_d(asc_desc);

    rmm::device_vector<index_type> merged_indices = generate_merged_indices(*left_key_table, *right_key_table, asc_desc_d, nulls_are_smallest, stream);

    // Allocate output table
    bool nullable = has_nulls(left_sync_table) || has_nulls(right_sync_table);
    table destination_table(left_sync_table.num_rows() + right_sync_table.num_rows(),
                            column_dtypes(left_sync_table),
                            column_dtype_infos(left_sync_table),
                            nullable, false, stream);
    for (cudf::size_type i = 0; i < destination_table.num_columns(); i++) {
        gdf_column const* left_col = left_sync_table.get_column(i);
        gdf_column * out_col = destination_table.get_column(i);
        
        if (left_col->dtype != GDF_STRING_CATEGORY){
            continue;
        }

        NVCategory * category = static_cast<NVCategory*>(left_col->dtype_info.category);
        out_col->dtype_info.category = category->copy();
    }
    
    // Materialize
    auto left_device_table_ptr = device_table::create(left_sync_table, stream);
    auto right_device_table_ptr = device_table::create(right_sync_table, stream);
    auto output_device_table_ptr = device_table::create(destination_table, stream);
    auto& left_device_table = *left_device_table_ptr;
    auto& right_device_table = *right_device_table_ptr;
    auto& output_device_table = *output_device_table_ptr;

    auto index_start_it = thrust::make_zip_iterator(thrust::make_tuple(
                                                    thrust::make_counting_iterator(static_cast<cudf::size_type>(0)), 
                                                    merged_indices.begin()));
    auto index_end_it = thrust::make_zip_iterator(thrust::make_tuple(
                                                thrust::make_counting_iterator(static_cast<cudf::size_type>(merged_indices.size())),
                                                merged_indices.end()));

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                    index_start_it,
                    index_end_it,
                    [=] __device__ (auto const & idx_tuple){
                        cudf::size_type dest_row = thrust::get<0>(idx_tuple);
                        index_type merged_idx = thrust::get<1>(idx_tuple);
                        side src_side = thrust::get<0>(merged_idx);
                        cudf::size_type src_row  = thrust::get<1>(merged_idx);
                        device_table const & src_device_table = src_side == side::LEFT ? left_device_table : right_device_table;
                        copy_row<false>(output_device_table, dest_row, src_device_table, src_row);
                    });
    
    CHECK_CUDA(stream);

    if (nullable) {
        for (cudf::size_type i = 0; i < destination_table.num_columns(); i++) {
            gdf_column const* left_col = left_sync_table.get_column(i);
            gdf_column const* right_col = right_sync_table.get_column(i);
            gdf_column* out_col = destination_table.get_column(i);
            
            materialize_bitmask(left_col, right_col, out_col, merged_indices.data().get(), stream);
            
            out_col->null_count = left_col->null_count + right_col->null_count;
        }
    }

    return destination_table;
}

}  // namespace detail

table merge(table const& left_table,
            table const& right_table,
            std::vector<cudf::size_type> const& key_cols,
            std::vector<order_by_type> const& asc_desc,
            bool nulls_are_smallest) {
    return detail::merge(left_table, right_table, key_cols, asc_desc, nulls_are_smallest);
}

}  // namespace cudf
