
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
#include <cudf/copying.hpp>
#include <cudf/table.hpp>
#include "table/device_table.cuh"
#include "table/device_table_row_operators.cuh"
#include "bitmask/bit_mask.cuh"
#include "string/nvcategory_util.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cuda_utils.hpp"

namespace {

using bit_mask::bit_mask_t;

/**
 * @brief Source table identifier to copy data from.
 */
enum side_value { LEFT_TABLE = 0, RIGHT_TABLE };

/**
 * @brief Merges the bits of two validity bitmasks.
 *
 * Merges the bits from two source bitmask into the destination bitmask
 * according to `merged_table_indices` and `merged_row_indices` maps such
 * that bit `i` in `destination_mask` will be equal to bit 
 * `merged_row_indices[i]` from `source_left_mask` if `merged_table_indices[i]`
 * equals `LEFT_TABLE`; otherwise, from `source_right_mask`.
 *
 * `source_left_mask`, `source_right_mask` and `destination_mask` must not
 * overlap.
 *
 * @param[in] source_left_mask The left mask whose bits will be merged
 * @param[in] source_right_mask The right mask whose bits will be merged
 * @param[out] destination_mask The output mask after merging the left and right masks
 * @param[in] num_destination_rows The number of bits in the destination_mask
 * @param[in] merged_table_indices The map that indicates from which input mask bits
 * will be copied to the output. Length must be equal to `num_destination_rows`
 * @param[in] merged_row_indices The map that indicates which bit from the input
 * mask (indicated by `merged_table_indices`) will be copied to the output. Length
 * must be equal to `num_destination_rows`
 */
__global__ void materialize_merged_bitmask_kernel(
    bit_mask_t const* const __restrict__ source_left_mask,
    bit_mask_t const* const __restrict__ source_right_mask,
    bit_mask_t* const destination_mask,
    gdf_size_type const num_destination_rows,
    gdf_size_type const* const __restrict__ merged_table_indices,
    gdf_size_type const* const __restrict__ merged_row_indices) {

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    bit_mask_t const* source_mask = merged_table_indices[destination_row] == LEFT_TABLE ? source_left_mask : source_right_mask;
    bool const source_bit_is_valid{
        source_mask ? bit_mask::is_valid(source_mask, merged_row_indices[destination_row]) : true};
    
    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    bit_mask_t const result_mask{
        __ballot_sync(active_threads, source_bit_is_valid)};

    gdf_index_type const output_element = cudf::util::detail::bit_container_index<bit_mask_t, gdf_index_type>(destination_row);
    
    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask[output_element] = result_mask;
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
        __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

void materialize_bitmask(gdf_column const* leftCol,
                        gdf_column const* rightCol,
                        gdf_column* outCol,
                        gdf_size_type const* tableIndices,
                        gdf_size_type const* rowIndices,
                        cudaStream_t stream) {
    constexpr gdf_size_type BLOCK_SIZE{256};
    cudf::util::cuda::grid_config_1d grid_config {outCol->size, BLOCK_SIZE };
    // TODO: cant use just one template flag for valids, so need to test for null explicitly
    // or make the kernel receive just one input col and call it twice one for each input column
    materialize_merged_bitmask_kernel
    <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
    (reinterpret_cast<bit_mask_t*>(leftCol->valid),
     reinterpret_cast<bit_mask_t*>(rightCol->valid),
     reinterpret_cast<bit_mask_t*>(outCol->valid),
     outCol->size,
     tableIndices,
     rowIndices);

    CHECK_STREAM(stream);
}

std::pair<rmm::device_vector<gdf_size_type>, rmm::device_vector<gdf_size_type>>
generate_merged_indices(device_table const& left_table,
                        device_table const& right_table,
                        rmm::device_vector<int8_t> const& asc_desc,
                        bool nulls_are_smallest,
                        cudaStream_t stream) {

    const gdf_size_type left_size  = left_table.num_rows();
    const gdf_size_type right_size = right_table.num_rows();
    const gdf_size_type total_size = left_size + right_size;

    auto left_side = thrust::make_constant_iterator(static_cast<gdf_size_type>(LEFT_TABLE));
    auto right_side = thrust::make_constant_iterator(static_cast<gdf_size_type>(RIGHT_TABLE));

    auto left_indices = thrust::make_counting_iterator(static_cast<gdf_size_type>(0));
    auto right_indices = thrust::make_counting_iterator(static_cast<gdf_size_type>(0));

    auto left_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
    auto right_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

    auto left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
    auto right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side + right_size, right_indices + right_size));

    rmm::device_vector<gdf_size_type> outputTableIndices(total_size);
    rmm::device_vector<gdf_size_type> outputRowIndices(total_size);
    auto output_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(outputTableIndices.begin(), outputRowIndices.begin()));

    bool nullable = left_table.has_nulls() || right_table.has_nulls();
    if (nullable){
        auto ineq_op = row_inequality_comparator<true>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & rightTuple,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & leftTuple) {
                        return ineq_op(thrust::get<1>(rightTuple), thrust::get<1>(leftTuple));
                    });			        
    } else {
        auto ineq_op = row_inequality_comparator<false>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & rightTuple,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & leftTuple) {
                        return ineq_op(thrust::get<1>(rightTuple), thrust::get<1>(leftTuple));
                    });					        
    }

    CHECK_STREAM(stream);

    return std::make_pair(outputTableIndices, outputRowIndices);
}

} // namespace

namespace cudf {
namespace detail {

table sorted_merge(table const& left_table,
                   table const& right_table,
                   std::vector<gdf_size_type> const& key_cols,
                   std::vector<order_by_type> const& asc_desc,
                   bool nulls_are_smallest,
                   cudaStream_t stream = 0) {
    CUDF_EXPECTS(left_table.num_columns() == right_table.num_columns(), "Mismatched number of columns");
    if (left_table.num_columns() == 0) {
        return cudf::empty_like(left_table);
    }
    
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
    for (gdf_size_type i = 0; i < left_table.num_columns(); i++) {
        gdf_column * leftCol = const_cast<gdf_column*>(left_table.get_column(i));
        gdf_column * rightCol = const_cast<gdf_column*>(right_table.get_column(i));
        
        if (leftCol->dtype != GDF_STRING_CATEGORY){
            continue;
        }

        // If the inputs are nvcategory we need to make the dictionaries comparable

        temp_columns_to_free.push_back(gdf_col_ptr(new gdf_column{}, gdf_col_deleter));
        gdf_column * new_left_column_ptr = temp_columns_to_free.back().get();
        temp_columns_to_free.push_back(gdf_col_ptr(new gdf_column{}, gdf_col_deleter));
        gdf_column * new_right_column_ptr = temp_columns_to_free.back().get();

        *new_left_column_ptr = allocate_like(*leftCol, true, stream);
        if (new_left_column_ptr->valid) {
            CUDA_TRY( cudaMemcpyAsync(new_left_column_ptr->valid, leftCol->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(leftCol->size), cudaMemcpyDefault, stream) );
            new_left_column_ptr->null_count = leftCol->null_count;
        }
        
        *new_right_column_ptr = allocate_like(*rightCol, true, stream);
        if (new_right_column_ptr->valid) {
            CUDA_TRY( cudaMemcpyAsync(new_right_column_ptr->valid, rightCol->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(rightCol->size), cudaMemcpyDefault, stream) );
            new_right_column_ptr->null_count = rightCol->null_count;
        }

        gdf_column * tmp_arr_input[2] = {leftCol, rightCol};
        gdf_column * tmp_arr_output[2] = {new_left_column_ptr, new_right_column_ptr};
        CUDF_TRY( sync_column_categories(tmp_arr_input, tmp_arr_output, 2) );

        left_cols_sync[i] = new_left_column_ptr;
        right_cols_sync[i] = new_right_column_ptr;
    }

    table left_sync_table(left_cols_sync.data(), left_cols_sync.size());
    table right_sync_table(right_cols_sync.data(), right_cols_sync.size());

    std::vector<gdf_column*> left_key_cols_vect(key_cols.size());
    std::transform(key_cols.cbegin(), key_cols.cend(), left_key_cols_vect.begin(),
                  [&] (gdf_index_type const index) { return left_sync_table.get_column(index); });
    
    std::vector<gdf_column*> right_key_cols_vect(key_cols.size());
    std::transform(key_cols.cbegin(), key_cols.cend(), right_key_cols_vect.begin(),
                  [&] (gdf_index_type const index) { return right_sync_table.get_column(index); });

    auto leftKeyTable = device_table::create(left_key_cols_vect.size(), left_key_cols_vect.data());
    auto rightKeyTable = device_table::create(right_key_cols_vect.size(), right_key_cols_vect.data());
    rmm::device_vector<int8_t> asc_desc_d(asc_desc);

    rmm::device_vector<gdf_size_type> mergedTableIndices;
    rmm::device_vector<gdf_size_type> mergedRowIndices;
    std::tie(mergedTableIndices, mergedRowIndices) = generate_merged_indices(*leftKeyTable, *rightKeyTable, asc_desc_d, nulls_are_smallest, stream);

    // Allocate output table
    bool nullable = has_nulls(left_sync_table) || has_nulls(right_sync_table);
    table destination_table(left_sync_table.num_rows() + right_sync_table.num_rows(), column_dtypes(left_sync_table), nullable, false, stream);
    for (gdf_size_type i = 0; i < destination_table.num_columns(); i++) {
        gdf_column const* leftCol = left_sync_table.get_column(i);
        gdf_column * outCol = destination_table.get_column(i);
        
        if (leftCol->dtype != GDF_STRING_CATEGORY){
            continue;
        }

        NVCategory * category = static_cast<NVCategory*>(leftCol->dtype_info.category);
        outCol->dtype_info.category = category->copy();
    }
    
    // Materialize
    auto leftInputDeviceTablePtr = device_table::create(left_sync_table, stream);
    auto rightInputDeviceTablePtr = device_table::create(right_sync_table, stream);
    auto outputDeviceTablePtr = device_table::create(destination_table, stream);
    auto& leftInputDeviceTable = *leftInputDeviceTablePtr;
    auto& rightInputDeviceTable = *rightInputDeviceTablePtr;
    auto& outputDeviceTable = *outputDeviceTablePtr;

    auto index_start_it = thrust::make_zip_iterator(thrust::make_tuple(
                                                    thrust::make_counting_iterator(static_cast<gdf_size_type>(0)), 
                                                    mergedTableIndices.begin(),
                                                    mergedRowIndices.begin()));
    auto index_end_it = thrust::make_zip_iterator(thrust::make_tuple(
                                                thrust::make_counting_iterator(static_cast<gdf_size_type>(mergedTableIndices.size())),
                                                mergedTableIndices.end(),
                                                mergedRowIndices.end()));

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                    index_start_it,
                    index_end_it,
                    [=] __device__ (auto const & idx_tuple){
                        device_table const & sourceDeviceTable = thrust::get<1>(idx_tuple) == LEFT_TABLE ? leftInputDeviceTable : rightInputDeviceTable;
                        copy_row<false>(outputDeviceTable, thrust::get<0>(idx_tuple), sourceDeviceTable, thrust::get<2>(idx_tuple));
                    });
    
    CHECK_STREAM(0);

    if (nullable) {
        for (gdf_size_type i = 0; i < destination_table.num_columns(); i++) {
            gdf_column const* leftCol = left_sync_table.get_column(i);
            gdf_column const* rightCol = right_sync_table.get_column(i);
            gdf_column* outCol = destination_table.get_column(i);
            
            materialize_bitmask(leftCol, rightCol, outCol, mergedTableIndices.data().get(), mergedRowIndices.data().get(), stream);
            
            outCol->null_count = leftCol->null_count + rightCol->null_count;
        }
    }

    return destination_table;
}

}  // namespace detail

table sorted_merge(table const& left_table,
                   table const& right_table,
                   std::vector<gdf_size_type> const& key_cols,
                   std::vector<order_by_type> const& asc_desc,
                   bool nulls_are_smallest) {
    return detail::sorted_merge(left_table, right_table, key_cols, asc_desc, nulls_are_smallest);
}

}  // namespace cudf
