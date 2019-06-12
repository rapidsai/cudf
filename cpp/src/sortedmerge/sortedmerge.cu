#include <cudf.h>

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
#include <nvstrings/NVCategory.h>

#include "copying.hpp"
#include "table.hpp"
#include "table/device_table.cuh"
#include "table/device_table_row_operators.cuh"
#include "string/nvcategory_util.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cuda_utils.hpp"

namespace {
    
enum side_value { LEFT_TABLE = 0, RIGHT_TABLE };

__global__ void materialize_merged_bitmask_kernel(
    gdf_valid_type const* const __restrict__ source_left_mask,
    gdf_valid_type const* const __restrict__ source_right_mask,
    gdf_valid_type* const destination_mask,
    gdf_size_type const num_destination_rows,
    gdf_size_type const* const __restrict__ merged_table_indices,
    gdf_size_type const* const __restrict__ merged_row_indices) {
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  // Cast bitmask to a type to a 4B type
  // TODO: Update to use new bit_mask_t
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    gdf_valid_type const* source_mask = merged_table_indices[destination_row] == LEFT_TABLE ? source_left_mask : source_right_mask;
    bool const source_bit_is_valid{
        gdf_is_valid(source_mask, merged_row_indices[destination_row])};
    
    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    MaskType const result_mask{
        __ballot_sync(active_threads, source_bit_is_valid)};

    gdf_index_type const output_element = destination_row / BITS_PER_MASK;

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask32[output_element] = result_mask;
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
                        cudaStream_t stream = 0) {
    constexpr gdf_size_type BLOCK_SIZE{256};
    cudf::util::cuda::grid_config_1d grid_config {outCol->size, BLOCK_SIZE };
    
    materialize_merged_bitmask_kernel
    <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream>>>
    (leftCol->valid, rightCol->valid, outCol->valid, outCol->size, tableIndices, rowIndices);

    CHECK_STREAM(stream);
}

std::pair<rmm::device_vector<gdf_size_type>, rmm::device_vector<gdf_size_type>>
generate_merged_indices(device_table const& left_table,
                        device_table const& right_table,
                        rmm::device_vector<int8_t> const& asc_desc,
                        cudaStream_t stream = 0) {

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
        auto ineq_op = row_inequality_comparator<true>(right_table, left_table, false, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & tupleA,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & tupleB) {
                        const gdf_size_type left_row = thrust::get<0>(tupleA) == LEFT_TABLE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        const gdf_size_type right_row = thrust::get<0>(tupleA) == RIGHT_TABLE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        return ineq_op(right_row, left_row);
                    });			        
    } else {
        auto ineq_op = row_inequality_comparator<false>(right_table, left_table, false, asc_desc.data().get()); 
        thrust::merge(rmm::exec_policy(stream)->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & tupleA,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & tupleB) {
                        const gdf_size_type left_row = thrust::get<0>(tupleA) == LEFT_TABLE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        const gdf_size_type right_row = thrust::get<0>(tupleA) == RIGHT_TABLE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        return ineq_op(right_row, left_row);
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
                   std::vector<gdf_size_type> const& sort_by_cols,
                   rmm::device_vector<int8_t> const& asc_desc) {
    CUDF_EXPECTS(left_table.num_columns() == right_table.num_columns(), "Mismatched number of columns");
    CUDF_EXPECTS(left_table.num_columns() > 0, "Empty input tables");
    CUDF_EXPECTS(sort_by_cols.size() > 0, "Empty sort_by_cols");
    CUDF_EXPECTS(sort_by_cols.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in sort_by_cols");
    CUDF_EXPECTS(asc_desc.size() > 0, "Empty asc_desc");
    CUDF_EXPECTS(asc_desc.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in asc_desc");
    CUDF_EXPECTS(sort_by_cols.size() == asc_desc.size(), "Mismatched size between sort_by_cols and asc_desc");

    // Sync GDF_STRING_CATEGORY
    std::vector<gdf_column*> temp_columns_to_free;
    std::vector<gdf_column*> left_cols_sync(const_cast<gdf_column**>(left_table.begin()), const_cast<gdf_column**>(left_table.end()));
    std::vector<gdf_column*> right_cols_sync(const_cast<gdf_column**>(right_table.begin()), const_cast<gdf_column**>(right_table.end()));
    for (gdf_size_type i = 0; i < left_table.num_columns(); i++) {
        gdf_column * leftCol = const_cast<gdf_column*>(left_table.get_column(i));
        gdf_column * rightCol = const_cast<gdf_column*>(right_table.get_column(i));
        
        if (leftCol->dtype != GDF_STRING_CATEGORY){
            continue;
        }

        gdf_column * new_left_column_ptr = new gdf_column{};
        gdf_column * new_right_column_ptr = new gdf_column{};
        temp_columns_to_free.push_back(new_left_column_ptr);
        temp_columns_to_free.push_back(new_right_column_ptr);

        *new_left_column_ptr = allocate_like(*leftCol);
        if (new_left_column_ptr->valid) {
            CUDA_TRY( cudaMemcpy(new_left_column_ptr->valid, leftCol->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(leftCol->size), cudaMemcpyDeviceToDevice) );
            new_left_column_ptr->null_count = leftCol->null_count;
        }
        
        *new_right_column_ptr = allocate_like(*rightCol);
        if (new_right_column_ptr->valid) {
            CUDA_TRY( cudaMemcpy(new_right_column_ptr->valid, rightCol->valid, sizeof(gdf_valid_type)*gdf_num_bitmask_elements(rightCol->size), cudaMemcpyDeviceToDevice) );
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

    std::vector<gdf_column*> left_key_cols_vect(sort_by_cols.size());
    std::transform(sort_by_cols.cbegin(), sort_by_cols.cend(), left_key_cols_vect.begin(),
                  [&] (gdf_index_type const index) { return left_sync_table.get_column(index); });
    
    std::vector<gdf_column*> right_key_cols_vect(sort_by_cols.size());
    std::transform(sort_by_cols.cbegin(), sort_by_cols.cend(), right_key_cols_vect.begin(),
                  [&] (gdf_index_type const index) { return right_sync_table.get_column(index); });

    auto leftKeyTable = device_table::create(left_key_cols_vect.size(), left_key_cols_vect.data());
    auto rightKeyTable = device_table::create(right_key_cols_vect.size(), right_key_cols_vect.data());

    rmm::device_vector<gdf_size_type> mergedTableIndices;
    rmm::device_vector<gdf_size_type> mergedRowIndices;
    std::tie(mergedTableIndices, mergedRowIndices) = generate_merged_indices(*leftKeyTable, *rightKeyTable, asc_desc);

    // Allocate output columns
    bool nullable = has_nulls(left_sync_table) || has_nulls(right_sync_table);
    table destination_table(left_sync_table.num_rows() + right_sync_table.num_rows(), column_dtypes(left_sync_table), nullable);
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
    auto leftInputDeviceTablePtr = device_table::create(left_sync_table);
    auto rightInputDeviceTablePtr = device_table::create(right_sync_table);
    auto outputDeviceTablePtr = device_table::create(destination_table);
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

    thrust::for_each(rmm::exec_policy()->on(0),
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
            
            materialize_bitmask(leftCol, rightCol, outCol, mergedTableIndices.data().get(), mergedRowIndices.data().get());
            
            outCol->null_count = leftCol->null_count + rightCol->null_count;
        }
    }

    return destination_table;
}

}  // namespace detail

table sorted_merge(table const& left_table,
                   table const& right_table,
                   std::vector<gdf_size_type> const& sort_by_cols,
                   rmm::device_vector<int8_t> const& asc_desc) {
    return detail::sorted_merge(left_table, right_table, sort_by_cols, asc_desc);
}

}  // namespace cudf
