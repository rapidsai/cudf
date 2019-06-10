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

#include <cstdio>

#include "table.hpp"
#include "table/device_table.cuh"
#include "table/device_table_row_operators.cuh"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cuda_utils.hpp"

namespace {
    
enum side_value { LEFT_SIDE_VALUE = 0, RIGHT_SIDE_VALUE };

__global__ void materialize_merged_bitmask_kernel(
    gdf_valid_type const* const __restrict__ source_left_mask,
    gdf_valid_type const* const __restrict__ source_right_mask,
    gdf_valid_type* const destination_mask,
    gdf_size_type const num_destination_rows,
    gdf_size_type const* const __restrict__ table_indices,
    gdf_size_type const* const __restrict__ row_indices) {
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
    gdf_valid_type const* source_mask = table_indices[destination_row] == LEFT_SIDE_VALUE ? source_left_mask : source_right_mask;
    bool const source_bit_is_valid{
        gdf_is_valid(source_mask, row_indices[destination_row])};
    
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

void materialize_bitmask(gdf_column const* leftCol, gdf_column const* rightCol, gdf_column * outCol, gdf_size_type const* tableIndices, gdf_size_type const* rowIndices) {
    constexpr gdf_size_type BLOCK_SIZE{256};
    cudf::util::cuda::grid_config_1d grid_config {outCol->size, BLOCK_SIZE };
    
    materialize_merged_bitmask_kernel
    <<<grid_config.num_blocks, grid_config.num_threads_per_block>>>
    (leftCol->valid, rightCol->valid, outCol->valid, outCol->size, tableIndices, rowIndices);
}

std::pair<rmm::device_vector<gdf_size_type>, rmm::device_vector<gdf_size_type>>
generateMergedIndices(device_table const & leftTable,
                    device_table const & rightTable,
                    gdf_column *        asc_desc,
                    cudaStream_t       cudaStream = 0) {

    const gdf_size_type left_size  = leftTable.num_rows();
    const gdf_size_type right_size = rightTable.num_rows();
    const gdf_size_type total_size = left_size + right_size;

    auto left_side = thrust::make_constant_iterator(static_cast<gdf_size_type>(LEFT_SIDE_VALUE));
    auto right_side = thrust::make_constant_iterator(static_cast<gdf_size_type>(RIGHT_SIDE_VALUE));

    auto left_indices = thrust::make_counting_iterator(static_cast<gdf_size_type>(0));
    auto right_indices = thrust::make_counting_iterator(static_cast<gdf_size_type>(0));

    auto left_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
    auto right_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

    auto left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
    auto right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side + right_size, right_indices + right_size));

    rmm::device_vector<gdf_size_type> outputTableIndices(total_size);
    rmm::device_vector<gdf_size_type> outputRowIndices(total_size);
    auto output_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(outputTableIndices.begin(), outputRowIndices.begin()));

    bool nullable = leftTable.has_nulls() || rightTable.has_nulls();
    if (nullable){
        auto ineq_op = row_inequality_comparator<true>(rightTable, leftTable, false, static_cast<int8_t *>(asc_desc->data)); 
        thrust::merge(rmm::exec_policy(cudaStream)->on(cudaStream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & tupleA,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & tupleB) {
                        const gdf_size_type left_row = thrust::get<0>(tupleA) == LEFT_SIDE_VALUE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        const gdf_size_type right_row = thrust::get<0>(tupleA) == RIGHT_SIDE_VALUE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        return ineq_op(right_row, left_row);
                    });			        
    } else {
        auto ineq_op = row_inequality_comparator<false>(rightTable, leftTable, false, static_cast<int8_t *>(asc_desc->data)); 
        thrust::merge(rmm::exec_policy(cudaStream)->on(cudaStream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    output_zip_iterator,
                    [=] __device__ (thrust::tuple<gdf_size_type, gdf_size_type> const & tupleA,
                                    thrust::tuple<gdf_size_type, gdf_size_type> const & tupleB) {
                        const gdf_size_type left_row = thrust::get<0>(tupleA) == LEFT_SIDE_VALUE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        const gdf_size_type right_row = thrust::get<0>(tupleA) == RIGHT_SIDE_VALUE ? thrust::get<1>(tupleA) : thrust::get<1>(tupleB);
                        return ineq_op(right_row, left_row);
                    });					        
    }

    return std::make_pair(outputTableIndices, outputRowIndices);
}

} // namespace

gdf_error
gdf_sorted_merge(gdf_column **          left_cols,
                 gdf_column **          right_cols,
                 const gdf_size_type    ncols,
                 const gdf_size_type *  sort_by_cols,
                 const gdf_size_type    sort_by_cols_size,
                 gdf_column *           asc_desc,
                 gdf_column **          output_cols) {

    GDF_REQUIRE((nullptr != left_cols && nullptr != right_cols && nullptr != output_cols), GDF_DATASET_EMPTY);

    GDF_REQUIRE(nullptr != asc_desc, GDF_DATASET_EMPTY);
    GDF_REQUIRE(asc_desc->dtype == GDF_INT8, GDF_UNSUPPORTED_DTYPE);

    GDF_REQUIRE(asc_desc->size <= ncols, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(sort_by_cols_size <= ncols, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(asc_desc->size == sort_by_cols_size, GDF_COLUMN_SIZE_MISMATCH);

    std::vector<gdf_column*> left_key_cols_vect(sort_by_cols_size);
    std::transform(sort_by_cols, sort_by_cols+sort_by_cols_size, left_key_cols_vect.begin(),
                  [=] (gdf_index_type const index) { return left_cols[index]; });
    
    std::vector<gdf_column*> right_key_cols_vect(sort_by_cols_size);
    std::transform(sort_by_cols, sort_by_cols+sort_by_cols_size, right_key_cols_vect.begin(),
                  [=] (gdf_index_type const index) { return right_cols[index]; });

    auto leftKeyTable = device_table::create(left_key_cols_vect.size(), left_key_cols_vect.data());
    auto rightKeyTable = device_table::create(right_key_cols_vect.size(), right_key_cols_vect.data());

    rmm::device_vector<gdf_size_type> mergedTableIndices;
    rmm::device_vector<gdf_size_type> mergedRowIndices;
    std::tie(mergedTableIndices, mergedRowIndices) = generateMergedIndices(*leftKeyTable, *rightKeyTable, asc_desc);

    // Materialize V1
    auto leftInputDeviceTablePtr = device_table::create(ncols, left_cols);
    auto rightInputDeviceTablePtr = device_table::create(ncols, right_cols);
    auto outputDeviceTablePtr = device_table::create(ncols, output_cols);
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
                        device_table const & sourceDeviceTable = thrust::get<1>(idx_tuple) == LEFT_SIDE_VALUE ? leftInputDeviceTable : rightInputDeviceTable;
                        copy_row<false>(outputDeviceTable, thrust::get<0>(idx_tuple), sourceDeviceTable, thrust::get<2>(idx_tuple));
                    });
    
    //Update outputsize

    bool nullable = leftKeyTable->has_nulls() || rightKeyTable->has_nulls();
    if (nullable) {
        std::for_each(thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
                    thrust::make_counting_iterator(static_cast<gdf_size_type>(ncols)),
                    [&] (gdf_size_type colIdx){
                        const gdf_column* leftCol = left_cols[colIdx];
                        const gdf_column* rightCol = right_cols[colIdx];
                        gdf_column* outCol = output_cols[colIdx];
                        
                        materialize_bitmask(leftCol, rightCol, outCol, mergedTableIndices.data().get(), mergedRowIndices.data().get());
                        
                        outCol->null_count = leftCol->null_count + rightCol->null_count;
                    });
    }

    return GDF_SUCCESS;
}
