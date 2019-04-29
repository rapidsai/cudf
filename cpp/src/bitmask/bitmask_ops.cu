#include <rmm/thrust_rmm_allocator.h>
#include <bitmask/bit_mask.cuh>
#include <table/table.hpp>
#include "bitmask/legacy_bitmask.hpp"
#include "cudf.h"
#include "cudf/functions.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/transform.h>
#include <vector>

gdf_error all_bitmask_on(gdf_valid_type* valid_out,
                         gdf_size_type& out_null_count,
                         gdf_size_type num_values, cudaStream_t stream) {
  gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(num_values);

  gdf_valid_type max_char = 255;
  thrust::fill(rmm::exec_policy(stream)->on(stream), valid_out,
               valid_out + num_bitmask_elements, max_char);
  // we have no nulls so set all the bits in gdf_valid_type to 1
  out_null_count = 0;
  return GDF_SUCCESS;
}

gdf_error apply_bitmask_to_bitmask(gdf_size_type& out_null_count,
                                   gdf_valid_type* valid_out,
                                   gdf_valid_type* valid_left,
                                   gdf_valid_type* valid_right,
                                   cudaStream_t stream,
                                   gdf_size_type num_values) {
  gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(num_values);

  thrust::transform(rmm::exec_policy(stream)->on(stream), valid_left,
                    valid_left + num_bitmask_elements, valid_right, valid_out,
                    thrust::bit_and<gdf_valid_type>());

  gdf_size_type non_nulls;
  auto error = gdf_count_nonzero_mask(valid_out, num_values, &non_nulls);
  out_null_count = num_values - non_nulls;
  return error;
}

namespace cudf {
namespace {

/**
 * @brief  Computes a bitmask from the bitwise AND of a set of bitmasks.
 */
struct bitwise_and {
  bitwise_and(bit_mask::bit_mask_t** _masks, gdf_size_type _num_masks)
      : masks{_masks}, num_masks(_num_masks) {}

  __device__ inline bit_mask::bit_mask_t operator()(
      gdf_size_type mask_element_index) {
    using namespace bit_mask;
    bit_mask_t result_mask{~bit_mask_t{0}};  // all 1s
    for (gdf_size_type i = 0; i < num_masks; ++i) {
      result_mask &= masks[i][mask_element_index];
    }
    return result_mask;
  }

  gdf_size_type num_masks;
  bit_mask::bit_mask_t** masks;
};
}  // namespace

rmm::device_vector<bit_mask::bit_mask_t> row_bitmask(cudf::table const& table,
                                                     cudaStream_t stream) {
  using namespace bit_mask;
  rmm::device_vector<bit_mask_t> row_bitmask(num_elements(table.num_rows()),
                                             ~bit_mask_t{0});

  // Populate vector of pointers to the bitmasks of columns that contain
  // NULL values
  std::vector<bit_mask_t*> column_bitmasks{row_bitmask.data().get()};
  std::for_each(
      table.begin(), table.end(), [&column_bitmasks](gdf_column const* col) {
        if ((nullptr != col->valid) and (col->null_count > 0)) {
          column_bitmasks.push_back(reinterpret_cast<bit_mask_t*>(col->valid));
        }
      });
  rmm::device_vector<bit_mask_t*> d_column_bitmasks{column_bitmasks};

  // Compute bitwise AND of all key columns' bitmasks
  thrust::tabulate(
      rmm::exec_policy(stream)->on(stream), row_bitmask.begin(),
      row_bitmask.end(),
      bitwise_and(d_column_bitmasks.data().get(), d_column_bitmasks.size()));

  return row_bitmask;
}
}  // namespace cudf
