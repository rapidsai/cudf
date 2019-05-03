#include <bitmask/bit_mask.cuh>
#include <table.hpp>
#include "bitmask/legacy_bitmask.hpp"
#include "cudf.h"
#include "cudf/functions.h"
#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <cassert>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>

// To account for if gdf_valid_type is not a 4 byte type,
// compute the RATIO of the number of bytes in gdf_valid_type
// to the 4 byte type being used for casting
using valid32_t = uint32_t;
constexpr size_t RATIO = sizeof(valid32_t) / sizeof(gdf_valid_type);
constexpr int BITS_PER_MASK32 = GDF_VALID_BITSIZE * RATIO;

constexpr int block_size = 256;

namespace {

/**
 * @brief Kernel to count the number of set bits in a column's validity buffer
 *
 * The underlying buffer type may only be a 1B type, but it is casted to a 4B
 * type (valid32_t) such that __popc may be used to more efficiently count the
 * number of set bits. This requires handling the last 4B element as a special
 * case as the buffer may not be a multiple of 4 bytes.
 *
 * @param[in] masks32 Pointer to buffer (casted as a 4B type) whose bits will be
 * counted
 * @param[in] num_masks32 The number of 4B elements in the buffer
 * @param[in] num_rows The number of rows in the column, i.e., the number of
 * bits in the buffer that correspond to rows
 * @param[out] global_count The number of set bits in the range of bits [0,
 * num_rows)
 */
template <typename size_type>
__global__ void count_valid_bits(valid32_t const* const masks32,
                                 int const num_masks32, int const num_rows,
                                 size_type* const global_count) {
  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // If the number of rows is not a multiple of 32, then the remaining
  // rows need to be handled separtely because not all of its bits correspond
  // to rows
  int last_mask32{0};
  int const num_rows_last_mask{num_rows % BITS_PER_MASK32};
  if (0 == num_rows_last_mask)
    last_mask32 = num_masks32;
  else
    last_mask32 = num_masks32 - 1;

  int const idx{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  int cur_mask{idx};

  size_type my_count{0};

  // Use popc to count the valid bits for the all of the masks
  // where all of the bits correspond to rows
  while (cur_mask < last_mask32) {
    my_count += __popc(masks32[cur_mask]);
    cur_mask += blockDim.x * gridDim.x;
  }

  // Handle the remainder rows
  if (idx < num_rows_last_mask) {
    gdf_valid_type const* const valids{
        reinterpret_cast<gdf_valid_type const*>(masks32)};
    int const my_row{num_rows - idx - 1};

    if (true == gdf_is_valid(valids, my_row)) ++my_count;
  }

  // Reduces the count from each thread in a block into a block count
  int const block_count{BlockReduce(temp_storage).Sum(my_count)};

  // Store the block count into the global count
  if (threadIdx.x == 0) {
    atomicAdd(global_count, block_count);
  }
}
}  // namespace

gdf_error gdf_count_nonzero_mask(gdf_valid_type const* masks,
                                 gdf_size_type num_rows, gdf_size_type* count) {
  // TODO: add a default parameter cudaStream_t stream = 0 when we move API to
  // C++

  if ((nullptr == count)) {
    return GDF_DATASET_EMPTY;
  }

  if (0 == num_rows) {
    *count = 0;
    return GDF_SUCCESS;
  }

  if (nullptr == masks) {
    *count = num_rows;
    return GDF_SUCCESS;
  }

  // Masks will be proccessed as 4B types, therefore we require that the
  // underlying type be less than or equal to 4B
  static_assert(sizeof(valid32_t) >= sizeof(gdf_valid_type),
                "gdf_valid_type is assumed to be <= 4B type");

  // Number of gdf_valid_types in the validity bitmask
  gdf_size_type const num_masks{gdf_num_bitmask_elements(num_rows)};

  // Number of 4 byte types in the validity bit mask
  gdf_size_type num_masks32{static_cast<gdf_size_type>(
      std::ceil(static_cast<float>(num_masks) / RATIO))};

  gdf_size_type h_count{0};
  if (num_masks32 > 0) {
    // TODO: Probably shouldn't create/destroy the stream every time
    cudaStream_t count_stream;
    CUDA_TRY(cudaStreamCreate(&count_stream));
    int* d_count{nullptr};

    // Cast validity buffer to 4 byte type
    valid32_t const* masks32{reinterpret_cast<valid32_t const*>(masks)};

    RMM_TRY(RMM_ALLOC((void**)&d_count, sizeof(gdf_size_type), count_stream));
    CUDA_TRY(cudaMemsetAsync(d_count, 0, sizeof(gdf_size_type), count_stream));

    gdf_size_type const grid_size{(num_masks32 + block_size - 1) / block_size};

    count_valid_bits<<<grid_size, block_size, 0, count_stream>>>(
        masks32, num_masks32, num_rows, d_count);

    CUDA_TRY(cudaGetLastError());

    CUDA_TRY(cudaMemcpyAsync(&h_count, d_count, sizeof(gdf_size_type),
                             cudaMemcpyDeviceToHost, count_stream));
    RMM_TRY(RMM_FREE(d_count, count_stream));
    CUDA_TRY(cudaStreamSynchronize(count_stream));
    CUDA_TRY(cudaStreamDestroy(count_stream));
  }

  assert(h_count >= 0);
  assert(h_count <= num_rows);

  *count = h_count;

  return GDF_SUCCESS;
}

gdf_error gdf_mask_concat(gdf_valid_type* output_mask,
                          gdf_size_type output_column_length,
                          gdf_valid_type* masks_to_concat[],
                          gdf_size_type* column_lengths,
                          gdf_size_type num_columns) {
  // This lambda is executed in a thrust algorithm. Each thread computes and
  // returns one gdf_valid_type element for the concatenated output mask
  auto mask_concatenator = [=] __device__(gdf_size_type mask_index) {
    gdf_valid_type output_m = 0;

    int cur_mask_index = 0, cur_mask_start = 0;
    int cur_mask_len = column_lengths[0];

    // Each thread processes one GDF_VALID_BITSIZE worth of valid bits
    for (int bit = 0; bit < GDF_VALID_BITSIZE; ++bit) {
      gdf_size_type output_index = mask_index * GDF_VALID_BITSIZE + bit;

      // stop when we are beyond the length of the output column (in elements)
      if (output_index >= output_column_length) break;

      // find the next column's mask when we step past the current column's
      // length
      while ((cur_mask_start + cur_mask_len <= output_index) &&
             (cur_mask_index < num_columns - 1)) {
        cur_mask_start += cur_mask_len;
        cur_mask_len = column_lengths[++cur_mask_index];
      }

      // Set each valid bit at the right location in this thread's output
      // gdf_valid_type Note: gdf_is_valid returns true when the input mask is a
      // null pointer This makes it behave as if columns with null validity
      // masks have masks of all 1s, which is the desired behavior.
      gdf_size_type index = output_index - cur_mask_start;
      if (gdf_is_valid(masks_to_concat[cur_mask_index], index)) {
        output_m |= (1 << bit);
      }
    }

    return output_m;
  };

  // This is like thrust::for_each where the lambda gets the current index into
  // the output array as input
  thrust::tabulate(rmm::exec_policy()->on(0), output_mask,
                   output_mask + gdf_num_bitmask_elements(output_column_length),
                   mask_concatenator);

  CUDA_TRY(cudaGetLastError());

  return GDF_SUCCESS;
}

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
