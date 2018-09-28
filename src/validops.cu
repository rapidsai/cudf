#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <gdf/utils.h>
#include <vector>
#include <cassert>
#include <cub/cub.cuh>


using valid32_t = uint32_t;

// To account for if gdf_valid_type is not a 4 byte type,
// compute the RATIO of the number of bytes in gdf_valid_type
// to the 4 byte type being used for casting
constexpr size_t RATIO = sizeof(valid32_t) / sizeof(gdf_valid_type);

constexpr int block_size = 256;

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Counts the number of valid bits for the specified number of rows
   in the host vector of gdf_valid_type masks
 * 
 * @Param masks The host vector of masks whose bits will be counted
 * @Param num_rows The number of bits to count
 * 
 * @Returns  The number of valid bits in [0, num_rows) in the host vector of masks
 */
/* ----------------------------------------------------------------------------*/
size_t count_valid_bits_host(std::vector<gdf_valid_type> const & masks, const int num_rows)
{
  if((0 == num_rows) || (0 == masks.size())){
    return 0;
  }

  size_t count{0};

  // Count the valid bits for all masks except the last one
  for(size_t i = 0; i < (masks.size() - 1); ++i)
  {
    gdf_valid_type current_mask = masks[i];

    while(current_mask > 0)
    {
      current_mask &= (current_mask-1) ;
      count++;
    }
  }

  // Only count the bits in the last mask that correspond to rows
  int num_rows_last_mask = num_rows % GDF_VALID_BITSIZE;

  if(num_rows_last_mask == 0)
    num_rows_last_mask = GDF_VALID_BITSIZE;

  gdf_valid_type last_mask = *(masks.end() - 1);
  for(int i = 0; (i < num_rows_last_mask) && (last_mask > 0); ++i)
  {
    count += (last_mask & gdf_valid_type(1));
    last_mask >>= 1;
  }

  return count;
}

constexpr int BITS_PER_MASK32 = GDF_VALID_BITSIZE * RATIO;

__global__ 
void count_valid_bits(valid32_t const * const __restrict__ masks32, 
                      const int num_masks32, 
                      const int num_rows, 
                      int * const __restrict__ global_count)
{

  int cur_mask = threadIdx.x + blockIdx.x * blockDim.x;

  typedef cub::BlockReduce<int, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int my_count = 0;
  while(cur_mask < (num_masks32 - 1))
  {
    my_count += __popc(masks32[cur_mask]);
    cur_mask += blockDim.x * gridDim.x;
  }

  if(cur_mask == (num_masks32 - 1))
  {
    valid32_t last_mask = masks32[num_masks32 - 1];
    int num_rows_last_mask = num_rows % BITS_PER_MASK32;
    if(num_rows_last_mask == 0)
      num_rows_last_mask = BITS_PER_MASK32;

    for(int i = 0; i < num_rows_last_mask; ++i)
    {
      my_count += last_mask & gdf_valid_type(1);
      last_mask >>= 1;
    }
  }

  int block_count = BlockReduce(temp_storage).Sum(my_count);

  if(threadIdx.x == 0)
  {
    atomicAdd(global_count, block_count);
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Counts the number of valid bits for the specified number of rows
   in a validity bitmask.
 * 
 * @Param[in] masks The validity bitmask buffer in device memory
 * @Param[in] num_rows The number of bits to count
 * @Param[out] count The number of valid bits in the buffer from [0, num_rows)
 * 
 * @Returns  GDF_SUCCESS upon successful completion 
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_count_nonzero_mask(gdf_valid_type const * masks, int num_rows, int * count)
{

  // Why am I getting an unused function warning error if I don't do this?
  gdf_is_valid(nullptr, 0);

  if((nullptr == masks) || (nullptr == count)){return GDF_DATASET_EMPTY;}
  if(0 == num_rows) {return GDF_SUCCESS;}

  assert(sizeof(valid32_t) >= sizeof(gdf_valid_type));

  // Number of gdf_valid_types in the validity bitmask
  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);

  // Number of 4 byte types in the validity bit mask 
  size_t num_masks32 = std::ceil(static_cast<float>(num_masks) / RATIO);

  int h_count{0};
  if(num_masks32 > 0)
  {
    cudaStream_t count_stream;
    CUDA_TRY(cudaStreamCreate(&count_stream));
    int * d_count;
    // Cast validity buffer to 4 byte type
    valid32_t const * masks32 = reinterpret_cast<valid32_t const *>(masks);

    CUDA_TRY(cudaMalloc(&d_count, sizeof(int)));
    CUDA_TRY(cudaMemsetAsync(d_count, 0, sizeof(int),count_stream));

    const int grid_size = (num_masks32 + block_size - 1)/block_size;

    count_valid_bits<<<grid_size, block_size,0,count_stream>>>(masks32, num_masks32, num_rows, d_count);

    CUDA_TRY( cudaGetLastError() );

    CUDA_TRY(cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost,count_stream));
    CUDA_TRY(cudaStreamSynchronize(count_stream));
    CUDA_TRY(cudaStreamDestroy(count_stream));
  }

  assert(h_count >= 0);
  assert(h_count <= num_rows);

  *count = h_count;

  return GDF_SUCCESS;
}



