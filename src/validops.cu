#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <gdf/utils.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


using valid32_t = uint32_t;

// To account for if gdf_valid_type is not a 4 byte type,
// compute the RATIO of the number of bytes in gdf_valid_type
// to the 4 byte type being used for casting
constexpr size_t RATIO = sizeof(valid32_t) / sizeof(gdf_valid_type);

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

  if((nullptr == masks) || (nullptr == count)){return GDF_DATASET_EMPTY;}
  if(0 == num_rows) {return GDF_SUCCESS;}

  assert(sizeof(valid32_t) >= sizeof(gdf_valid_type));

  // Number of gdf_valid_types in the validity bitmask
  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);

  // Number of 4 byte types in the validity bit mask 
  const size_t num_masks32 = num_masks / RATIO;

  // If the total number of masks is not a multiple of the RATIO 
  // between the original mask type and the 4 byte masks type, then 
  // these "remainder" masks cannot be proccessed in the transform_reduce
  // and must be handled separately
  cudaStream_t copy_stream;
  const size_t num_remainder_masks = num_masks % RATIO;
  std::vector<gdf_valid_type> remainder_masks(num_remainder_masks); 
  if(remainder_masks.size() > 0)
  {

    CUDA_TRY(cudaStreamCreate(&copy_stream));

    // Copy the remainder masks to the host
    // FIXME: Is this endian safe?
    const gdf_valid_type * first_remainder_mask = (masks + num_masks) - num_remainder_masks;
    CUDA_TRY( cudaMemcpyAsync(remainder_masks.data(), 
                              first_remainder_mask, 
                              num_remainder_masks * sizeof(gdf_valid_type), 
                              cudaMemcpyDeviceToHost, 
                              copy_stream) );
  }


  // Device lambda to count the number of valid bits in each mask
  auto mask_bit_counter = [] __device__ (valid32_t const mask)
  {
    // TODO What type will Thrust use for the temporary storage
    // during the transform_reduce? We could store this result in
    // an int8_t, but the reduction would overflow an int8_t accumulator
    return __popc(mask);
  };

  // Cast validity buffer to 4 byte type
  thrust::device_ptr<valid32_t const> masks32 = thrust::device_pointer_cast(reinterpret_cast<valid32_t const *>(masks));

  // Count the number of valid bits in all the masks that are a 
  // multiple of 4 bytes
  const size_t count32 = thrust::transform_reduce(masks32,
                                                  masks32 + num_masks32,
                                                  mask_bit_counter,
                                                  0,
                                                  thrust::plus<int>());

  CUDA_TRY( cudaGetLastError() );

  // Count the number of valid bits in the remainder masks
  size_t remainder_count = 0;
  if(remainder_masks.size() > 0)
  {
    CUDA_TRY(cudaStreamSynchronize(copy_stream));
    CUDA_TRY(cudaStreamDestroy(copy_stream));
    remainder_count = count_valid_bits_host(remainder_masks, num_rows);
  }

  // The final count of valid bits is the sum of the result from the
  // transform_reduce and the remainder masks
  *count = (count32 + remainder_count);

  return GDF_SUCCESS;
}



