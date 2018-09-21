#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

gdf_error gdf_count_nonzero_mask(gdf_valid_type * masks, int num_rows, int * count)
{
  // Number of masks in the array of bitmasks
  const size_t num_masks = gdf_get_num_chars_bitmask(num_rows);

  auto mask_bit_counter = [masks, num_masks, num_rows] __device__ (int index)
  {

    constexpr int BITS_PER_MASK = sizeof(gdf_valid_type) * 8;

    gdf_valid_type count = 0;
    gdf_valid_type mask = masks[index];
    if(index < num_masks)
    {
      // FIXME: If gdf_valid_type was 32 bits, we could use __popc
      while( mask > 0 )
      {
        mask &= (mask-1) ;
        count++;
      }
    }
    // Not all bits in the last mask correspond to rows,
    // only count the ones that correspond to rows
    else if(index == num_masks)
    {
      const int num_rows_last_mask = num_rows % BITS_PER_MASK;
      for(int i = 0; i < num_rows_last_mask; ++i)
      {
        count += mask & gdf_valid_type(1);
        mask >>= 1;
      }
    }
    return count;
  };

  thrust::device_vector<gdf_valid_type> mask_counts(num_masks);

  // FIXME: We could use transform_reduce if we didn't have to 
  // handle the last mask as a special case
  thrust::tabulate(thrust::device,
                   mask_counts.begin(),
                   mask_counts.end(),
                   mask_bit_counter);


  *count = thrust::reduce(mask_counts.begin(), 
                          mask_counts.end());

  return GDF_SUCCESS;
}


