/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
/** ---------------------------------------------------------------------------*
 * @brief Operations on GDF column validity bitmasks
 * 
 * @file column.cpp
 * ---------------------------------------------------------------------------**/
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

/** --------------------------------------------------------------------------*
 * @Synopsis  Counts the number of valid bits for the specified number of rows
 * in the host vector of gdf_valid_type masks
 * 
 * @Param masks The host vector of masks whose bits will be counted
 * @Param num_rows The number of bits to count
 * 
 * @Returns  The number of valid bits in [0, num_rows) in the host vector of masks
 * ----------------------------------------------------------------------------*/
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

  constexpr int BITS_PER_MASK = sizeof(gdf_valid_type) * 8;

  // Only count the bits in the last mask that correspond to rows
  int num_rows_last_mask = num_rows % BITS_PER_MASK;

  if(num_rows_last_mask == 0)
    num_rows_last_mask = BITS_PER_MASK;

  gdf_valid_type last_mask = *(masks.end() - 1);
  for(int i = 0; (i < num_rows_last_mask) && (last_mask > 0); ++i)
  {
    count += (last_mask & gdf_valid_type(1));
    last_mask >>= 1;
  }

  return count;
}

/** --------------------------------------------------------------------------*
 * @Synopsis  Counts the number of valid bits for the specified number of rows
 * in a validity bitmask.
 * 
 * @Param[in] masks The validity bitmask buffer in device memory
 * @Param[in] num_rows The number of bits to count
 * @Param[out] count The number of valid bits in the buffer from [0, num_rows)
 * 
 * @Returns  GDF_SUCCESS upon successful completion 
 *
 * ----------------------------------------------------------------------------*/
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

/** ---------------------------------------------------------------------------*
 * @brief Concatenate the validity bitmasks of multiple columns
 * 
 * Accounts for the differences between lengths of columns and their bitmasks 
 * (e.g. because gdf_valid_type is larger than one bit).
 * 
 * @param[out] output_mask The concatenated mask
 * @param[in] output_column_length The total length (in data elements) of the 
 *                                 concatenated column
 * @param[in] masks_to_concat The array of device pointers to validity bitmasks
 *                            for the columns to concatenate
 * @param[in] column_lengths An array of lengths of the columns to concatenate
 * @param[in] num_columns The number of columns to concatenate
 * @return gdf_error GDF_SUCCESS or GDF_CUDA_ERROR if there is a runtime CUDA
           error
 * ---------------------------------------------------------------------------**/
gdf_error gdf_mask_concat(gdf_valid_type *output_mask,
                          gdf_size_type output_column_length,            
                          gdf_valid_type *masks_to_concat[], 
                          gdf_size_type *column_lengths, 
                          gdf_size_type num_columns)
{
    // This lambda is executed in a thrust algorithm. Each thread computes and
    // returns one gdf_valid_type element for the concatenated output mask
    auto mask_concatenator = [=] __device__ (gdf_size_type mask_index) {
      gdf_valid_type output_m = 0;
     
      int cur_mask_index = 0, cur_mask_start = 0;
      int cur_mask_len = column_lengths[0];
      
      // Each thread processes one GDF_VALID_BITSIZE worth of valid bits
      for (int bit = 0; bit < GDF_VALID_BITSIZE; ++bit) 
      { 
        gdf_size_type output_index = mask_index * GDF_VALID_BITSIZE + bit;

        // stop when we are beyond the length of the output column (in elements)
        if (output_index >= output_column_length) break;
        
        // find the next column's mask when we step past the current column's length
        while ( (cur_mask_start + cur_mask_len <= output_index) && (cur_mask_index < num_columns - 1) )
        {
          cur_mask_start += cur_mask_len;
          cur_mask_len = column_lengths[++cur_mask_index];           
        }
        
        // Set each valid bit at the right location in this thread's output gdf_valid_type
        // Note: gdf_is_valid returns true when the input mask is a null pointer
        // This makes it behave as if columns with null validity masks have masks of all 1s,
        // which is the desired behavior.
        gdf_size_type index = output_index - cur_mask_start;
        if ( gdf_is_valid(masks_to_concat[cur_mask_index], index) ) 
        {
          output_m |= (1 << bit);     
        }
      }

      return output_m;
    };

    // This is like thrust::for_each where the lambda gets the current index into the output array
    // as input
    thrust::tabulate(thrust::cuda::par,
                     output_mask,
                     output_mask + gdf_get_num_chars_bitmask(output_column_length),
                     mask_concatenator);

    CUDA_TRY( cudaGetLastError() );
        
    return GDF_SUCCESS;
}


