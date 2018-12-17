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
#include <vector>
#include <cassert>

#include "cudf.h"
#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.h"
#include "utilities/cudf_utils.h"
#include "bitmask_ops.h"

#include <thrust/tabulate.h>


/** --------------------------------------------------------------------------*
 * @Synopsis  Counts the number of valid bits for the specified number of rows
 * in the host vector of gdf_valid_type masks
 * 
 * @Param masks The host vector of masks whose bits will be counted
 * @Param num_rows The number of bits to count
 * 
 * @Returns  The number of valid bits in [0, num_rows) in the host vector of masks
 * ----------------------------------------------------------------------------*/
size_t count_valid_bits_host(std::vector<gdf_valid_type> const & masks, int const num_rows)
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


/* ---------------------------------------------------------------------------*
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
  gdf_error gdf_status{GDF_SUCCESS};
  
  // TODO: Probably shouldn't create/destroy the stream every time
  cudaStream_t count_stream;
  CUDA_TRY(cudaStreamCreate(&count_stream));
  
  gdf_status = count_nonzero_mask(masks, num_rows, *count, count_stream);

  CUDA_TRY(cudaStreamDestroy(count_stream));

  return gdf_status;
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
    thrust::tabulate(rmm::exec_policy(cudaStream_t{0}),
                     output_mask,
                     output_mask + gdf_get_num_chars_bitmask(output_column_length),
                     mask_concatenator);

    CUDA_TRY( cudaGetLastError() );
        
    return GDF_SUCCESS;
}


