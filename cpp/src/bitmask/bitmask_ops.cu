#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "cudf/functions.h"
#include "rmm/thrust_rmm_allocator.h"

#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_vector.h>

/*
 * bit_mask_null_counts Generated using the following code

#include <iostream>


int main()
{
	for (int i = 0 ; i != 256 ; i++) {
		int count = 0;
		for (int p = 0 ; p != 8 ; p++) {
			if (i & (1 << p)) {
				count++;
			}
		}
		std::cout<<(8-count)<<", ";
	}
	std::cout<<std::endl;
}
 */
std::vector<gdf_valid_type> bit_mask_null_counts = { 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0 };

unsigned char gdf_num_bits_zero_after_pos(unsigned char number, int pos){
	//if pos == 0 then its aligned
	if(pos == 0){
		return 0;
	}
	unsigned char count = 0;
	for (int p = pos ; p != 8 ; p++) {
		if (number & (number << p)) {
			count++;
		}
	}
	return (8 - pos) - count;
}

gdf_error all_bitmask_on(gdf_valid_type * valid_out, gdf_size_type & out_null_count, gdf_size_type num_values, cudaStream_t stream){
	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );

	thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
	gdf_valid_type max_char = 255;
	thrust::fill(rmm::exec_policy(stream),thrust::detail::make_normal_iterator(valid_out_ptr),thrust::detail::make_normal_iterator(valid_out_ptr + num_chars_bitmask),max_char);
	//we have no nulls so set all the bits in gdf_valid_type to 1
	out_null_count = 0;
	return GDF_SUCCESS;
}

gdf_error apply_bitmask_to_bitmask(gdf_size_type & out_null_count, gdf_valid_type * valid_out, gdf_valid_type * valid_left, gdf_valid_type * valid_right,
		cudaStream_t stream, gdf_size_type num_values){

	gdf_size_type num_chars_bitmask = ( ( num_values +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::device_ptr<gdf_valid_type> valid_out_ptr = thrust::device_pointer_cast(valid_out);
	thrust::device_ptr<gdf_valid_type> valid_left_ptr = thrust::device_pointer_cast(valid_left);
	//here we are basically figuring out what is the last pointed to unsigned char that can contain part of the bitmask
	thrust::device_ptr<gdf_valid_type> valid_left_end_ptr = thrust::device_pointer_cast(valid_left + num_chars_bitmask );
	thrust::device_ptr<gdf_valid_type> valid_right_ptr = thrust::device_pointer_cast(valid_right);


	thrust::transform(rmm::exec_policy(stream), thrust::detail::make_normal_iterator(valid_left_ptr),
			thrust::detail::make_normal_iterator(valid_left_end_ptr), thrust::detail::make_normal_iterator(valid_right_ptr),
			thrust::detail::make_normal_iterator(valid_out_ptr), thrust::bit_and<gdf_valid_type>());


	char * last_char = new char[1];
	cudaError_t error = cudaMemcpyAsync(last_char,valid_out + ( num_chars_bitmask-1),sizeof(gdf_valid_type),cudaMemcpyDeviceToHost,stream);


	rmm::device_vector<gdf_valid_type> bit_mask_null_counts_device(bit_mask_null_counts);

	//this permutation iterator makes it so that each char basically gets replaced with its number of null counts
	//so if you sum up this perm iterator you add up all of the counts for null values per unsigned char
	thrust::permutation_iterator<rmm::device_vector<gdf_valid_type>::iterator,thrust::detail::normal_iterator<thrust::device_ptr<gdf_valid_type> > >
	null_counts_iter( bit_mask_null_counts_device.begin(),thrust::detail::make_normal_iterator(valid_out_ptr));

	//you will notice that we subtract the number of zeros we found in the last character
	out_null_count = thrust::reduce(rmm::exec_policy(stream),null_counts_iter, null_counts_iter + num_chars_bitmask) - gdf_num_bits_zero_after_pos(*last_char,num_values % GDF_VALID_BITSIZE );

	delete[] last_char;
	return GDF_SUCCESS;
}

// To account for if gdf_valid_type is not a 4 byte type,
// compute the RATIO of the number of bytes in gdf_valid_type
// to the 4 byte type being used for casting
using valid32_t = uint32_t;
constexpr size_t RATIO = sizeof(valid32_t) / sizeof(gdf_valid_type);
constexpr int BITS_PER_MASK32 = GDF_VALID_BITSIZE * RATIO;

constexpr int block_size = 256;

/* --------------------------------------------------------------------------*/
/** 
 * @brief Kernel to count the number of set bits in a column's validity buffer
 *
 * The underlying buffer type may only be a 1B type, but it is casted to a 4B 
 * type (valid32_t) such that __popc may be used to more efficiently count the 
 * number of set bits. This requires handling the last 4B element as a special 
 * case as the buffer may not be a multiple of 4 bytes.
 * 
 * @Param[in] masks32 Pointer to buffer (casted as a 4B type) whose bits will be counted
 * @Param[in] num_masks32 The number of 4B elements in the buffer
 * @Param[in] num_rows The number of rows in the column, i.e., the number of bits
 * in the buffer that correspond to rows
 * @Param[out] global_count The number of set bits in the range of bits [0, num_rows)
 */
/* ----------------------------------------------------------------------------*/
template <typename size_type>
__global__ 
void count_valid_bits(valid32_t const * const masks32, 
                      int const num_masks32, 
                      int const num_rows, 
                      size_type * const global_count)
{
  using BlockReduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // If the number of rows is not a multiple of 32, then the remaining 
  // rows need to be handled separtely because not all of its bits correspond
  // to rows
  int last_mask32{0};
  int const num_rows_last_mask{num_rows % BITS_PER_MASK32};
  if(0 == num_rows_last_mask)
    last_mask32 = num_masks32;
  else
    last_mask32 = num_masks32 - 1;

  int const idx{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  int cur_mask{idx};

  size_type my_count{0};

  // Use popc to count the valid bits for the all of the masks 
  // where all of the bits correspond to rows
  while(cur_mask < last_mask32)
  {
    my_count += __popc(masks32[cur_mask]);
    cur_mask += blockDim.x * gridDim.x;
  }

  // Handle the remainder rows
  if(idx < num_rows_last_mask)
  {
    gdf_valid_type const * const valids{reinterpret_cast<gdf_valid_type const *>(masks32)};
    int const my_row{num_rows - idx - 1};

    if(true == gdf_is_valid(valids,my_row))
      ++my_count;
  }

  // Reduces the count from each thread in a block into a block count
  int const block_count{BlockReduce(temp_storage).Sum(my_count)};

  // Store the block count into the global count
  if(threadIdx.x == 0)
  {
    atomicAdd(global_count, block_count);
  }
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
gdf_error count_nonzero_mask(gdf_valid_type const * masks,
                                 int num_rows,
                                 int& count,
                                 cudaStream_t stream)
{
  GDF_REQUIRE(masks != nullptr, GDF_DATASET_EMPTY);
  
  if(0 == num_rows) {return GDF_SUCCESS;}

  // Masks will be proccessed as 4B types, therefore we require that the underlying
  // type be less than or equal to 4B
  assert(sizeof(valid32_t) >= sizeof(gdf_valid_type));

  // Number of gdf_valid_types in the validity bitmask
  size_t const num_masks{gdf_get_num_chars_bitmask(num_rows)};

  // Number of 4 byte types in the validity bit mask 
  size_t num_masks32{static_cast<size_t>(std::ceil(static_cast<float>(num_masks) / RATIO))};

  int h_count{0};
  if(num_masks32 > 0)
  {
    int* d_count{nullptr};

    // Cast validity buffer to 4 byte type
    valid32_t const* masks32{reinterpret_cast<const valid32_t*>(masks)};

    RMM_TRY(RMM_ALLOC((void**)&d_count, sizeof(int), stream));
    CUDA_TRY(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    size_t const grid_size{(num_masks32 + block_size - 1)/block_size};

    count_valid_bits<<<grid_size, block_size,0,stream>>>(masks32, num_masks32, num_rows, d_count);

    CUDA_TRY( cudaGetLastError() );

    CUDA_TRY(cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    RMM_TRY(RMM_FREE(d_count, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  assert(h_count >= 0);
  assert(h_count <= num_rows);

  count = h_count;

  return GDF_SUCCESS;
}
