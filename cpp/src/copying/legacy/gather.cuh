#include <cudf/legacy/copying.hpp>
#include <cudf/cudf.h>
#include <utilities/legacy/bit_util.cuh>
#include <utilities/legacy/cudf_utils.h>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <table/legacy/device_table.cuh>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <utilities/legacy/column_utils.hpp>
#include <utilities/legacy/cuda_utils.hpp>

#include <bitmask/legacy/valid_if.cuh>

using bit_mask::bit_mask_t;

namespace cudf {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Function object to check if an index is within the bounds [begin,
 * end).
 *
 *---------------------------------------------------------------------------**/
template <typename map_type>
struct bounds_checker {
  cudf::size_type begin;
  cudf::size_type end;

  __device__ bounds_checker(cudf::size_type begin_, cudf::size_type end_)
    : begin{begin_}, end{end_} {}

  __device__ __forceinline__ bool operator()(map_type const index) {
    return ((index >= begin) && (index < end));
  }
};


template <bool ignore_out_of_bounds, typename map_iterator_type>
__global__ void gather_bitmask_kernel(const bit_mask_t *const *source_valid,
                                      cudf::size_type num_source_rows,
                                      map_iterator_type gather_map,
                                      bit_mask_t **destination_valid,
                                      cudf::size_type num_destination_rows,
                                      cudf::size_type *d_count,
                                      cudf::size_type num_columns) {
  for (cudf::size_type i = 0; i < num_columns; i++) {
    const bit_mask_t *__restrict__ source_valid_col = source_valid[i];
    bit_mask_t *__restrict__ destination_valid_col = destination_valid[i];

    const bool src_has_nulls = source_valid_col != nullptr;
    const bool dest_has_nulls = destination_valid_col != nullptr;

    if (dest_has_nulls) {
      cudf::size_type destination_row_base = blockIdx.x * blockDim.x;

      cudf::size_type valid_count_accumulate = 0;

      while (destination_row_base < num_destination_rows) {
        cudf::size_type destination_row = destination_row_base + threadIdx.x;

        const bool thread_active = destination_row < num_destination_rows;
        cudf::size_type source_row =
          thread_active ? gather_map[destination_row] : 0;

        const uint32_t active_threads =
          __ballot_sync(0xffffffff, thread_active);

        bool source_bit_is_valid;
        if (ignore_out_of_bounds) {
          if (0 <= source_row && source_row < num_source_rows) {
            source_bit_is_valid =
              src_has_nulls ? bit_mask::is_valid(source_valid_col, source_row)
              : true;
          } else {
            // If gather_map does not include this row we should just keep the
            // original value,
            source_bit_is_valid =
              bit_mask::is_valid(destination_valid_col, destination_row);
          }
        } else {
          source_bit_is_valid =
            src_has_nulls ? bit_mask::is_valid(source_valid_col, source_row)
            : true;
        }

        // Use ballot to find all valid bits in this warp and create the output
        // bitmask element
        const uint32_t valid_warp =
          __ballot_sync(active_threads, source_bit_is_valid);

        const cudf::size_type valid_index =
          cudf::util::detail::bit_container_index<bit_mask_t>(
              destination_row);
        // Only one thread writes output
        if (0 == threadIdx.x % warp_size && thread_active) {
          destination_valid_col[valid_index] = valid_warp;
        }
        valid_count_accumulate += cudf::detail::single_lane_popc_block_reduce(valid_warp);

        destination_row_base += blockDim.x * gridDim.x;
      }
      if (threadIdx.x == 0) {
        atomicAdd(d_count + i, valid_count_accumulate);
      }
    }
  }
}


/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * gdf_column. To be used with the cudf::type_dispatcher.
 *
 *---------------------------------------------------------------------------**/
struct column_gatherer {
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam column_type Dispatched type for the column being gathered
   * @param source_column The column to gather from
   * @param gather_map An iterator over integral values representing the gather map
   * @param destination_column The column to gather into
   * @param ignore_out_of_bounds Ignore values in `gather_map` that are
   * out of bounds
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename column_type, typename iterator_type>
  void operator()(gdf_column const *source_column,
		  iterator_type gather_map,
		  gdf_column *destination_column, bool ignore_out_of_bounds,
		  bool sync_nvstring_category, cudaStream_t stream) {
    column_type const *source_data{
      static_cast<column_type const *>(source_column->data)};
    column_type *destination_data{
      static_cast<column_type *>(destination_column->data)};

    cudf::size_type const num_destination_rows{destination_column->size};

    // If gathering in-place or scattering nvstring
    // (in which case the sync_nvstring_category should be set to true)
    // allocate temporary buffers to hold intermediate results
    bool const sync_category =
      std::is_same<column_type, cudf::nvstring_category>::value &&
      sync_nvstring_category;
    bool const in_place = !sync_category && (source_data == destination_data);

    gdf_column temp_src{};
    gdf_column temp_dest{};

    if (sync_category) {
      // sync the categories.
      temp_src = cudf::copy(*source_column);
      temp_dest = cudf::copy(*destination_column);
      const gdf_column *const input_columns[2] = {source_column, &temp_dest};
      gdf_column *output_columns[2] = {&temp_src, destination_column};

      CUDF_EXPECTS(GDF_SUCCESS ==
		   sync_column_categories(input_columns, output_columns, 2),
		   "Failed to synchronize NVCategory");

      source_data = static_cast<column_type *>(temp_src.data);
    }

    rmm::device_vector<column_type> in_place_buffer;
    if (in_place) {
      in_place_buffer.resize(num_destination_rows);
      destination_data = in_place_buffer.data().get();
    }

    if (ignore_out_of_bounds) {
      thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map,
			gather_map + num_destination_rows, gather_map,
			source_data, destination_data,
			bounds_checker<decltype(*gather_map)>{0, source_column->size});
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map,
		     gather_map+num_destination_rows, source_data,
		     destination_data);
    }

    if (sync_category) {
      gdf_column_free(&temp_src);
      gdf_column_free(&temp_dest);
    }

    // Copy temporary buffers used for in-place gather to destination column
    if (in_place) {
      thrust::copy(rmm::exec_policy(stream)->on(stream), destination_data,
		   destination_data + num_destination_rows,
		   static_cast<column_type *>(destination_column->data));
    }

    CHECK_CUDA(stream);
  }
};

/**
 * @brief Specifies the behavior of index_converter
 */
enum index_conversion {
    NEGATIVE_TO_POSITIVE = 0,
    NONE
};

/**---------------------------------------------------------------------------*
* @brief Function object for applying a transformation on the gathermap
* that converts negative indices to positive indices
*
* A negative index `i` is transformed to `i + size`, where `size` is
* the number of elements in the column being gathered from.
* Allowable values for the index `i` are in the range `[-size, size)`.
* Thus, when gathering from a column of size `10`, the index `-1`
* is transformed to `9` (i.e., the last element), `-2` is transformed
* to `8` (the second-to-last element) and so on.
* Positive indices are unchanged by this transformation.
*---------------------------------------------------------------------------**/
template <typename map_type, index_conversion ic=index_conversion::NONE>
struct index_converter : public thrust::unary_function<map_type,map_type>{};


template <typename map_type>
struct index_converter<map_type, index_conversion::NEGATIVE_TO_POSITIVE>
{
  index_converter(cudf::size_type n_rows)
  : n_rows(n_rows) {}

  __device__
  map_type operator()(map_type in) const
  {
    return ((in % n_rows) + n_rows) % n_rows;
  }
  cudf::size_type n_rows;
};

template <typename map_type>
struct index_converter<map_type, index_conversion::NONE>
{
  index_converter(cudf::size_type n_rows)
  : n_rows(n_rows) {}

  __device__
  map_type operator()(map_type in) const
  {
    return in;
  }
  cudf::size_type n_rows;
};


template <typename iterator_type>
void gather(table const *source_table, iterator_type gather_map,
	    table *destination_table, bool check_bounds, bool ignore_out_of_bounds,
	    bool sync_nvstring_category, bool allow_negative_indices)
{
  auto source_n_cols = source_table->num_columns();
  auto source_n_rows = source_table->num_rows();

  std::vector<util::cuda::scoped_stream> v_stream(source_n_cols);

  for (cudf::size_type i = 0; i < source_n_cols; i++) {
    // Perform sanity checks
    gdf_column *dest_col = destination_table->get_column(i);
    const gdf_column *src_col = source_table->get_column(i);

    CUDF_EXPECTS(src_col->dtype == dest_col->dtype, "Column type mismatch");

    CUDF_EXPECTS(dest_col->data != nullptr, "Missing source data buffer.");

    // The data gather for n columns will be put on the first n streams
    cudf::type_dispatcher(src_col->dtype, column_gatherer{}, src_col,
			  gather_map, dest_col, ignore_out_of_bounds,
			  sync_nvstring_category, v_stream[i]);

    if (cudf::has_nulls(*src_col)) {
      CUDF_EXPECTS(cudf::is_nullable(*dest_col),
		   "Missing destination null mask.");
    }
  }

  rmm::device_vector<cudf::size_type> null_counts(source_n_cols, 0);

  std::vector<bit_mask_t*> source_bitmasks_host(source_n_cols);
  std::vector<bit_mask_t*> destination_bitmasks_host(source_n_cols);

  std::vector<rmm::device_vector<bit_mask_t>> inplace_buffers(source_n_cols);

  // loop over each column, check if inplace and allocate buffer if true.
  for (cudf::size_type i = 0; i < source_n_cols; i++) {
    const gdf_column *dest_col = destination_table->get_column(i);
    source_bitmasks_host[i] =
      reinterpret_cast<bit_mask_t *>(source_table->get_column(i)->valid);
    // Allocate inplace buffer
    if (cudf::is_nullable(*dest_col) &&
	dest_col->valid == source_table->get_column(i)->valid) {
      inplace_buffers[i].resize(gdf_valid_allocation_size(dest_col->size));
      destination_bitmasks_host[i] = inplace_buffers[i].data().get();
    } else {
      destination_bitmasks_host[i] = reinterpret_cast<bit_mask_t *>(dest_col->valid);
    }
  }

  // In the following we allocate the device array thats hold the valid
  // bits.
  rmm::device_vector<bit_mask_t*> source_bitmasks(source_bitmasks_host);
  rmm::device_vector<bit_mask_t*> destination_bitmasks(destination_bitmasks_host);

  auto bitmask_kernel =
    ignore_out_of_bounds ? gather_bitmask_kernel<true, decltype(gather_map)> : gather_bitmask_kernel<false, decltype(gather_map)>;

  int gather_grid_size;
  int gather_block_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
	       &gather_grid_size, &gather_block_size, bitmask_kernel));

  bitmask_kernel<<<gather_grid_size, gather_block_size>>>(
      source_bitmasks.data().get(), source_table->num_rows(),
      gather_map, destination_bitmasks.data().get(), destination_table->num_rows(),
      null_counts.data().get(), source_n_cols);

  std::vector<cudf::size_type> h_count(source_n_cols);
  CUDA_TRY(cudaMemcpy(h_count.data(), null_counts.data().get(),
		      sizeof(cudf::size_type) * source_n_cols, cudaMemcpyDeviceToHost));

  // loop over each column, check if inplace and copy the result from the
  // buffer back to destination if true.
  for (cudf::size_type i = 0; i < destination_table->num_columns(); i++) {
    gdf_column *dest_col = destination_table->get_column(i);
    if (is_nullable(*dest_col)) {
      // Copy temp buffer content back to column
      if (dest_col->valid == source_table->get_column(i)->valid) {
	cudf::size_type num_bitmask_elements =
	  gdf_num_bitmask_elements(dest_col->size);
	CUDA_TRY(cudaMemcpy(dest_col->valid, destination_bitmasks_host[i],
			    num_bitmask_elements, cudaMemcpyDeviceToDevice));
      }
      dest_col->null_count = dest_col->size - h_count[i];
    } else {
      dest_col->null_count = 0;
    }
  }
}


} // namespace detail
} // namespace cudf
