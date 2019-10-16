#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/cudf_utils.h>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include <bitmask/valid_if.cuh>

namespace cudf {
namespace experimental {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Function object to check if an index is within the bounds [begin,
 * end).
 *
 *---------------------------------------------------------------------------**/
template <typename map_type>
struct bounds_checker {
  size_type begin;
  size_type end;

  __device__ bounds_checker(size_type begin_, size_type end_)
    : begin{begin_}, end{end_} {}

  __device__ __forceinline__ bool operator()(map_type const index) {
    return ((index >= begin) && (index < end));
  }
};


template <bool ignore_out_of_bounds, typename map_iterator_type>
__global__ void gather_bitmask_kernel(const bitmask_type *const *source_valid,
				      size_type num_source_rows,
				      map_iterator_type gather_map,
				      bitmask_type **destination_valid,
				      size_type num_destination_rows,
				      size_type *d_count,
				      size_type num_columns) {
  for (size_type i = 0; i < num_columns; i++) {
    const bitmask_type *__restrict__ source_valid_col = source_valid[i];
    bitmask_type *__restrict__ destination_valid_col = destination_valid[i];

    const bool src_has_nulls = source_valid_col != nullptr;
    const bool dest_has_nulls = destination_valid_col != nullptr;

    const int warp_size = 32;

    if (dest_has_nulls) {
      size_type destination_row_base = blockIdx.x * blockDim.x;

      size_type valid_count_accumulate = 0;

      while (destination_row_base < num_destination_rows) {
	size_type destination_row = destination_row_base + threadIdx.x;

	const bool thread_active = destination_row < num_destination_rows;
	size_type source_row =
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

	const size_type valid_index =
	  cudf::util::detail::bit_container_index<bitmask_type>(
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
  template <typename column_type, typename iterator_type,
    std::enable_if_t<std::is_integral<column_type>::value or
                     std::is_floating_point<column_type>::value>* = nullptr>
    void operator()(column_view source_column,
		  iterator_type gather_map,
		  mutable_column_view destination_column,
		  bool ignore_out_of_bounds,
		  util::cuda::scoped_stream stream) {
    column_type const *source_data{source_column.data<column_type>()};
    column_type *destination_data{destination_column.data<column_type>()};

    size_type const num_destination_rows{destination_column.size()};

    // If gathering in-place allocate temporary buffers to hold intermediate results
    bool const in_place = (source_data == destination_data);

    rmm::device_vector<column_type> in_place_buffer;
    if (in_place) {
      in_place_buffer.resize(num_destination_rows);
      destination_data = in_place_buffer.data().get();
    }

    if (ignore_out_of_bounds) {
      thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map,
			gather_map + num_destination_rows, gather_map,
			source_data, destination_data,
			bounds_checker<decltype(*gather_map)>{0, source_column.size()});
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map,
		     gather_map+num_destination_rows, source_data,
		     destination_data);
    }

    // Copy temporary buffers used for in-place gather to destination column
    if (in_place) {
      thrust::copy(rmm::exec_policy(stream)->on(stream), destination_data,
		   destination_data + num_destination_rows,
		   destination_column.data<column_type>());
    }

    CHECK_STREAM(stream);
  }

  template <typename column_type, typename iterator_type,
    std::enable_if_t<not std::is_integral<column_type>::value and
                     not std::is_floating_point<column_type>::value>* = nullptr>
  void operator()(column_view source_column,
		  iterator_type gather_map,
		  mutable_column_view destination_column,
		  bool ignore_out_of_bounds,
		  util::cuda::scoped_stream stream) {
    CUDF_FAIL("Column type must be numeric");
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
  index_converter(size_type n_rows)
  : n_rows(n_rows) {}

  __device__
  map_type operator()(map_type in) const
  {
    return ((in % n_rows) + n_rows) % n_rows;
  }
  size_type n_rows;
};

template <typename map_type>
struct index_converter<map_type, index_conversion::NONE>
{
  index_converter(size_type n_rows)
  : n_rows(n_rows) {}

  __device__
  map_type operator()(map_type in) const
  {
    return in;
  }
  size_type n_rows;
};


template <typename iterator_type>
void gather(table_view source_table, iterator_type gather_map,
	    mutable_table_view destination_table, bool check_bounds = false,
	    bool ignore_out_of_bounds = false, bool allow_negative_indices = false,
	    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource()
	    )
{
  auto source_n_cols = source_table.num_columns();
  auto source_n_rows = source_table.num_rows();

  std::vector<util::cuda::scoped_stream> v_stream(source_n_cols);

  for (size_type i = 0; i < source_n_cols; i++) {
    // Perform sanity checks
    mutable_column_view dest_col = destination_table.column(i);
    column_view src_col = source_table.column(i);

    CUDF_EXPECTS(src_col.type() == dest_col.type(), "Column type mismatch");

    // The data gather for n columns will be put on the first n streams
    cudf::experimental::type_dispatcher(src_col.type(), column_gatherer{}, src_col,
					gather_map, dest_col, ignore_out_of_bounds,
					v_stream[i]);

    if (src_col.has_nulls()) {
      CUDF_EXPECTS(dest_col.nullable(),
		   "Missing destination null mask.");
    }
  }

  rmm::device_vector<size_type> null_counts(source_n_cols, 0);

  std::vector<bitmask_type const*> source_bitmasks_host(source_n_cols);
  std::vector<bitmask_type*> destination_bitmasks_host(source_n_cols);

  std::vector<rmm::device_vector<bitmask_type>> inplace_buffers(source_n_cols);

  // loop over each column, check if inplace and allocate buffer if true.
  for (size_type i = 0; i < source_n_cols; i++) {
    mutable_column_view dest_col = destination_table.column(i);
    source_bitmasks_host[i] = source_table.column(i).null_mask();
    // Allocate inplace buffer
    if (dest_col.nullable() &&
	dest_col.null_mask() == source_table.column(i).null_mask()) {
      inplace_buffers[i].resize(bitmask_allocation_size_bytes(dest_col.size()));
      destination_bitmasks_host[i] = inplace_buffers[i].data().get();
    } else {
      destination_bitmasks_host[i] = dest_col.null_mask();
    }
  }

  // In the following we allocate the device array thats hold the valid
  // bits.
  rmm::device_vector<bitmask_type const*> source_bitmasks(source_bitmasks_host);
  rmm::device_vector<bitmask_type*> destination_bitmasks(destination_bitmasks_host);

  auto bitmask_kernel =
    ignore_out_of_bounds ? gather_bitmask_kernel<true, decltype(gather_map)> : gather_bitmask_kernel<false, decltype(gather_map)>;

  int gather_grid_size;
  int gather_block_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
	       &gather_grid_size, &gather_block_size, bitmask_kernel));

  bitmask_kernel<<<gather_grid_size, gather_block_size>>>(
      source_bitmasks.data().get(), source_table.num_rows(),
      gather_map, destination_bitmasks.data().get(), destination_table.num_rows(),
      null_counts.data().get(), source_n_cols);

  std::vector<size_type> h_count(source_n_cols);
  CUDA_TRY(cudaMemcpy(h_count.data(), null_counts.data().get(),
		      sizeof(size_type) * source_n_cols, cudaMemcpyDeviceToHost));

  // loop over each column, check if inplace and copy the result from the
  // buffer back to destination if true.
  for (size_type i = 0; i < destination_table.num_columns(); i++) {
    mutable_column_view dest_col = destination_table.column(i);
    if (dest_col.nullable()) {
      // Copy temp buffer content back to column
      if (dest_col.null_mask() == source_table.column(i).null_mask()) {
	size_type num_bitmask_elements =
	  gdf_num_bitmask_elements(dest_col.size());
	CUDA_TRY(cudaMemcpy(dest_col.null_mask(), destination_bitmasks_host[i],
			    num_bitmask_elements, cudaMemcpyDeviceToDevice));
      }
      dest_col.set_null_count(dest_col.size() - h_count[i]);
    } else {
      dest_col.set_null_count(0);
    }
  }
}


} // namespace detail
} // namespace experimental
} // namespace cudf