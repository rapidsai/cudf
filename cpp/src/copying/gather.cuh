#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <utilities/bit_util.cuh>
#include <utilities/cudf_utils.h>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>


#include <cub/cub.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <utilities/release_assert.cuh>

#include <bitmask/legacy/valid_if.cuh>

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

  __device__ bool operator()(map_type const index) {
    return ((index >= begin) && (index < end));
  }
};


template <bool ignore_out_of_bounds, typename map_iterator_type>
__global__ void gather_bitmask_kernel(table_device_view source_table,
				      map_iterator_type gather_map,
				      mutable_table_device_view destination_table) {
  for (size_type i = 0; i < source_table.num_columns(); i++) {

    const int warp_size = 32;

    column_device_view source_col = source_table.column(i);
    mutable_column_device_view destination_col = destination_table.column(i);

    if (source_col.has_nulls()) {
      size_type destination_row_base = blockIdx.x * blockDim.x;

      while (destination_row_base < destination_table.num_rows()) {
	size_type destination_row = destination_row_base + threadIdx.x;

	const bool thread_active = destination_row < destination_col.size();
	size_type source_row =
	  thread_active ? gather_map[destination_row] : 0;

	const uint32_t active_threads =
	  __ballot_sync(0xffffffff, thread_active);

	bool source_bit_is_valid = source_col.has_nulls()
	  ? source_col.is_valid_nocheck(source_row)
	  : true;

	// Use ballot to find all valid bits in this warp and create the output
	// bitmask element
	const uint32_t valid_warp =
	  __ballot_sync(active_threads, source_bit_is_valid);

	const size_type valid_index =
	  cudf::util::detail::bit_container_index<bitmask_type>(
	      destination_row);

	// Only one thread writes output
	if (0 == threadIdx.x % warp_size && thread_active) {
	  destination_col.set_mask_word(valid_index, valid_warp);
	}
	destination_row_base += blockDim.x * gridDim.x;
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
   * @tparam Element Dispatched type for the column being gathered
   * @param source_column The column to gather from
   * @param gather_map An iterator over integral values representing the gather map
   * @param destination_column The column to gather into
   * @param ignore_out_of_bounds Ignore values in `gather_map` that are
   * out of bounds
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename Element, typename MapIterator,
    std::enable_if_t<is_fixed_width<Element>()>* = nullptr>
    std::unique_ptr<column> operator()(column_view const& source_column,
				       MapIterator gather_map,
				       size_type num_destination_rows,
				       bool ignore_out_of_bounds,
				       cudaStream_t stream) {

    std::unique_ptr<column> destination_column =
      allocate_like(source_column, num_destination_rows);


    Element const *source_data{source_column.data<Element>()};
    Element *destination_data{destination_column->mutable_view().data<Element>()};

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

    CHECK_STREAM(stream);
    return destination_column;
  }

  template <typename Element, typename MapIterator,
    std::enable_if_t<not is_fixed_width<Element>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source_column,
				     MapIterator gather_map,
				     size_type num_destination_rows,
				     bool ignore_out_of_bounds,
				     util::cuda::scoped_stream stream) {
    CUDF_FAIL("Column type must be numeric");
  }

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
template <typename map_type>
struct index_converter : public thrust::unary_function<map_type,map_type>
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


template <typename iterator_type>
std::unique_ptr<table>
gather(table_view const& source_table, iterator_type gather_map,
       size_type num_destination_rows, bool check_bounds = false,
       bool ignore_out_of_bounds = false,
       bool allow_negative_indices = false,
       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  auto source_n_cols = source_table.num_columns();
  auto source_n_rows = source_table.num_rows();

  std::vector<util::cuda::scoped_stream> v_stream(source_n_cols);

  std::vector<std::unique_ptr<column>> destination_columns;

  for (size_type i = 0; i < source_n_cols; i++) {
    column_view src_col = source_table.column(i);
    // Perform sanity checks
    CUDF_EXPECTS(src_col.data<void*>() != nullptr, "Missing source data buffer");

    // The data gather for n columns will be put on the first n streams
    destination_columns.push_back(
				  cudf::experimental::type_dispatcher(src_col.type(),
								      column_gatherer{},
								      src_col,
								      gather_map,
								      num_destination_rows,
								      ignore_out_of_bounds,
								      v_stream[i]));
  }


  std::unique_ptr<table> destination_table = std::make_unique<table>(std::move(destination_columns));

  auto bitmask_kernel =
    ignore_out_of_bounds ? gather_bitmask_kernel<true, decltype(gather_map)> : gather_bitmask_kernel<false, decltype(gather_map)>;

  int gather_grid_size;
  int gather_block_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
	       &gather_grid_size, &gather_block_size, bitmask_kernel));


  auto source_table_view = table_device_view::create(source_table);
  auto destination_table_view = mutable_table_device_view::create(destination_table->mutable_view());

  bitmask_kernel<<<gather_grid_size, gather_block_size>>>(
							  *source_table_view,
							  gather_map,
							  *destination_table_view);


  mutable_table_view dest_view = destination_table->mutable_view();

  // set null_count to UNKNOWN_NULL_COUNT
  for (size_type i = 0; i < destination_table->num_columns(); i++) {
    mutable_column_view dest_col_view = dest_view.column(i);
    dest_col_view.set_null_count(UNKNOWN_NULL_COUNT);
  }

  return destination_table;
}


} // namespace detail
} // namespace experimental
} // namespace cudf