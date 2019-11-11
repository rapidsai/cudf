#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <algorithm>

#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>

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




template <bool ignore_out_of_bounds, typename MapIterator>
__global__ void gather_bitmask_kernel(table_device_view source_table,
                                      MapIterator gather_map,
                                      mutable_table_device_view destination_table,
                                      size_type* valid_counts) {

  for (size_type i = 0; i < source_table.num_columns(); i++) {

    constexpr int warp_size = 32;

    column_device_view source_col = source_table.column(i);
    mutable_column_device_view destination_col = destination_table.column(i);

    if (source_col.has_nulls()) {
      size_type destination_row_base = blockIdx.x * blockDim.x;
      cudf::size_type valid_count_accumulate = 0;

      while (destination_row_base < destination_table.num_rows()) {
        size_type destination_row = destination_row_base + threadIdx.x;

        const bool thread_active = destination_row < destination_col.size();
        size_type source_row =
          thread_active ? gather_map[destination_row] : 0;

        bool source_bit_is_valid = source_col.has_nulls()
          ? source_col.is_valid_nocheck(source_row)
          : true;

        // Use ballot to find all valid bits in this warp and create the output
        // bitmask element
        const uint32_t valid_warp =
          __ballot_sync(0xffffffff, thread_active && source_bit_is_valid);

        const size_type valid_index = word_index(destination_row);

        // Only one thread writes output
        if (0 == threadIdx.x % warp_size) {
          destination_col.set_mask_word(valid_index, valid_warp);
        }
        valid_count_accumulate += single_lane_block_popc_reduce(valid_warp);
        destination_row_base += blockDim.x * gridDim.x;
      }
      if (threadIdx.x == 0) {
        atomicAdd(valid_counts + i, valid_count_accumulate);
      }
    }
  }
}


/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * column. To be used with the cudf::type_dispatcher.
 *
 *---------------------------------------------------------------------------**/
struct column_gatherer
{
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam Element Dispatched type for the column being gathered
   * @tparam MapIterator Iterator type for the gather map
   * @param source_column View into the column to gather from
   * @param gather_map_begin Beginning of iterator range of integral values representing the gather map
   * @param gather_map_end End of iterator range of integral values representing the gather map
   * @param ignore_out_of_bounds Ignore values in `gather_map` that are out of bounds
   * @param mr Memory resource to use for all allocations
   * @param stream CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename Element, typename MapIterator,
    std::enable_if_t<is_fixed_width<Element>()>* = nullptr>
    std::unique_ptr<column> operator()(column_view const& source_column,
                                       MapIterator gather_map_begin,
                                       MapIterator gather_map_end,
                                       bool ignore_out_of_bounds,
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream) {

    auto num_destination_rows = std::distance(gather_map_begin, gather_map_end);
    std::unique_ptr<column> destination_column =
      allocate_like(source_column, num_destination_rows,
                    cudf::experimental::mask_allocation_policy::RETAIN, mr);

    Element const *source_data{source_column.data<Element>()};
    Element *destination_data{destination_column->mutable_view().data<Element>()};

    using map_type = typename std::iterator_traits<MapIterator>::value_type;

    if (ignore_out_of_bounds) {
      thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map_begin,
                        gather_map_end, gather_map_begin,
                        source_data, destination_data,
                        bounds_checker<map_type>{0, source_column.size()});
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map_begin,
                     gather_map_end, source_data, destination_data);
    }

    return destination_column;
  }

  template <typename Element, typename MapIterator,
    std::enable_if_t<not is_fixed_width<Element>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source_column,
                                     MapIterator gather_map_begin,
                                     MapIterator gather_map_end,
                                     bool ignore_out_of_bounds,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {
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


/**
 * @brief Gathers the specified rows of a set of columns according to a gather map.
 *
 * Gathers the rows of the source columns according to `gather_map` such that row "i"
 * in the resulting table's columns will contain row "gather_map[i]" from the source columns.
 * The number of rows in the result table will be equal to the number of elements in
 * `gather_map`.
 *
 * A negative value `i` in the `gather_map` is interpreted as `i+n`, where
 * `n` is the number of rows in the `source_table`.
 *
 * @throws `cudf::logic_error` if `check_bounds == true` and an index exists in
 * `gather_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the source table. If `check_bounds == false`, the behavior is undefined.
 *
 * tparam MapIterator Iterator type for the gather map
 * @param[in] source_table View into the table containing the input columns whose rows will be gathered
 * @param[in] gather_map_begin Beginning of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] gather_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the destination columns
 * @param[in] check_bounds Optionally perform bounds checking on the values of `gather_map` and throw
 * an error if any of its values are out of bounds.
 * @param[in] ignore_out_of_bounds Ignore values in `gather_map` that are out of bounds. Currently
 * incompatible with `allow_negative_indices`, i.e., setting both to `true` is undefined.
 * @param[in] allow_negative_indices Interpret each negative index `i` in the gathermap as the
 * positive index `i+num_source_rows`.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The CUDA stream on which to execute kernels
 * @return cudf::table Result of the gather
 */
template <typename MapIterator>
std::unique_ptr<table>
gather(table_view const& source_table, MapIterator gather_map_begin,
       MapIterator gather_map_end, bool check_bounds = false,
       bool ignore_out_of_bounds = false,
       bool allow_negative_indices = false,
       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
       cudaStream_t stream = 0) {
  auto source_n_cols = source_table.num_columns();
  auto source_n_rows = source_table.num_rows();
  auto num_destination_rows = std::distance(gather_map_begin, gather_map_end);

  std::vector<std::unique_ptr<column>> destination_columns;

  // TODO: Could be beneficial to use streams internally here
  
  for(auto const& source_column : source_table) {
    // The data gather for n columns will be put on the first n streams
    destination_columns.push_back(
                                  cudf::experimental::type_dispatcher(source_column.type(),
                                                                      column_gatherer{},
                                                                      source_column,
                                                                      gather_map_begin,
                                                                      gather_map_end,
                                                                      ignore_out_of_bounds,
                                                                      mr,
                                                                      stream));

  }

  std::unique_ptr<table> destination_table = std::make_unique<table>(std::move(destination_columns));

  rmm::device_vector<cudf::size_type> valid_counts(source_table.num_columns(), 0);

  auto bitmask_kernel =
    ignore_out_of_bounds ? gather_bitmask_kernel<true, decltype(gather_map_begin)> : gather_bitmask_kernel<false, decltype(gather_map_begin)>;

  int gather_grid_size;
  int gather_block_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
               &gather_grid_size, &gather_block_size, bitmask_kernel));


  auto source_table_view = table_device_view::create(source_table);
  auto destination_table_view = mutable_table_device_view::create(destination_table->mutable_view());

  bitmask_kernel<<<gather_grid_size, gather_block_size, 0, stream>>>(*source_table_view,
                                                          gather_map_begin,
                                                          *destination_table_view,
                                                          valid_counts.data().get());

  thrust::host_vector<cudf::size_type> h_valid_counts(valid_counts);

  for (auto i=0; i<destination_table->num_columns(); ++i) {
    if (destination_table->get_column(i).nullable()) {
      destination_table->get_column(i).set_null_count(destination_table->num_rows()
                                                      - h_valid_counts[i]);
    }
  }

  return destination_table;
}


} // namespace detail
} // namespace experimental
} // namespace cudf
