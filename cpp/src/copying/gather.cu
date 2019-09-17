/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <utilities/bit_util.cuh>
#include <utilities/cudf_utils.h>
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
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include <bitmask/valid_if.cuh>

using bit_mask::bit_mask_t;

namespace cudf {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Function object to check if an index is within the bounds [begin,
 * end).
 *
 *---------------------------------------------------------------------------**/
template <typename MapType>
struct bounds_checker {
  gdf_index_type begin;
  gdf_index_type end;

  __device__ bounds_checker(gdf_index_type begin_, gdf_index_type end_)
      : begin{begin_}, end{end_} {}

  __device__ __forceinline__ bool operator()(MapType const index) {
    return ((index >= begin) && (index < end));
  }
};


template <bool ignore_out_of_bounds, typename MapIterator>
__global__ void gather_bitmask_kernel(const bit_mask_t *const *source_valid,
                                      gdf_size_type num_source_rows,
                                      MapIterator gather_map,
                                      bit_mask_t **destination_valid,
                                      gdf_size_type num_destination_rows,
                                      gdf_size_type *d_count,
                                      gdf_size_type num_columns) {
  for (gdf_index_type i = 0; i < num_columns; i++) {
    const bit_mask_t *__restrict__ source_valid_col = source_valid[i];
    bit_mask_t *__restrict__ destination_valid_col = destination_valid[i];

    const bool src_has_nulls = source_valid_col != nullptr;
    const bool dest_has_nulls = destination_valid_col != nullptr;

    if (dest_has_nulls) {
      gdf_index_type destination_row_base = blockIdx.x * blockDim.x;

      gdf_size_type valid_count_accumulate = 0;

      while (destination_row_base < num_destination_rows) {
        gdf_index_type destination_row = destination_row_base + threadIdx.x;

        const bool thread_active = destination_row < num_destination_rows;
        gdf_index_type source_row =
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

        const gdf_index_type valid_index =
            cudf::util::detail::bit_container_index<bit_mask_t>(
                destination_row);
        // Only one thread writes output
        if (0 == threadIdx.x % warp_size && thread_active) {
          destination_valid_col[valid_index] = valid_warp;
        }
        valid_count_accumulate += single_lane_popc_block_reduce(valid_warp);

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
   * @tparam ColumnType Dispatched type for the column being gathered
   * @param source_column The column to gather from
   * @param gather_map Array of indices that maps source elements to destination
   * elements
   * @param destination_column The column to gather into
   * @param check_bounds Optionally perform bounds checking on the values of
   * `gather_map`
   * @param ignore_out_of_bounds Ignore values in `gather_map` that are
   * out of bounds
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename ColumnType, typename MapIterator>
  void operator()(gdf_column const *source_column,
		  MapIterator gather_map,
		  gdf_column *destination_column, bool ignore_out_of_bounds,
		  cudaStream_t stream, bool sync_nvstring_category = false) {
    ColumnType const *source_data{
      static_cast<ColumnType const *>(source_column->data)};
    ColumnType *destination_data{
      static_cast<ColumnType *>(destination_column->data)};

    gdf_size_type const num_destination_rows{destination_column->size};

    // If gathering in-place or scattering nvstring
    // (in which case the sync_nvstring_category should be set to true)
    // allocate temporary buffers to hold intermediate results
    bool const sync_category =
      std::is_same<ColumnType, nvstring_category>::value &&
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

      source_data = static_cast<ColumnType *>(temp_src.data);
    }

    rmm::device_vector<ColumnType> in_place_buffer;
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
		   static_cast<ColumnType *>(destination_column->data));
    }

    CHECK_STREAM(stream);
  }
};

/**---------------------------------------------------------------------------*
* @brief Function object for applying a transformation on the gathermap
* that converts negative indices to positive indices
*---------------------------------------------------------------------------**/
template <typename MapType>
struct map_transform : public thrust::unary_function<MapType,MapType>
{
  map_transform(gdf_size_type n_rows, bool transform_negative_indices)
    : n_rows(n_rows), transform_negative_indices(transform_negative_indices){}

  __device__
  MapType operator()(MapType in) const
  {
    if (transform_negative_indices)
      return ((in % n_rows) + n_rows) % n_rows;
    else
      return in;
  }
  gdf_size_type n_rows;
  bool transform_negative_indices;
};


struct dispatch_map_type {

  template <typename MapType, std::enable_if_t<std::is_integral<MapType>::value>* = nullptr>
  void operator()(table const *source_table, gdf_column gather_map,
		  table *destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool sync_nvstring_category = false,
		  bool transform_negative_indices = false) {

    auto source_n_cols = source_table->num_columns();
    auto source_n_rows = source_table->num_rows();

    std::vector<util::cuda::scoped_stream> v_stream(source_n_cols);

    MapType const * typed_gather_map = static_cast<MapType const*>(gather_map.data);

    if (check_bounds) {

      gdf_index_type begin = (transform_negative_indices) ? -source_n_rows : 0;
      CUDF_EXPECTS(
	  destination_table->num_rows() == thrust::count_if(
	      rmm::exec_policy()->on(0),
	      typed_gather_map,
	      typed_gather_map + destination_table->num_rows(),
	      bounds_checker<MapType>{begin, source_n_rows}),
	  "Index out of bounds.");
    }

    auto gather_map_iterator = thrust::make_transform_iterator(
	typed_gather_map,
	map_transform<MapType>{source_n_rows, transform_negative_indices});

    for (gdf_size_type i = 0; i < source_n_cols; i++) {
      // Perform sanity checks
      gdf_column *dest_col = destination_table->get_column(i);
      const gdf_column *src_col = source_table->get_column(i);

      CUDF_EXPECTS(src_col->dtype == dest_col->dtype, "Column type mismatch");

      CUDF_EXPECTS(dest_col->data != nullptr, "Missing source data buffer.");

      // The data gather for n columns will be put on the first n streams
      cudf::type_dispatcher(src_col->dtype, column_gatherer{}, src_col,
			    gather_map_iterator, dest_col,
			    ignore_out_of_bounds, v_stream[i],
			    sync_nvstring_category);

      if (cudf::has_nulls(*src_col)) {
	CUDF_EXPECTS(cudf::is_nullable(*dest_col),
		     "Missing destination null mask.");
      }
    }

    rmm::device_vector<gdf_size_type> null_counts(source_n_cols, 0);

    std::vector<bit_mask_t*> source_bitmasks_host(source_n_cols);
    std::vector<bit_mask_t*> destination_bitmasks_host(source_n_cols);

    std::vector<rmm::device_vector<bit_mask_t>> inplace_buffers(source_n_cols);

    // loop over each column, check if inplace and allocate buffer if true.
    for (gdf_size_type i = 0; i < source_n_cols; i++) {
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
      ignore_out_of_bounds ? gather_bitmask_kernel<true, decltype(gather_map_iterator)> : gather_bitmask_kernel<false, decltype(gather_map_iterator)>;

    int gather_grid_size;
    int gather_block_size;
    CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
		 &gather_grid_size, &gather_block_size, bitmask_kernel));

    bitmask_kernel<<<gather_grid_size, gather_block_size>>>(
	source_bitmasks.data().get(), source_table->num_rows(),
	gather_map_iterator,
	destination_bitmasks.data().get(), destination_table->num_rows(),
	null_counts.data().get(), source_n_cols);

    std::vector<gdf_size_type> h_count(source_n_cols);
    CUDA_TRY(cudaMemcpy(h_count.data(), null_counts.data().get(),
			sizeof(gdf_size_type) * source_n_cols, cudaMemcpyDeviceToHost));

    // loop over each column, check if inplace and copy the result from the
    // buffer back to destination if true.
    for (gdf_size_type i = 0; i < destination_table->num_columns(); i++) {
      gdf_column *dest_col = destination_table->get_column(i);
      if (is_nullable(*dest_col)) {
	// Copy temp buffer content back to column
	if (dest_col->valid == source_table->get_column(i)->valid) {
	  gdf_size_type num_bitmask_elements =
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

  template <typename MapType, std::enable_if_t<not std::is_integral<MapType>::value>* = nullptr>
  void operator()(table const *source_table, gdf_column const gather_map,
                  table *destination_table, bool check_bounds,
		  bool ignore_out_of_bounds, bool sync_nvstring_category = false,
		  bool transform_negative_indices = false) {
   CUDF_FAIL("Gather map must be an integral type.");
  }
};


void gather(table const *source_table, gdf_column const gather_map,
            table *destination_table, bool check_bounds, bool ignore_out_of_bounds,
            bool sync_nvstring_category, bool transform_negative_indices) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != gather_map.data, "gather_map data is null");
  CUDF_EXPECTS(cudf::has_nulls(gather_map) == false, "gather_map contains nulls");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
               "Mismatched number of columns");

  cudf::type_dispatcher(gather_map.dtype, dispatch_map_type{},
			source_table, gather_map, destination_table, check_bounds,
			ignore_out_of_bounds,
			sync_nvstring_category, transform_negative_indices);
}

void gather(table const *source_table, gdf_index_type const gather_map[],
	    table *destination_table, bool check_bounds, bool ignore_out_of_bounds,
	    bool sync_nvstring_category, bool transform_negative_indices) {
  gdf_column gather_map_column{};
  gdf_column_view(&gather_map_column,
		  const_cast<gdf_index_type*>(gather_map),
		  nullptr,
		  destination_table->num_rows(),
		  gdf_dtype_of<gdf_index_type>());
  gather(source_table, gather_map_column, destination_table, check_bounds,
	 ignore_out_of_bounds, sync_nvstring_category,
	 transform_negative_indices);
}


} // namespace detail

table gather(table const *source_table, gdf_column const gather_map, bool check_bounds) {
  table destination_table = cudf::allocate_like(*source_table,
						gather_map.size);
  detail::gather(source_table, gather_map, &destination_table,
		 check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, destination_table);
  return destination_table;
}

void gather(table const *source_table, gdf_column const gather_map,
	    table *destination_table, bool check_bounds) {
  detail::gather(source_table, gather_map, destination_table, check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, *destination_table);
}

void gather(table const *source_table, gdf_index_type const gather_map[],
	    table *destination_table, bool check_bounds) {
  detail::gather(source_table, gather_map, destination_table, check_bounds, false, false, true);
  nvcategory_gather_table(*source_table, *destination_table);
}

} // namespace cudf

