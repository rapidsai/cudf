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

#ifndef DEVICE_TABLE_H
#define DEVICE_TABLE_H

#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/error_utils.hpp"
#include "hash/hash_functions.cuh"
#include "hash/managed.cuh"
#include "bitmask/legacy_bitmask.hpp"
#include "utilities/type_dispatcher.hpp"
#include <utilities/error_utils.hpp>
#include <types.hpp>


#include <thrust/tabulate.h>
#include <thrust/logical.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
/**
 * @brief  Computes the validity mask for the rows in the device_table.

   If a single value in a row of the table is NULL, then the entire row is
   considered to be NULL. Therefore, we can AND all of the bitmasks of each
   column together to get a bitmask for the validity of each row.
 */
struct row_masker {
  row_masker(gdf_valid_type** column_masks, const gdf_size_type num_cols)
      : column_valid_masks{column_masks}, _num_columns(num_cols) {}

  /**
   * @brief Computes the bit-wise AND across all columns for the specified mask
   *
   * @param mask_number The index of the mask to compute the bit-wise AND across
   * all columns
   *
   * @returns The bit-wise AND across all columns for the specified mask number
   */
  __device__ __forceinline__ gdf_valid_type
  operator()(const gdf_size_type mask_number) {
    // Intialize row validity mask with all bits set to 1
    gdf_valid_type row_valid_mask{0};
    row_valid_mask = ~(row_valid_mask);

    for (gdf_size_type i = 0; i < _num_columns; ++i) {
      const gdf_valid_type* current_column_mask = column_valid_masks[i];

      // The column validity mask is optional and can be nullptr
      if (nullptr != current_column_mask) {
        row_valid_mask &= current_column_mask[mask_number];
      }
    }
    return row_valid_mask;
  }

  const gdf_size_type _num_columns;
  gdf_valid_type** column_valid_masks;
};


}  // namespace

/** 
 * @brief Provides row-level device functions for operating on a set of columns.
 */
class device_table : public managed
{
public:

  using size_type = int64_t;

  /**---------------------------------------------------------------------------*
   * @brief Factory function to construct a device_table wrapped in a
   * unique_ptr.
   *
   * Constructing a `device_table` via a factory function is required to ensure
   * that it is constructed via the `new` operator that allocates the class with
   * managed memory such that it can be accessed via pointer or reference in
   * device code. A `unique_ptr` is used to ensure the object is cleaned-up correctly.
   * 
   * Usage:
   * ```
   * gdf_column * col;
   * auto device_table_ptr = device_table::create(1, &col);
   * 
   * // Because `device_table` is allocated with managed memory, a pointer to 
   * // the object can be passed directly into device code
   * some_kernel<<<...>>>(device_table_ptr.get());
   * ```
   *
   * @param num_columns The number of columns
   * @param cols Array of columns
   * @return A unique_ptr containing a device_table object
   *---------------------------------------------------------------------------**/
  static auto create(gdf_size_type num_columns, gdf_column* cols[],
                     cudaStream_t stream = 0) {
    auto deleter = [](device_table* d) { d->destroy(); };

    std::unique_ptr<device_table, decltype(deleter)> p{
        new device_table(num_columns, cols, stream), deleter};

    int dev_id = 0;
    CUDA_TRY(cudaGetDevice(&dev_id));
    CUDA_TRY(cudaMemPrefetchAsync(p.get(), sizeof(*p), dev_id, stream))

    CHECK_STREAM(stream);

    return p;
  }

  static auto create(cudf::table& t, cudaStream_t stream = 0) {
    return device_table::create(t.num_columns(), t.begin(), stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Destroys the `device_table`.
   *
   * @note This function is required because the destructor is protected to prohibit
   * stack allocation.
   *
   *---------------------------------------------------------------------------**/
  void destroy(void){
      delete this;
  }

  device_table() = delete;
  device_table(device_table const& other) = delete;
  device_table& operator=(device_table const& other) = delete;

  /** 
   * @brief  Updates the size of the gdf_columns in the table
   * 
   * @param new_length The new length
   */
  void set_num_rows(const size_type new_length)
  {
    _num_rows = new_length;

    for(gdf_size_type i = 0; i < _num_columns; ++i)
    {
      host_columns[i]->size = this->_num_rows;
    }
  }

  gdf_size_type num_columns() const
  {
    return _num_columns;
  }

  __host__
  gdf_column * get_column(gdf_size_type column_index) const
  {
    return host_columns[column_index];
  }

   __host__ 
  gdf_column ** columns() const
  {
    return host_columns;
  }

  __host__ __device__
  gdf_size_type num_rows() const
  {
    return _num_rows;
  }

  __device__ bool row_has_nulls(gdf_size_type row_index) const{
      return not gdf_is_valid(d_row_valid, row_index);
  }

  /** 
   * @brief  Gets the size in bytes of a row in the device_table, i.e., the sum of 
   * the byte widths of all columns in the table
   * 
   * @returns The size in bytes of the row in the table
   */
  gdf_size_type get_row_size_bytes() const
  {
    return row_size_bytes;
  }

  /** 
   * @brief  Packs the elements of a specified row into a contiguous byte-buffer
   *
   * This function is called by a single thread, and the thread will copy each element
   * of the row into a single contiguous buffer. TODO: This could be done by multiple threads
   * by passing in a cooperative group. 
   * 
   * @param index The row of the table to return
   * @param row_byte_buffer A pointer to a preallocated buffer large enough to hold a 
      row of the table 
   * 
   */
  __device__
  gdf_error pack_row(gdf_size_type row_index, void * row_byte_buffer) const
  {
    if(nullptr == row_byte_buffer) {
      return GDF_DATASET_EMPTY;
    }

    // Pack the element from each column in the row into the buffer
    for(gdf_size_type i = 0; i < _num_columns; ++i)
    {
      const gdf_dtype source_col_type = d_columns_types_ptr[i];

      cudf::type_dispatcher(source_col_type,
                            copy_element{},
                            row_byte_buffer, i,
                            d_columns_data_ptr[i], row_index);

    }
    return GDF_SUCCESS;
  }

  struct copy_element{
    template <typename ColumnType>
    __device__ __forceinline__
    void operator()(void * target_column, gdf_size_type target_row_index,
                    void const * source_column, gdf_size_type source_row_index)
    {
      ColumnType& target_value { static_cast<ColumnType*>(target_column)[target_row_index] };
      ColumnType const& source_value{static_cast<ColumnType const*>(source_column)[source_row_index]};
      target_value = source_value;
    }

  };

  /**
   * @brief  Copies a row from a source table to a target row in this table
   *
   * This device function should be called by a single thread and the thread
   * will copy all of the elements in the row from one table to the other. TODO:
   * In the future, this could be done by multiple threads by passing in a
   * cooperative group.
   *
   * @param other The other table from which the row is copied
   * @param target_row_index The index of the row in this table that will be written
   * to
   * @param source_row_index The index of the row from the other table that will
   * be copied from
   */
  __device__ 
  gdf_error copy_row(device_table const & source,
                     const gdf_size_type target_row_index,
                     const gdf_size_type source_row_index)
  {
    for(gdf_size_type i = 0; i < _num_columns; ++i)
    {
      const gdf_dtype target_col_type = d_columns_types_ptr[i];
      const gdf_dtype source_col_type = source.d_columns_types_ptr[i];
    
      if(target_col_type != source_col_type)
      {
        return GDF_DTYPE_MISMATCH;
      }

      cudf::type_dispatcher(target_col_type,
                            copy_element{},
                            d_columns_data_ptr[i],
                            target_row_index,
                            source.d_columns_data_ptr[i],
                            source_row_index);

    }
    return GDF_SUCCESS;
  }


  struct elements_are_equal{
    template <typename ColumnType>
    __device__ __forceinline__ bool operator()(
        void const* lhs_data, gdf_valid_type const* lhs_bitmask,
        gdf_size_type lhs_row_index, void const* rhs_data,
        gdf_valid_type const* rhs_bitmask, gdf_size_type rhs_row_index,
        bool nulls_are_equal = false) {
    
      bool const lhs_is_valid{gdf_is_valid(lhs_bitmask, lhs_row_index)};
      bool const rhs_is_valid{gdf_is_valid(rhs_bitmask, rhs_row_index)};

      // If both values are non-null, compare them
      if (lhs_is_valid and rhs_is_valid) {
        return static_cast<ColumnType const*>(lhs_data)[lhs_row_index] ==
               static_cast<ColumnType const*>(rhs_data)[rhs_row_index];
      }

      // If both values are null
      if (not lhs_is_valid and
          not rhs_is_valid) {
        return nulls_are_equal;
      }

      // If only one value is null, they can never be equal
      return false;
    }
  };

  /**
   * @brief  Checks for equality between a target row in this table and a source
   * row in another table.
   *
   * @param rhs The other table whose row is compared to this tables
   * @param this_row_index The row index of this table to compare
   * @param rhs_row_index The row index of the rhs table to compare
   * @param nulls_are_equal Flag indicating if two null values are considered
   * equal
   *
   * @returns True if the elements in both rows are equivalent, otherwise False
   */
  __device__
  bool rows_equal(device_table const & rhs, 
                  const gdf_size_type this_row_index, 
                  const gdf_size_type rhs_row_index,
                  bool nulls_are_equal = false) const
  {
    bool const rows_have_nulls =
        this->row_has_nulls(this_row_index) or rhs.row_has_nulls(rhs_row_index);

    if (rows_have_nulls and not nulls_are_equal) {
      return false;
    }

    auto equal_elements =
        [this, &rhs, this_row_index,
         rhs_row_index, nulls_are_equal](gdf_size_type column_index) {

          bool const type_mismatch{d_columns_types_ptr[column_index] !=
                                   rhs.d_columns_types_ptr[column_index]};

          if (type_mismatch) {
            return false;
          }

          return cudf::type_dispatcher(
              d_columns_types_ptr[column_index], elements_are_equal{},
              d_columns_data_ptr[column_index],
              d_columns_valids_ptr[column_index], this_row_index,
              rhs.d_columns_data_ptr[column_index],
              rhs.d_columns_valids_ptr[column_index], rhs_row_index);
        };

    return thrust::all_of(thrust::seq, thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(_num_columns),
                          equal_elements);
  }

  template < template <typename> typename hash_function >
  struct hash_element
  {
    template <typename col_type>
    __device__ __forceinline__
    void operator()(hash_value_type& hash_value, 
                    void const * col_data,
                    gdf_size_type row_index,
                    gdf_size_type col_index,
                    bool use_initial_value = false,
                    const hash_value_type& initial_value = 0)
    {
      hash_function<col_type> hasher;
      col_type const * const current_column{static_cast<col_type const*>(col_data)};
      hash_value_type key_hash{hasher(current_column[row_index])};

      if (use_initial_value)
        key_hash = hasher.hash_combine(initial_value, key_hash);

      // Only combine hash-values after the first column
      if(0 == col_index)
        hash_value = key_hash;
      else
        hash_value = hasher.hash_combine(hash_value, key_hash);
    }
  };

  /** --------------------------------------------------------------------------*
   * @brief Device function to compute a hash value for a given row in the table
   * 
   * @param[in] row_index The row of the table to compute the hash value for
   * @param[in] num_columns_to_hash The number of columns in the row to hash. If 0,
   * hashes all columns
   * @param[in] initial_hash_values Optional initial hash values to combine with each column's hashed values
   * @tparam hash_function The hash function that is used for each element in the row,
   * as well as combine hash values
   * 
   * @return The hash value of the row
   * ----------------------------------------------------------------------------**/
  template <template <typename> class hash_function = default_hash>
  __device__ 
  hash_value_type hash_row(gdf_size_type row_index,
                           hash_value_type* initial_hash_values = nullptr,
                           gdf_size_type num_columns_to_hash = 0) const
  {
    hash_value_type hash_value{0};

    // If num_columns_to_hash is zero, hash all columns
    if(0 == num_columns_to_hash) 
    {
      num_columns_to_hash = this->_num_columns;
    }

    bool const use_initial_value{ initial_hash_values != nullptr };
    // Iterate all the columns and hash each element, combining the hash values together
    for(gdf_size_type i = 0; i < num_columns_to_hash; ++i)
    {
      gdf_dtype const current_column_type = d_columns_types_ptr[i];

      hash_value_type const initial_hash_value = (use_initial_value) ? initial_hash_values[i] : 0;
      cudf::type_dispatcher(current_column_type, 
                          hash_element<hash_function>{}, 
                          hash_value, d_columns_data_ptr[i], row_index, i,
                          use_initial_value, initial_hash_value);
    }

    return hash_value;
  }

private:

  const gdf_size_type _num_columns; /** The number of columns in the table */
  gdf_size_type _num_rows{0};     /** The number of rows in the table */

  gdf_column ** host_columns{nullptr};  /** The set of gdf_columns that this table wraps */

  rmm::device_vector<void*> device_columns_data; /** Device array of pointers to each columns data */
  void ** d_columns_data_ptr{nullptr};                  /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_valid_type*> device_columns_valids;  /** Device array of pointers to each columns validity bitmask*/
  gdf_valid_type** d_columns_valids_ptr{nullptr};                   /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_valid_type> device_row_valid;  /** Device array of bitmask for the validity of each row. */
  gdf_valid_type * d_row_valid{nullptr};                   /** Raw pointer to device array's data */

  rmm::device_vector<gdf_dtype> device_columns_types; /** Device array of each columns data type */
  gdf_dtype * d_columns_types_ptr{nullptr};                 /** Raw pointer to the device array's data */

  gdf_size_type row_size_bytes{0};
  rmm::device_vector<gdf_size_type> device_column_byte_widths;
  gdf_size_type * d_columns_byte_widths_ptr{nullptr};

protected:
 /**---------------------------------------------------------------------------*
  * @brief Constructs a new device_table object from an array of `gdf_column*`.
  *
  * This constructor is protected to require use of the device_table::create
  * factory method. This will ensure the device_table is constructed via the overloaded
  * new operator that allocates the object using managed memory.
  *
  * @param num_cols
  * @param gdf_columns
  *---------------------------------------------------------------------------**/
 device_table(size_type num_cols, gdf_column** gdf_columns, cudaStream_t stream = 0)
     : _num_columns(num_cols), host_columns(gdf_columns) {
   CUDF_EXPECTS(num_cols > 0, "Attempt to create table with zero columns.");
   CUDF_EXPECTS(nullptr != host_columns[0],
                "Attempt to create table with a null column.");
   _num_rows = host_columns[0]->size;

   // Copy pointers to each column's data, types, and validity bitmasks
   // to contiguous host vectors (AoS to SoA conversion)
   std::vector<void*> columns_data(num_cols);
   std::vector<gdf_valid_type*> columns_valids(num_cols);
   std::vector<gdf_dtype> columns_types(num_cols);
   std::vector<gdf_size_type> columns_byte_widths(num_cols);

   for (size_type i = 0; i < num_cols; ++i) {
     gdf_column* const current_column = host_columns[i];
     CUDF_EXPECTS(nullptr != current_column, "Column is null");
     CUDF_EXPECTS(_num_rows == current_column->size, "Column size mismatch");
     if (_num_rows > 0) {
       CUDF_EXPECTS(nullptr != current_column->data, "Column missing data.");
     }

     // Compute the size of a row in the table in bytes
     int column_width_bytes{0};
     if (GDF_SUCCESS ==
         get_column_byte_width(current_column, &column_width_bytes)) {
       row_size_bytes += column_width_bytes;
       // Store the byte width of each column in a device array
       columns_byte_widths[i] = row_size_bytes;
     } else {
       std::cerr << "Attempted to get column byte width of unsupported GDF "
                    "datatype.\n";
       columns_byte_widths[i] = 0;
     }

     columns_data[i] = (host_columns[i]->data);
     columns_valids[i] = (host_columns[i]->valid);
     columns_types[i] = (host_columns[i]->dtype);
   }

   // Copy host vectors to device vectors
   device_columns_data = columns_data;
   device_columns_valids = columns_valids;
   device_columns_types = columns_types;
   device_column_byte_widths = columns_byte_widths;

   d_columns_data_ptr = device_columns_data.data().get();
   d_columns_valids_ptr = device_columns_valids.data().get();
   d_columns_types_ptr = device_columns_types.data().get();
   d_columns_byte_widths_ptr = device_column_byte_widths.data().get();

   // Allocate storage sufficient to hold a validity bit for every row
   // in the table
   const size_type mask_size = gdf_valid_allocation_size(_num_rows);
   device_row_valid.resize(mask_size);

   // If a row contains a single NULL value, then the entire row is considered
   // to be NULL, therefore initialize the row-validity mask with the
   // bit-wise AND of the validity mask of all the columns
   thrust::tabulate(rmm::exec_policy(stream)->on(stream), device_row_valid.begin(),
                    device_row_valid.end(),
                    row_masker(d_columns_valids_ptr, num_cols));

   d_row_valid = device_row_valid.data().get();

   CHECK_STREAM(stream);
  }

 /**---------------------------------------------------------------------------*
  * @brief Destructor is protected to prevent stack allocation.
  *
  * The device_table class is allocated with managed memory via an overloaded
  * `new` operator.
  *
  * This requires that the `device_table` always be allocated on the heap via
  * `new`.
  *
  * Therefore, to protect users for errors, stack allocation should be
  * prohibited.
  *
  *---------------------------------------------------------------------------**/
 ~device_table() = default;

};



#endif
