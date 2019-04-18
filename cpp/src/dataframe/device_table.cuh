/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cudf_utils.h>
#include <bitmask/legacy_bitmask.hpp>
#include <hash/hash_functions.cuh>
#include <hash/managed.cuh>
#include <types.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>

/**
 * @brief Flattens a gdf_column array-of-structs into a struct-of-arrays and
 * provides row-level device functions.
 *
 * @note Because the class is allocated with managed memory, instances of
 * `device_table` can be passed directly via pointer or reference into device
 * code.
 *
 */
class device_table : public managed {
 public:
  using size_type = int64_t;

  /**---------------------------------------------------------------------------*
   * @brief Factory function to construct a device_table wrapped in a
   * unique_ptr.
   *
   * Constructing a `device_table` via a factory function is required to ensure
   * that it is constructed via the `new` operator that allocates the class with
   * managed memory such that it can be accessed via pointer or reference in
   * device code. A `unique_ptr` is used to ensure the object is cleaned-up
   * correctly.
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
   * @note This function is required because the destructor is protected to
   *prohibit stack allocation.
   *
   *---------------------------------------------------------------------------**/
  void destroy(void) { delete this; }

  device_table() = delete;
  device_table(device_table const& other) = delete;
  device_table& operator=(device_table const& other) = delete;

  /**
   * @brief  Updates the size of the gdf_columns in the table
   *
   * @note Does NOT shrink the column's allocations to the new size.
   *
   * @param new_length The new length
   */
  void set_num_rows(const size_type new_length) {
    _num_rows = new_length;

    for (gdf_size_type i = 0; i < _num_columns; ++i) {
      host_columns[i]->size = this->_num_rows;
    }
  }

  __host__ __device__ gdf_size_type num_columns() const { return _num_columns; }

  __host__ gdf_column* get_column(gdf_size_type column_index) const {
    return host_columns[column_index];
  }

  __device__ gdf_column const* device_column(gdf_size_type index) const {
    return &device_columns[index];
  }
  __device__ gdf_column* device_column(gdf_size_type index) {
    return &device_columns[index];
  }

  __device__ gdf_column const* begin() const { return device_columns; }
  __device__ gdf_column* begin() { return device_columns; }

  __device__ gdf_column const* end() const {
    return device_columns + _num_columns;
  }
  __device__ gdf_column* end() { return device_columns + _num_columns; }

  __host__ gdf_column** columns() const { return host_columns; }

  __host__ __device__ gdf_size_type num_rows() const { return _num_rows; }

  const gdf_size_type _num_columns; /** The number of columns in the table */
  gdf_size_type _num_rows{0};       /** The number of rows in the table */

  gdf_column** host_columns{
      nullptr}; /** The set of gdf_columns that this table wraps */

  rmm::device_vector<void*>
      device_columns_data; /** Device array of pointers to each column's data */
  void** d_columns_data_ptr{
      nullptr}; /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_valid_type*>
      device_columns_valids; /** Device array of pointers to each column's
                                validity bitmask*/
  gdf_valid_type** d_columns_valids_ptr{
      nullptr}; /** Raw pointer to the device array's data */

  rmm::device_vector<gdf_dtype>
      device_columns_types; /** Device array of each column's data type */
  gdf_dtype* column_types{
      nullptr}; /** Raw pointer to the device array's data */

  gdf_column* device_columns{nullptr};

 protected:
  /**---------------------------------------------------------------------------*
   * @brief Constructs a new device_table object from an array of `gdf_column*`.
   *
   * This constructor is protected to require use of the device_table::create
   * factory method. This will ensure the device_table is constructed via the
   *overloaded new operator that allocates the object using managed memory.
   *
   * @param num_cols
   * @param gdf_columns
   *---------------------------------------------------------------------------**/
  device_table(size_type num_cols, gdf_column** gdf_columns,
               cudaStream_t stream = 0)
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

    std::vector<gdf_column> temp_columns(num_cols);

    for (size_type i = 0; i < num_cols; ++i) {
      gdf_column* const current_column = host_columns[i];
      CUDF_EXPECTS(nullptr != current_column, "Column is null");
      CUDF_EXPECTS(_num_rows == current_column->size, "Column size mismatch");
      if (_num_rows > 0) {
        CUDF_EXPECTS(nullptr != current_column->data, "Column missing data.");
      }

      columns_data[i] = (host_columns[i]->data);
      columns_valids[i] = (host_columns[i]->valid);
      columns_types[i] = (host_columns[i]->dtype);

      temp_columns[i] = *gdf_columns[i];
    }

    // Copy columns to device
    RMM_ALLOC(&device_columns, num_cols * sizeof(gdf_column), stream);
    CUDA_TRY(cudaMemcpyAsync(device_columns, temp_columns.data(),
                             num_cols * sizeof(gdf_column),
                             cudaMemcpyHostToDevice, stream));

    // Copy host vectors to device vectors
    device_columns_data = columns_data;
    device_columns_valids = columns_valids;
    device_columns_types = columns_types;

    d_columns_data_ptr = device_columns_data.data().get();
    d_columns_valids_ptr = device_columns_valids.data().get();
    column_types = device_columns_types.data().get();

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

namespace {
struct elements_are_equal {
  template <typename ColumnType>
  __device__ __forceinline__ bool operator()(gdf_column const& lhs,
                                             gdf_size_type lhs_index,
                                             gdf_column const& rhs,
                                             gdf_size_type rhs_index,
                                             bool nulls_are_equal = false) {
    bool const lhs_is_valid{gdf_is_valid(lhs.valid, lhs_index)};
    bool const rhs_is_valid{gdf_is_valid(rhs.valid, rhs_index)};

    // If both values are non-null, compare them
    if (lhs_is_valid and rhs_is_valid) {
      return static_cast<ColumnType const*>(lhs.data)[lhs_index] ==
             static_cast<ColumnType const*>(rhs.data)[rhs_index];
    }

    // If both values are null
    if (not lhs_is_valid and not rhs_is_valid) {
      return nulls_are_equal;
    }

    // If only one value is null, they can never be equal
    return false;
  }
};
}  // namespace

/**
 * @brief  Checks for equality between two rows between two tables.
 *
 * @param lhs The left table
 * @param lhs_index The index of the row in the rhs table to compare
 * @param rhs The right table
 * @param rhs_index The index of the row within rhs table to compare
 * @param nulls_are_equal Flag indicating whether two null values are considered
 * equal
 *
 * @returns true If the two rows are element-wise equal
 * @returns false If any element differs between the two rows
 */
__device__ inline bool rows_equal(device_table const& lhs,
                                  const gdf_size_type lhs_index,
                                  device_table const& rhs,
                                  const gdf_size_type rhs_index,
                                  bool nulls_are_equal = false) {
  auto equal_elements = [lhs_index, rhs_index, nulls_are_equal](
                            gdf_column const& l, gdf_column const& r) {
    return cudf::type_dispatcher(l.dtype, elements_are_equal{}, l, lhs_index, r,
                                 rhs_index, nulls_are_equal);
  };

  return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(),
                       equal_elements);
}

namespace {
template <template <typename> typename hash_function>
struct hash_element {
  template <typename col_type>
  __device__ __forceinline__ void operator()(
      hash_value_type& hash_value, void const* col_data,
      gdf_size_type row_index, gdf_size_type col_index,
      bool use_initial_value = false,
      const hash_value_type& initial_value = 0) {
    hash_function<col_type> hasher;
    col_type const* const current_column{
        static_cast<col_type const*>(col_data)};
    hash_value_type key_hash{hasher(current_column[row_index])};

    if (use_initial_value)
      key_hash = hasher.hash_combine(initial_value, key_hash);

    // Only combine hash-values after the first column
    if (0 == col_index)
      hash_value = key_hash;
    else
      hash_value = hasher.hash_combine(hash_value, key_hash);
  }
};
}  // namespace

/**
 * --------------------------------------------------------------------------*
 * @brief Computes the hash value for a row in a table
 *
 * @note NULL values are ignored. If a row contains all NULL values, it will
 * return a hash value of 0.
 *
 * @param[in] row_index The row of the table to compute the hash value for
 * @param[in] num_columns_to_hash The number of columns in the row to hash. If
 * 0, hashes all columns
 * @param[in] initial_hash_values Optional initial hash values to combine with
 * each column's hashed values
 * @tparam hash_function The hash function that is used for each element in
 * the row, as well as combine hash values
 *
 * @return The hash value of the row
 * ----------------------------------------------------------------------------**/
template <template <typename> class hash_function = default_hash>
__device__ inline hash_value_type hash_row(
    device_table const& t, gdf_size_type row_index,
    hash_value_type* initial_hash_values = nullptr,
    gdf_size_type num_columns_to_hash = 0) {
  hash_value_type hash_value{0};

  // If num_columns_to_hash is zero, hash all columns
  if (0 == num_columns_to_hash) {
    num_columns_to_hash = t.num_columns();
  }

  bool const use_initial_value{initial_hash_values != nullptr};

  // Iterate all the columns and hash each element, combining the hash values
  // together
  for (gdf_size_type i = 0; i < num_columns_to_hash; ++i) {
    // Skip null values
    if (gdf_is_valid(t.d_columns_valids_ptr[i], row_index)) {
      hash_value_type const initial_hash_value =
          (use_initial_value) ? initial_hash_values[i] : 0;
      cudf::type_dispatcher(t.column_types[i], hash_element<hash_function>{},
                            hash_value, t.d_columns_data_ptr[i], row_index, i,
                            use_initial_value, initial_hash_value);
    }
  }
  return hash_value;
}

namespace {
struct copy_element {
  template <typename T>
  __device__ inline void operator()(gdf_column& target,
                                    gdf_size_type target_index,
                                    gdf_column const& source,
                                    gdf_size_type source_index) {
    static_cast<T*>(target.data)[target_index] =
        static_cast<T const*>(source.data)[source_index];
  }
};
}  // namespace

/**
 * @brief  Copies a row from a source table to a target table.
 *
 * This device function should be called by a single thread and the thread
 * will copy all of the elements in the row from one table to the other.
 *
 * @note This function do not guard against race conditions in either the source
 * or target row.
 *
 * FIXME: Does NOT set null bitmask for the target row.
 *
 * @param other The other table from which the row is copied
 * @param target_row_index The index of the row in this table that will be
 * written to
 * @param source_row_index The index of the row from the other table that will
 * be copied from
 */
__device__ inline void copy_row(device_table& target,
                                gdf_size_type target_index,
                                device_table const& source,
                                gdf_size_type source_index) {
  for (gdf_size_type i = 0; i < target.num_columns(); ++i) {
    cudf::type_dispatcher(target.device_column(i)->dtype, copy_element{},
                          *target.device_column(i), target_index,
                          *source.device_column(i), source_index);
  }
}

#endif
