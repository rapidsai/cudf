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
#include <table/table.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>

/**
 * @brief Lightweight wrapper for a device array of `gdf_column`s of the same
 * size
 *
 */
class device_table {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Factory function to construct a device_table wrapped in a
   * unique_ptr.
   *
   * Because this class allocates device memory, and we want to be sure that
   * device memory is free'd, this factory function returns a device_table
   * wrapped in a unique_ptr with a custom deleter that frees the device memory.
   *
   * The class' destructor does **not** free the device memory, because we would
   * like to be able to pass instances of this class by value into kernels via a
   * shallow-copy (i.e., just copying the pointers without allocating any new
   * device memory). Since it is shallow copied, if the destructor were to free
   * the device memory, then you would end up trying to free the underlying
   * device memory any time a copy is destroyed.
   *
   * Instead, the underlying device memory will not be free'd until the returned
   * `unique_ptr` invokes its deleter.
   *
   * The methods of this class with `stream` parameters are asynchronous with
   *respect to other CUDA streams and do not synchronize `stream`. Usage:
   * ```
   * gdf_column * col;
   * auto device_table_ptr = device_table::create(1, &col);
   *
   * // Table is passed by **value**, i.e., shallow copy into kernel
   * some_kernel<<<...>>>(*device_table_ptr);
   * ```
   *
   * @param[in] num_columns The number of columns
   * @param[in] cols Array of columns
   * @param[in] stream CUDA stream to use for device operations
   * @return A unique_ptr containing a device_table object
   *---------------------------------------------------------------------------**/
  static auto create(gdf_size_type num_columns, gdf_column* cols[],
                     cudaStream_t stream = 0) {
    auto deleter = [](device_table* d) { d->destroy(); };

    std::unique_ptr<device_table, decltype(deleter)> p{
        new device_table(num_columns, cols, stream), deleter};

    CHECK_STREAM(stream);

    return p;
  }

  /**---------------------------------------------------------------------------*
   * @brief Create a device_table from a `cudf::table`
   *
   * @param[in] t The `cudf::table` to wrap
   * @param[in] stream The stream to use for allocations/frees
   * @return A unique_ptr containing a device_table object
   *---------------------------------------------------------------------------**/
  static auto create(cudf::table const& t, cudaStream_t stream = 0) {
    return device_table::create(t.num_columns(), t.columns(), stream);
  }

  /**---------------------------------------------------------------------------*
   * @brief Destroys the `device_table`.
   *
   * Frees the underlying device memory and deletes the object.
   *
   * This function is invoked by the deleter of the
   * unique_ptr returned from device_table::create.
   *
   *---------------------------------------------------------------------------**/
  __host__ void destroy(void) {
    RMM_FREE(device_columns, _stream);
    delete this;
  }

  device_table() = delete;
  device_table(device_table const& other) = default;
  device_table& operator=(device_table const& other) = default;
  ~device_table() = default;

  __device__ gdf_column const* get_column(gdf_size_type index) const {
    return &device_columns[index];
  }
  __device__ gdf_column const* begin() const { return device_columns; }

  __device__ gdf_column const* end() const {
    return device_columns + _num_columns;
  }

  __host__ __device__ gdf_size_type num_columns() const { return _num_columns; }
  __host__ __device__ gdf_size_type num_rows() const { return _num_rows; }

 private:
  gdf_size_type _num_columns;  ///< The number of columns in the table
  gdf_size_type _num_rows{0};  ///< The number of rows in the table
  gdf_column* device_columns{
      nullptr};  ///< Array of `gdf_column`s in device memory
  cudaStream_t
      _stream;  ///< Stream used to allocate/free the table's device memory

 protected:
  /**---------------------------------------------------------------------------*
   * @brief Constructs a new device_table object from an array of `gdf_column*`.
   *
   * This constructor is protected to require use of the device_table::create
   * factory method.
   *
   * @param num_cols The number of columns to wrap
   * @param columns An array of columns to copy to device memory
   *---------------------------------------------------------------------------**/
  device_table(gdf_size_type num_cols, gdf_column** columns,
               cudaStream_t stream = 0)
      : _num_columns(num_cols) {
    CUDF_EXPECTS(num_cols > 0, "Attempt to create table with zero columns.");
    CUDF_EXPECTS(nullptr != columns,
                 "Attempt to create table with a null column.");
    _num_rows = columns[0]->size;

    std::vector<gdf_column> temp_columns(num_cols);

    for (gdf_size_type i = 0; i < num_cols; ++i) {
      CUDF_EXPECTS(nullptr != columns[i], "Column is null");
      CUDF_EXPECTS(_num_rows == columns[i]->size, "Column size mismatch");
      if (_num_rows > 0) {
        CUDF_EXPECTS(nullptr != columns[i]->data, "Column missing data.");
      }
      temp_columns[i] = *columns[i];
    }

    // Copy columns to device
    RMM_ALLOC(&device_columns, num_cols * sizeof(gdf_column), stream);
    CUDA_TRY(cudaMemcpyAsync(device_columns, temp_columns.data(),
                             num_cols * sizeof(gdf_column),
                             cudaMemcpyHostToDevice, stream));
    CHECK_STREAM(stream);
  }
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
  __device__ inline hash_value_type operator()(gdf_column const& col,
                                               gdf_size_type row_index) {
    hash_function<col_type> hasher;

    // treat null values as the lowest possible value of the type
    col_type const value_to_hash =
        gdf_is_valid(col.valid, row_index)
            ? static_cast<col_type*>(col.data)[row_index]
            : std::numeric_limits<col_type>::lowest();

    return hasher(value_to_hash);
  }
};
}  // namespace

/**
 * --------------------------------------------------------------------------*
 * @brief Computes the hash value for a row in a table with an initial hash
 * value for each column.
 *
 * @note NULL values are treated as an implementation defined discrete value,
 * such that hashing any two NULL values in columns of the same type will return
 * the same hash value.
 *
 * @param[in] t The table whose row will be hashed
 * @param[in] row_index The row of the table to compute the hash value for
 * @param[in] initial_hash_values Array of initial hash values for each
 * column
 * @tparam hash_function The hash function that is used for each element in
 * the row, as well as combine hash values
 *
 * @return The hash value of the row
 * ----------------------------------------------------------------------------**/
template <template <typename> class hash_function = default_hash>
__device__ inline hash_value_type hash_row(
    device_table const& t, gdf_size_type row_index,
    hash_value_type const* __restrict__ initial_hash_values) {
  auto hash_combiner = [](hash_value_type lhs, hash_value_type rhs) {
    return hash_function<hash_value_type>{}.hash_combine(lhs, rhs);
  };

  // Hashes an element in a column and optionally combines it with an initial
  // hash value
  auto hasher = [row_index, &t, initial_hash_values,
                 hash_combiner](gdf_size_type column_index) {
    hash_value_type hash_value = cudf::type_dispatcher(
        t.get_column(column_index)->dtype, hash_element<hash_function>{},
        *t.get_column(column_index), row_index);

    hash_value = hash_combiner(initial_hash_values[column_index], hash_value);

    return hash_value;
  };

  // Hash each element and combine all the hash values together
  return thrust::transform_reduce(
      thrust::seq, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(t.num_columns()), hasher,
      hash_value_type{0}, hash_combiner);
}

/**
 * --------------------------------------------------------------------------*
 * @brief Computes the hash value for a row in a table
 *
 * @note NULL values are treated as an implementation defined discrete value,
 * such that hashing any two NULL values in columns of the same type will return
 * the same hash value.
 *
 * @param[in] t The table whose row will be hashed
 * @param[in] row_index The row of the table to compute the hash value for
 * @tparam hash_function The hash function that is used for each element in
 * the row, as well as combine hash values
 *
 * @return The hash value of the row
 * ----------------------------------------------------------------------------**/
template <template <typename> class hash_function = default_hash>
__device__ inline hash_value_type hash_row(device_table const& t,
                                           gdf_size_type row_index) {
  auto hash_combiner = [](hash_value_type lhs, hash_value_type rhs) {
    return hash_function<hash_value_type>{}.hash_combine(lhs, rhs);
  };

  // Hashes an element in a column
  auto hasher = [row_index, &t, hash_combiner](gdf_size_type column_index) {
    return cudf::type_dispatcher(t.get_column(column_index)->dtype,
                                 hash_element<hash_function>{},
                                 *t.get_column(column_index), row_index);
  };

  // Hash each element and combine all the hash values together
  return thrust::transform_reduce(
      thrust::seq, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(t.num_columns()), hasher,
      hash_value_type{0}, hash_combiner);
}

namespace {
struct copy_element {
  template <typename T>
  __device__ inline void operator()(gdf_column const& target,
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
 * FIXME: Does NOT set null bitmask for the target row.
 *
 * @param[in,out] The table whose row will be updated
 * @param[in] The index of the row to update in the target table
 * @param[in] source The table whose row will be copied
 * @param source_row_index The index of the row to copy in the source table
 */
__device__ inline void copy_row(device_table const& target,
                                gdf_size_type target_index,
                                device_table const& source,
                                gdf_size_type source_index) {
  for (gdf_size_type i = 0; i < target.num_columns(); ++i) {
    cudf::type_dispatcher(target.get_column(i)->dtype, copy_element{},
                          *target.get_column(i), target_index,
                          *source.get_column(i), source_index);
  }
}

#endif
