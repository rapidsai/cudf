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

#ifndef COLUMN_WRAPPER_H
#define COLUMN_WRAPPER_H

#include <utilities/bit_util.cuh>
#include <utilities/type_dispatcher.hpp>
#include <tests/utilities/cudf_test_utils.cuh>
#include <cudf.h>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <thrust/equal.h>
#include <thrust/logical.h>

#include <string>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  do {                                                                        \
    cudaError_t cudaStatus = (call);                                          \
    if (cudaSuccess != cudaStatus) {                                          \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
      exit(1);                                                                \
    }                                                                         \
  } while (0)
#endif

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Wrapper for a gdf_column used for unit testing.
 *
 * An abstraction on top of a gdf_column that provides functionality for
 * allocating, initializing, and otherwise managing gdf_columns for passing to
 * libcudf APIs in unit testing.
 *
 * @tparam ColumnType The underlying data type of the column
 *---------------------------------------------------------------------------**/
template <typename ColumnType>
struct column_wrapper {
  /**---------------------------------------------------------------------------*
   * @brief Copy constructor copies from another column_wrapper of the same
   * type.
   *
   * @param other The column_wraper to copy
   *---------------------------------------------------------------------------**/
  column_wrapper(column_wrapper<ColumnType> const& other)
      : data{other.data}, bitmask{other.bitmask}, the_column{other.the_column} {
    the_column.data = data.data().get();
    the_column.valid = bitmask.data().get();
  }

  column_wrapper& operator=(column_wrapper<ColumnType> other) = delete;

  ~column_wrapper() = default;

  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to a gdf_column pointer.
   *
   * Allows for implicit conversion of a column_wrapper to a pointer to its
   * underlying gdf_column.
   *
   * In this way, a column_wrapper can be passed directly into a libcudf API
   * and will be implicitly converted to a pointer to its underlying gdf_column
   * without the need to use the `get()` member.
   *
   * @return gdf_column* Pointer to the underlying gdf_column
   *---------------------------------------------------------------------------**/
  operator gdf_column*() { return &the_column; };

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper of a specified size with default
   * initialized data and optionally allocated bitmask.
   *
   * Constructs a column_wrapper of the specified size with default intialized
   * data.
   *
   * Optionally allocates a default-initialized bitmask (i.e., all bits are
   *null).
   *
   * @param column_size The desired size of the column
   * @param allocate_bitmask Optionally allocate a zero-initialized bitmask
   *---------------------------------------------------------------------------**/
  column_wrapper(gdf_size_type column_size, bool allocate_bitmask = false) {
    std::vector<ColumnType> host_data(column_size);
    std::vector<gdf_valid_type> host_bitmask;

    if (allocate_bitmask) {
      host_bitmask.resize(gdf_valid_allocation_size(column_size));
    }

    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using host vectors for data and
   * bitmask.
   *
   * Constructs a column_wrapper using a std::vector for the host data and valid
   * bitmasks.
   *
   * @param host_data The vector of data to use for the column
   * @param host_bitmask The validity bitmask to use for the column
   *---------------------------------------------------------------------------**/
  column_wrapper(std::vector<ColumnType> const& host_data,
                 std::vector<gdf_valid_type> const& host_bitmask) {
    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using host vector for data with
   * unallocated bitmask.
   *
   * Constructs a column_wrapper using a std::vector for the host data.
   *
   * The valid bitmask is not allocated nor initialized.
   *
   * @param host_data The vector of data to use for the column
   *---------------------------------------------------------------------------**/
  column_wrapper(std::vector<ColumnType> const& host_data) {
    initialize_with_host_data(host_data);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using host vector for column data and
   * lambda initializer for the bitmask.
   *
   * Constructs a column_wrapper using a std::vector for the host data.
   *
   * The valid bitmask is initialized using the specified bit_initializer unary
   * lambda that returns a bool. Bit `i` in the bitmask will be equal to
   * `bit_intiializer(i)`.
   *
   * @tparam BitInitializerType The type of the bit initializer unary lambda
   * @param host_data The vector of data to use for the column
   * @param bit_initializer The unary lambda to intialize each bit of the
   * bitmask
   *---------------------------------------------------------------------------**/
  template <typename BitInitializerType>
  column_wrapper(std::vector<ColumnType> const& host_data,
                 BitInitializerType bit_initializer) {
    const size_t num_masks = gdf_valid_allocation_size(host_data.size());
    const gdf_size_type num_rows{static_cast<gdf_size_type>(host_data.size())};

    // Initialize the valid mask for this column using the initializer
    std::vector<gdf_valid_type> host_bitmask(num_masks, 0);
    for (gdf_index_type row = 0; row < num_rows; ++row) {
      if (true == bit_initializer(row)) {
        gdf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using lambda initializers for both
   * the column's data and bitmask.
   *
   * Constructs a column wrapper using a unary lambda to initialize both the
   * column's data and validity bitmasks.
   *
   * Element `i` in the column's data will be equal to `value_initializer(i)`.
   *
   * Bit `i` in the column's bitmask will be equal to `bit_initializer(i)`.
   *
   * @tparam ValueInitializerType The type of the value_initializer lambda
   * @tparam BitInitializerType The type of the bit_initializer lambda
   * @param column_size The desired size of the column
   * @param value_initalizer The unary lambda to initialize each value in the
   * column's data
   * @param bit_initializer The unary lambda to initialize each bit in the
   * column's bitmask
   *---------------------------------------------------------------------------**/
  template <typename ValueInitializerType, typename BitInitializerType>
  column_wrapper(gdf_size_type column_size,
                 ValueInitializerType value_initalizer,
                 BitInitializerType bit_initializer) {
    const size_t num_masks = gdf_valid_allocation_size(column_size);

    // Initialize the values and bitmask using the initializers
    std::vector<ColumnType> host_data(column_size);
    std::vector<gdf_valid_type> host_bitmask(num_masks, 0);

    for (gdf_index_type row = 0; row < column_size; ++row) {
      host_data[row] = value_initalizer(row);

      if (true == bit_initializer(row)) {
        gdf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the underlying gdf_column.
   *
   *---------------------------------------------------------------------------**/
  gdf_column* get() { return &the_column; }
  gdf_column const* get() const { return &the_column; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the null count of the underlying column.
   *
   *---------------------------------------------------------------------------**/
  gdf_size_type null_count() const { return the_column.null_count; }

  /**---------------------------------------------------------------------------*
   * @brief Copies the underying gdf_column's data and bitmask to the host.
   *
   * Returns a tuple of two std::vectors. The first is the column's data, and
   * the second is the column's bitmask.
   *
   *---------------------------------------------------------------------------**/
  auto to_host() const {
    gdf_size_type const num_masks{gdf_valid_allocation_size(the_column.size)};
    std::vector<ColumnType> host_data;
    std::vector<gdf_valid_type> host_bitmask;

    if (nullptr != the_column.data) {
      // TODO Is there a nicer way to get a `std::vector` from a device_vector?
      host_data.resize(the_column.size);
      CUDA_RT_CALL(cudaMemcpy(host_data.data(), the_column.data,
                              the_column.size * sizeof(ColumnType),
                              cudaMemcpyDeviceToHost));
    }

    if (nullptr != the_column.valid) {
      host_bitmask.resize(num_masks);
      CUDA_RT_CALL(cudaMemcpy(host_bitmask.data(), the_column.valid,
                              num_masks * sizeof(gdf_valid_type),
                              cudaMemcpyDeviceToHost));
    }

    return std::make_tuple(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Prints the values of the underlying gdf_column.
   *
   *---------------------------------------------------------------------------**/
  void print() const {
    // TODO Move the implementation of `print_gdf_column` here once it's removed
    // from usage elsewhere
    print_gdf_column(&the_column);
  }


  gdf_size_type size() const{
      return the_column.size;
  }

  /**---------------------------------------------------------------------------*
   * @brief Functor for comparing if two elements between two gdf_columns are
   * equal.
   *
   *---------------------------------------------------------------------------**/
  struct elements_equal {
    gdf_column lhs_col;
    gdf_column rhs_col;
    bool nulls_are_equivalent;

    /**---------------------------------------------------------------------------*
     * @brief Constructs functor for comparing elements between two gdf_column's
     *
     * @param lhs The left column for comparison
     * @param rhs The right column for comparison
     * @param nulls_are_equal Desired behavior for whether or not nulls are
     * treated as equal to other nulls. Defaults to true.
     *---------------------------------------------------------------------------**/
    __host__ __device__ elements_equal(gdf_column lhs, gdf_column rhs,
                                       bool nulls_are_equal = true)
        : lhs_col{lhs}, rhs_col{rhs}, nulls_are_equivalent{nulls_are_equal} {}

    __device__ bool operator()(gdf_index_type row) {
      bool const lhs_is_valid{gdf_is_valid(lhs_col.valid, row)};
      bool const rhs_is_valid{gdf_is_valid(rhs_col.valid, row)};

      if (lhs_is_valid and rhs_is_valid) {
        return static_cast<ColumnType const*>(lhs_col.data)[row] ==
               static_cast<ColumnType const*>(rhs_col.data)[row];
      }

      // If one value is valid but the other is not
      if (lhs_is_valid != rhs_is_valid) {
        return false;
      }

      return nulls_are_equivalent;
    }
  };

  /**---------------------------------------------------------------------------*
   * @brief Compares if another column_wrapper is equal to this wrapper.
   *
   * Treats NULL == NULL
   *
   * @param rhs  The other column_wrapper to check for equality
   * @return true The two columns are equal
   * @return false The two columns are not equal
   *---------------------------------------------------------------------------**/
  bool operator==(column_wrapper<ColumnType> const& rhs) const {
      return *this == *rhs.get();
  }

  /**---------------------------------------------------------------------------*
   * @brief Compares if gdf_column is equal to this wrapper.
   *
   * Treats NULL == NULL
   *
   * @param rhs  The gdf_column to check for equality
   * @return true The two columns are equal
   * @return false The two columns are not equal
   *---------------------------------------------------------------------------**/
  bool operator==(gdf_column const& rhs) const {
    if (the_column.size != rhs.size) return false;
    if (the_column.dtype != rhs.dtype) return false;
    if (the_column.null_count != rhs.null_count) return false;
    if (the_column.dtype_info.time_unit != rhs.dtype_info.time_unit)
      return false;

    if (!(the_column.data && rhs.data))
      return false;  // if one is null but not both

    if (not thrust::all_of(rmm::exec_policy()->on(0),
                           thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(the_column.size),
                           elements_equal{the_column, rhs})) {
      return false;
    }

    CUDA_RT_CALL(cudaPeekAtLastError());

    return true;
  }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates and initializes the underyling gdf_column with host data.
   *
   * Creates a gdf_column and copies data from the host for it's data and
   * bitmask. Sets the corresponding dtype based on the column_wrapper's
   * ColumnType.
   *
   * @param host_data The vector of host data to copy to device for the column's
   * data
   * @param host_bitmask Optional vector of host data for the column's bitmask
   * to copy to device
   *---------------------------------------------------------------------------**/
  void initialize_with_host_data(
      std::vector<ColumnType> const& host_data,
      std::vector<gdf_valid_type> const& host_bitmask =
          std::vector<gdf_valid_type>{}) {
    // thrust::device_vector takes care of host to device copy assignment
    data = host_data;

    // Fill the gdf_column members
    the_column.data = data.data().get();
    the_column.size = data.size();
    the_column.dtype = cudf::gdf_dtype_of<ColumnType>();
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    the_column.dtype_info = extra_info;

    // If a validity bitmask vector was passed in, allocate device storage
    // and copy its contents from the host vector
    if (host_bitmask.size() > 0) {
      gdf_size_type const required_bitmask_size{
          gdf_valid_allocation_size(host_data.size())};

      if (host_bitmask.size() < static_cast<size_t>(required_bitmask_size)) {
        throw std::runtime_error("Insufficiently sized bitmask vector.");
      }
      bitmask = host_bitmask;
      the_column.valid = bitmask.data().get();
    } else {
      the_column.valid = nullptr;
    }

    set_null_count(the_column);
  }

  rmm::device_vector<ColumnType> data;  ///< Container for the column's data

  // If the column's bitmask does not exist (doesn't contain null values), then
  // the size of this vector will be zero
  rmm::device_vector<gdf_valid_type> bitmask;  ///< Container for the column's bitmask

  gdf_column the_column;
};

}  // namespace test
}  // namespace cudf
#endif
