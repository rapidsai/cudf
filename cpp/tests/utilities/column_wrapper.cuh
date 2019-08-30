/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 William Malpica <william@blazingdb.com>
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

#include <cudf/cudf.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <utilities/bit_util.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <nvstrings/NVCategory.h>

#include <initializer_list>
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

namespace {

gdf_dtype_extra_info copy_extra_info(gdf_column const& column) {
  gdf_dtype_extra_info extra_info = column.dtype_info;

  // make a copy of the category if there is one
  if (column.dtype_info.category != nullptr) {
    extra_info.category =
        static_cast<NVCategory*>(column.dtype_info.category)->copy();
  }
  return extra_info;
}

};  // namespace

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Wrapper for a gdf_column used for unit testing.
 *
 *
 * The `column_wrapper<T>` class template is designed to simplify the creation
 * and management of `gdf_column`s for the purposes of unit testing.
 *
 * `column_wrapper<T>` provides a number of constructors that allow easily
 * constructing a `gdf_column` with the appropriate `gdf_dtype` enum set based
 * on mapping `T` to an enum, e.g., `column_wrapper<int>` will correspond to a
 * `gdf_column` whose `gdf_dtype` is set to `GDF_INT32`.
 *
 * The simplest constructor creates an unitilized `gdf_column` of a specified
 * type with a specified size:
 *
 * ```
 * cudf::test::column_wrapper<T>  col(size);
 * ```
 *
 * You can also construct a `gdf_column` that uses a `std::vector` to initialize
 * the `data` and `valid` bitmask of the `gdf_column`.
 *
 * ```
 *  std::vector<T> values(size);
 *
 *  std::vector<gdf_valid_type>
 *expected_bitmask(gdf_valid_allocation_size(size), 0xFF);
 *
 *  cudf::test::column_wrapper<T> const col(values, bitmask);
 * ```
 *
 * Another constructor allows passing in an initializer function that accepts a
 * row index that will be invoked for every index `[0, size)` in the column:
 *
 * ```
 *   // This creates a gdf_column with data elements {0, 1, ..., size-1} with a
 * valid bitmask
 *   // that indicates all of the values are non-null
 *   cudf::test::column_wrapper<T> col(size,
 *       [](auto row) { return row; },
 *       [](auto row) { return true; });
 * ```
 *
 * You can also construct a `column_wrapper<T>` using an initializer_list:
 *
 * ```
 * // Constructs a column with elements {1,2,3,4} and no bitmask
 * column_wrapper<T>{1,2,3,4};
 *
 * // Constructs a column with elements {1,2,3,4} and a bitmask
 * // where all elements are valid
 * column_wrapper<T>({1,2,3,4},[](auto row) { return true; })
 * ```
 *
 * To access the underlying `gdf_column` for passing into a libcudf function,
 * the `column_wrapper::get` function can be used to provide a pointer to the
 * underlying `gdf_column`.
 *
 * ```
 * column_wrapper<T> col(size);
 * gdf_column* gdf_col = col.get();
 * some_libcudf_function(gdf_col...);
 *
 * @tparam ColumnType The underlying data type of the column
 *---------------------------------------------------------------------------**/
template <typename ColumnType>
struct column_wrapper {
  /**---------------------------------------------------------------------------*
   * @brief Default constructor initializes an empty gdf_column with proper
   * dtype
   *
   *---------------------------------------------------------------------------**/
  column_wrapper() : the_column{} {
    the_column.dtype = cudf::gdf_dtype_of<ColumnType>();
  }

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
    the_column.dtype_info = copy_extra_info(other);
  }

  column_wrapper& operator=(column_wrapper<ColumnType> other) = delete;

  // column data and bitmask destroyed by device_vector dtor
  ~column_wrapper() {
    if (std::is_same<ColumnType, cudf::nvstring_category>::value) {
      if (nullptr != the_column.dtype_info.category) {
        NVCategory::destroy(
            reinterpret_cast<NVCategory*>(the_column.dtype_info.category));
        the_column.dtype_info.category = 0;
      }
    }
  }

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

  operator gdf_column&() { return the_column; };
  operator const gdf_column&() const { return the_column; };

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
   * @brief Construct a new column wrapper of a specified size with lambda data
   * initializer and optionally allocated bitmask.
   *
   * Optionally allocates a default-initialized bitmask (i.e., all bits are
   * null).
   *
   * @param column_size The desired size of the column
   * @param value_initalizer The unary lambda to initialize each value in the
   * column's data
   * @param allocate_bitmask Optionally allocate a zero-initialized bitmask
   *---------------------------------------------------------------------------**/
  template <typename ValueInitializerType>
  column_wrapper(gdf_size_type column_size,
                 ValueInitializerType value_initalizer,
                 bool allocate_bitmask = false) {
    std::vector<ColumnType> host_data(column_size);
    std::vector<gdf_valid_type> host_bitmask;

    for (gdf_index_type row = 0; row < column_size; ++row) {
      host_data[row] = value_initalizer(row);
    }

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
   * @brief Construct a new column wrapper using an initializer list for the
   *column's data.
   *
   * The bitmask is not allocated.
   *
   * @param list initializer_list to use for column's data
   *---------------------------------------------------------------------------**/
  column_wrapper(std::initializer_list<ColumnType> list)
      : column_wrapper{std::vector<ColumnType>(list)} {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using an already existing gdf_column*
   *
   * Constructs a column_wrapper using a gdf_column*. The data in gdf_column* is
   * copied over. The allocations in the original gdf_column* are not managed,
   * and wont be freed by the destruction of this column wrapper
   *
   * @param column The gdf_column* that contains the originating data
   *---------------------------------------------------------------------------**/
  column_wrapper(const gdf_column& column) {
    CUDF_EXPECTS(gdf_dtype_of<ColumnType>() == column.dtype,
                 "Type mismatch between column_wrapper and gdf_column");

    if (column.data != nullptr) {
      // Using device_vector::assign causes a segfault on CentOS7 when trying to
      // assign `wrapper` types. This is a workaround
      data.resize(column.size);
      CUDA_TRY(cudaMemcpy(data.data().get(), column.data,
                          sizeof(ColumnType) * column.size, cudaMemcpyDefault));
    }

    if (column.valid != nullptr) {
      bitmask.assign(column.valid,
                     column.valid + gdf_valid_allocation_size(column.size));
    }
    the_column.data = data.data().get();
    the_column.size = data.size();
    the_column.dtype = column.dtype;

    the_column.dtype_info = copy_extra_info(column);

    if (bitmask.size() > 0) {
      the_column.valid = bitmask.data().get();
    } else {
      the_column.valid = nullptr;
    }
    the_column.null_count = column.null_count;
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using host vector for column data and
   * lambda initializer for the bitmask.
   *
   * Constructs a column_wrapper using a std::vector for the host data.
   *
   * Allocates and initializes the column's bitmask using the specified
   * bit_initializer unary callable. Bit `i` in the column's bitmask will be
   * equal to `bit_initializer(i)`.
   * @param host_data The vector of data to use for the column
   * @param bit_initializer The unary callable to initialize the bitmask
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
        cudf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using an initializer list for the
   * column's data and a lambda initializer for the bitmask.
   *
   * Allocates and initializes the column's bitmask using the specified
   * bit_initializer unary callable. Bit `i` in the column's bitmask will be
   * equal to `bit_initializer(i)`.
   *
   * @param list initializer_list to use for column's data
   * @param bit_initializer The unary callable to initialize the bitmask
   *---------------------------------------------------------------------------**/
  template <typename BitInitializerType>
  column_wrapper(std::initializer_list<ColumnType> list,
                 BitInitializerType bit_initializer)
      : column_wrapper{std::vector<ColumnType>(list), bit_initializer} {}

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
        cudf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper using an array of strings for data
   * with unallocated bitmask.
   *
   * Constructs a column wrapper of type nvstring_category from an array of
   * string host data.
   *
   * The valid bitmask is not allocated nor initialized.
   *
   * @param column_size The desired size of the column
   * @param string_values The array of strings to initialize column category
   * values
   *---------------------------------------------------------------------------**/
  column_wrapper(gdf_size_type column_size,
                 char const ** string_values) {
    // Initialize the values and bitmask using the initializers
    std::vector<ColumnType> host_data(column_size);

    NVCategory* category = NVCategory::create_from_array(string_values,
                                                         column_size);
    gdf_nvstring_category *category_data = new gdf_nvstring_category[column_size];
    category->get_values(category_data, false);

    for (gdf_index_type row = 0; row < column_size; ++row) {
      host_data[row] = ColumnType{category_data[row]};
    }
    initialize_with_host_data(host_data);
    the_column.dtype_info.category = category;
    delete[] category_data;
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column wrapper from an array of strings and a
   * lambda initializer for the column's bitmask.
   *
   * Constructs a column wrapper of type nvstring_category from a std::vector of
   * string host data using a unary lambda to initialize the column's validity
   * bitmask.
   *
   * Bit `i` in the column's bitmask will be equal to `bit_initializer(i)`.
   *
   * @tparam BitInitializerType The type of the bit_initializer lambda
   * @param string_values The array of strings to initialize column category
   * values
   * @param column_size The desired size of the column
   * @param bit_initializer The unary lambda to initialize each bit in the
   * column's bitmask
   *---------------------------------------------------------------------------**/
  template <typename BitInitializerType>
  column_wrapper(gdf_size_type column_size,
                 char const ** string_values,
                 BitInitializerType bit_initializer) {
    const size_t num_masks = gdf_valid_allocation_size(column_size);

    // Initialize the values and bitmask using the initializers
    std::vector<ColumnType> host_data(column_size);
    std::vector<gdf_valid_type> host_bitmask(num_masks, 0);

    NVCategory* category = NVCategory::create_from_array(string_values,
                                                         column_size);
    gdf_nvstring_category *category_data = new gdf_nvstring_category[column_size];
    category->get_values(category_data, false);

    for (gdf_index_type row = 0; row < column_size; ++row) {
      host_data[row] = ColumnType{category_data[row]};

      if (true == bit_initializer(row)) {
        cudf::util::turn_bit_on(host_bitmask.data(), row);
      }
    }
    initialize_with_host_data(host_data, host_bitmask);
    the_column.dtype_info.category = category;
    delete[] category_data;
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
      // TODO Is there a nicer way to get a `std::vector` from a
      // device_vector?
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
    // TODO Move the implementation of `print_gdf_column` here once it's
    // removed from usage elsewhere
    print_gdf_column(&the_column);
  }

  gdf_size_type size() const { return the_column.size; }

  /**---------------------------------------------------------------------------*
   * @brief Prints the values of the underlying gdf_column to a string
   * 
   *---------------------------------------------------------------------------**/
  std::string to_str() const {
    std::ostringstream buffer;
    print_gdf_column(&the_column, 1, buffer);
    return buffer.str();
  }

  /**---------------------------------------------------------------------------*
   * @brief Compares this wrapper to a gdf_column for equality.
   *
   * Treats NULL == NULL
   *
   * @param rhs  The gdf_column to check for equality
   * @return true The two columns are equal
   * @return false The two columns are not equal
   *---------------------------------------------------------------------------**/
  bool operator==(gdf_column const& rhs) const {
    return gdf_equal_columns(the_column, rhs);
  }

  /**---------------------------------------------------------------------------*
   * @brief Compares this wrapper to another column_wrapper for equality.
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

  rmm::device_vector<ColumnType>& get_data() { return data; }

  rmm::device_vector<ColumnType> const& get_data() const { return data; }

  rmm::device_vector<gdf_valid_type>& get_bitmask() { return bitmask; }

  rmm::device_vector<gdf_valid_type> const& get_bitmask() const {
    return bitmask;
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
    gdf_dtype_extra_info extra_info{TIME_UNIT_NONE};
    extra_info.category = nullptr;
    gdf_column_view_augmented(&the_column, data.data().get(), nullptr,
                              data.size(), cudf::gdf_dtype_of<ColumnType>(),
                              0, extra_info);

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

  rmm::device_vector<ColumnType> data{};  ///< Container for the column's data

  // If the column's bitmask does not exist (doesn't contain null values), then
  // the size of this vector will be zero
  rmm::device_vector<gdf_valid_type>
      bitmask{};  ///< Container for the column's bitmask

  gdf_column the_column{};
};

}  // namespace test
}  // namespace cudf
#endif
