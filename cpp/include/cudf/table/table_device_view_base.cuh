/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

/**
 * @file table_device_view.cuh
 * @brief Table device view class definitions
 */

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Base class for a device table of `ColumnDeviceView`s
 *
 * This class should not be used directly. Instead:
 * - `table_device_view` should be used for a table of columns of type `column_device_view`
 * - `mutable_table_device_view` should be used for a table of columns of type
 * `mutable_column_device_view`
 *
 * All public constructors and member functions of `table_device_view_base` are
 * available in both `table_device_view` and `mutable_table_device_view`.
 *
 * @tparam ColumnDeviceView The type of column device view the table contains:
 *                          expects column_device_view or mutable_column_device_view
 * @tparam HostTableView The type of table view used to create the table device view:
 *                       expects table_view or mutable_table_view
 */
template <typename ColumnDeviceView>
class table_device_view_base {
 public:
  using column_type = ColumnDeviceView;  ///< Type of column device view the table contains

  table_device_view_base()                              = delete;
  ~table_device_view_base()                             = default;
  table_device_view_base(table_device_view_base const&) = default;  ///< Copy constructor
  table_device_view_base(table_device_view_base&&)      = default;  ///< Move constructor

  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  table_device_view_base& operator=(table_device_view_base const&) = default;

  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  table_device_view_base& operator=(table_device_view_base&&) = default;

  /**
   * @brief Returns an iterator to the first view in the `table`.
   *
   * @return An iterator to the first view in the `table`
   */
  __device__ column_type* begin() const noexcept { return _columns; }

  /**
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the `table`
   */
  __device__ column_type* end() const noexcept { return _columns + _num_columns; }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ column_type const& column(size_type column_index) const noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ column_type& column(size_type column_index) noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type num_columns() const noexcept { return _num_columns; }

  /**
   * @brief Returns the number of rows
   *
   * @return The number of rows
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type num_rows() const noexcept { return _num_rows; }

 protected:
  /// @brief Constructs a table device view from a host table view
  /// @param columns Array of column device views in device memory
  /// @param num_rows The number of rows in the table
  /// @param num_columns The number of columns in the table
  /// @param descendant_storage Pointer to device memory holding the descendant storage
  template <typename DescendantStorageType>
  CUDF_HOST_DEVICE table_device_view_base(column_type* columns,
                                          size_type num_rows,
                                          size_type num_columns,
                                          DescendantStorageType* descendant_storage)
    : _columns(columns),
      _num_rows(num_rows),
      _num_columns(num_columns),
      _descendant_storage(descendant_storage)
  {
  }

  /// @brief Returns a pointer to the descendant storage
  /// @tparam DescendantStorageType The type of the descendant storage
  /// @return Pointer to the descendant storage
  /// @note The type of the descendant storage must match the type used in the constructor
  /// @note This function is used to access the descendant storage in derived classes
  ///       that need to access the descendant storage directly.
  ///       It is not intended for general use.
  template <typename DescendantStorageType>
  CUDF_HOST_DEVICE DescendantStorageType* descendant_storage() const noexcept
  {
    return static_cast<DescendantStorageType*>(_descendant_storage);
  }

  ColumnDeviceView* _columns{};  ///< Array of view objects in device memory
  size_type _num_rows{};         ///< Number of rows in the table
  size_type _num_columns{};      ///< Number of columns in the table
  void* _descendant_storage{};   ///< Pointer to device memory holding the descendant storage
};

}  // namespace detail

/**
 * @brief Table device view that is usable in device memory
 */
class table_device_view_core : public detail::table_device_view_base<column_device_view_core> {
 public:
  using base =
    detail::table_device_view_base<column_device_view_core>;  ///< Base class for table device view
  using column_type =
    typename base::column_type;  ///< Type of column device view the table contains

 protected:
  /// @brief Constructs a table device view from a host table view
  /// @param columns Array of column device views in device memory
  /// @param num_rows The number of rows in the table
  /// @param num_columns The number of columns in the table
  /// @param descendant_storage Pointer to device memory holding the descendant storage
  template <typename DescendantStorageType>
  CUDF_HOST_DEVICE table_device_view_core(column_type* columns,
                                          size_type num_rows,
                                          size_type num_columns,
                                          DescendantStorageType* descendant_storage)
    : base(columns, num_rows, num_columns, descendant_storage)
  {
  }
};

/**
 * @brief Mutable table device view that is usable in device memory
 *
 * Elements of the table can be modified in device memory.
 */
class mutable_table_device_view_core
  : public detail::table_device_view_base<mutable_column_device_view_core> {
 public:
  using base =
    detail::table_device_view_base<mutable_column_device_view_core>;  ///< Base class for table
                                                                      ///< device view
  using column_type =
    typename base::column_type;  ///< Type of column device view the table contains

 protected:
  /// @brief Constructs a table device view from a host table view
  /// @param columns Array of column device views in device memory
  /// @param num_rows The number of rows in the table
  /// @param num_columns The number of columns in the table
  /// @param descendant_storage Pointer to device memory holding the descendant storage
  template <typename DescendantStorageType>
  CUDF_HOST_DEVICE mutable_table_device_view_core(column_type* columns,
                                                  size_type num_rows,
                                                  size_type num_columns,
                                                  DescendantStorageType* descendant_storage)
    : base(columns, num_rows, num_columns, descendant_storage)
  {
  }
};

}  // namespace CUDF_EXPORT cudf
